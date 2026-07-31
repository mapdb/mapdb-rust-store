//! `StringPrefixFormat` — front-coded (shared-prefix compressed) group format for
//! `String`, the LevelDB/RocksDB block style. Periodic
//! RESTART points keep the byte side O(log n) without materializing the group.
//!
//! Every entry is `packInt(sharedPrefixLen) packInt(suffixLen) suffix_utf8`, where
//! `sharedPrefixLen` is the byte length shared with the PREVIOUS entry's UTF-8
//! form. Entry `i` with `i % K == 0` is a restart: `sharedPrefixLen` is forced to
//! 0 so it decodes without history and is addressable through the restart table.
//!
//! Wire layout for a group of `n` elements (`n` from the node header;
//! `nRestarts = ceil(n/K)`):
//! `i32 blobLen; i32 restartOff[nRestarts]; byte blob[blobLen]`.
//!
//! Order is `String.compareTo` (UTF-16 code-unit order), matched on the byte side
//! by [`compare_utf8`](super::util::compare_utf8). `binary_search` binary-searches
//! the restart entries in place, then rolls forward through at most K entries of
//! one interval, reconstructing incrementally into a scratch `Vec<u8>`. Every
//! length is clamped: garbage fails fast as `Err`, allocation stays
//! bounded by the blob, and every loop is bounded by `n` or `K`.

use super::util::{common_prefix_len, compare_utf16, compare_utf8};
use super::{serializers, GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2, SliceInput};
use std::cmp::Ordering;

/// Restart every K entries (LevelDB default; two intervals at maxNodeSize 32).
const RESTART_INTERVAL: usize = 16;

/// Checked offset math: a torn/oversize node must fail fast rather than wrap (D4).
#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("prefix group seek overflow")
}

/// `base + idx * width`, checked against overflow.
#[inline]
fn elem_off(base: usize, idx: usize, width: usize) -> Result<usize> {
    idx.checked_mul(width)
        .and_then(|o| base.checked_add(o))
        .ok_or_else(seek_overflow)
}

/// `ceil(count / K)` restart count, overflow-free (untrusted `count`).
#[inline]
fn n_restarts(count: usize) -> usize {
    count / RESTART_INTERVAL + !count.is_multiple_of(RESTART_INTERVAL) as usize
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StringPrefixFormat;

pub static STRING_PREFIX_FORMAT: StringPrefixFormat = StringPrefixFormat;

fn bsearch_str(g: &[String], key: &str) -> SearchResult {
    let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
    while lo <= hi {
        let mid = ((lo + hi) as usize) >> 1;
        match compare_utf16(&g[mid], key) {
            Ordering::Equal => return Ok(mid),
            Ordering::Less => lo = mid as isize + 1,
            Ordering::Greater => hi = mid as isize - 1,
        }
    }
    Err(lo as usize)
}

/// Position `input` at restart `r`'s entry, validating the offset.
fn seek_restart(
    input: &mut dyn DataInput2,
    rest_base: usize,
    blob_base: usize,
    blob_len: usize,
    r: usize,
) -> Result<()> {
    input.seek(elem_off(rest_base, r, 4)?)?;
    let off = input.read_i32()?;
    if off < 0 || off as usize > blob_len {
        return Err(DbError::corrupt("corrupt prefix group restart offset"));
    }
    input.seek(
        blob_base
            .checked_add(off as usize)
            .ok_or_else(seek_overflow)?,
    )?;
    Ok(())
}

/// Decode one entry into `scratch` (`scratch` holds the previous entry's bytes on
/// entry; restart entries must carry shared == 0). Every length is clamped.
fn read_entry(
    input: &mut dyn DataInput2,
    scratch: &mut Vec<u8>,
    end: usize,
    restart: bool,
) -> Result<()> {
    let cur_len = scratch.len();
    let shared = input.unpack_int()?;
    if shared < 0
        || (if restart {
            shared != 0
        } else {
            shared as usize > cur_len
        })
    {
        return Err(DbError::corrupt("corrupt prefix group sharedLen"));
    }
    let shared = shared as usize;
    let suffix_len = input.unpack_int()?;
    // subtraction form: pos() + suffixLen could overflow on garbage. If pos > end
    // the remainder is "negative", so any suffixLen errors (matches Java).
    let rem = end.checked_sub(input.pos()).filter(|_| suffix_len >= 0);
    match rem {
        Some(rem) if (suffix_len as usize) <= rem => {}
        _ => return Err(DbError::corrupt("corrupt prefix group suffixLen")),
    }
    let suffix_len = suffix_len as usize;
    let new_len = shared + suffix_len; // bounded: shared <= cur_len, suffixLen <= blob remainder
    scratch.try_reserve(new_len.saturating_sub(scratch.capacity()))?;
    scratch.resize(new_len, 0); // preserves [0..shared]
    input.read_fully(&mut scratch[shared..new_len])?;
    Ok(())
}

/// Compare restart entry `r` against `key16` in place — no copy, no String.
fn compare_restart(
    input: &mut dyn DataInput2,
    rest_base: usize,
    blob_base: usize,
    blob_len: usize,
    r: usize,
    key16: &[u16],
) -> Result<Ordering> {
    seek_restart(input, rest_base, blob_base, blob_len, r)?;
    let shared = input.unpack_int()?;
    if shared != 0 {
        return Err(DbError::corrupt("corrupt prefix group restart sharedLen"));
    }
    let len = input.unpack_int()?;
    // subtraction form: pos() + len could overflow on garbage. If pos > end the
    // remainder is "negative", so any len errors (matches Java).
    let end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
    let rem = end.checked_sub(input.pos()).filter(|_| len >= 0);
    match rem {
        Some(rem) if (len as usize) <= rem => {}
        _ => return Err(DbError::corrupt("corrupt prefix group restart suffixLen")),
    }
    compare_utf8(input, len as usize, key16)
}

impl GroupFormat for StringPrefixFormat {
    type Elem = String;
    type Group = Vec<String>;

    fn element(&self) -> &dyn Serializer<String> {
        &serializers::STRING
    }
    fn empty(&self) -> Vec<String> {
        Vec::new()
    }
    fn size(&self, g: &Vec<String>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<String>, pos: usize) -> String {
        g[pos].clone()
    }
    fn search(&self, g: &Vec<String>, key: &String) -> SearchResult {
        bsearch_str(g, key)
    }
    fn insert(&self, g: &Vec<String>, pos: usize, v: String) -> Vec<String> {
        let mut r = Vec::with_capacity(g.len() + 1);
        r.extend_from_slice(&g[..pos]);
        r.push(v);
        r.extend_from_slice(&g[pos..]);
        r
    }
    fn set(&self, g: &Vec<String>, pos: usize, v: String) -> Vec<String> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<String>, pos: usize) -> Vec<String> {
        let mut r = Vec::with_capacity(g.len() - 1);
        r.extend_from_slice(&g[..pos]);
        r.extend_from_slice(&g[pos + 1..]);
        r
    }
    fn copy_range(&self, g: &Vec<String>, from: usize, to: usize) -> Vec<String> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[String]) -> Vec<String> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<String>) {
        let n = g.len();
        let nrest = n.div_ceil(RESTART_INTERVAL);
        let mut rest_off = vec![0i32; nrest];
        let mut blob = DataOutput2::with_capacity((n * 8).max(16));
        for i in 0..n {
            let enc = g[i].as_bytes();
            let shared = if i % RESTART_INTERVAL == 0 {
                rest_off[i / RESTART_INTERVAL] = blob.pos() as i32;
                0
            } else {
                common_prefix_len(g[i - 1].as_bytes(), enc)
            };
            blob.pack_int(shared as i32);
            blob.pack_int((enc.len() - shared) as i32);
            blob.write_all(&enc[shared..]);
        }
        out.write_i32(blob.pos() as i32);
        for &off in &rest_off {
            out.write_i32(off);
        }
        out.write_all(&blob.buf);
    }

    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<String>> {
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt prefix group blobLen"));
        }
        let blob_len = blob_len as usize;
        let nrest = n_restarts(count);
        input.skip_bytes(nrest.checked_mul(4).ok_or_else(seek_overflow)?)?; // sequential decode does not need the restart table
        let end = input
            .pos()
            .checked_add(blob_len)
            .ok_or_else(seek_overflow)?; // blob end; every entry must decode within it
        let mut r = Vec::new();
        r.try_reserve(count)?;
        let mut cur: Vec<u8> = Vec::new();
        for i in 0..count {
            let restart = i % RESTART_INTERVAL == 0;
            read_entry(input, &mut cur, end, restart)?;
            // Lossy materialization (Java `new String(bytes, UTF_8)`); spec 01 §3.
            r.push(String::from_utf8_lossy(&cur).into_owned());
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &String,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt prefix group blobLen"));
        }
        let blob_len = blob_len as usize;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("prefix group count too large"));
        }
        let nrest = n_restarts(count);
        let rest_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(rest_base, nrest, 4)?;
        let end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
        let key16: Vec<u16> = key.encode_utf16().collect();

        // 1. binary search the restarts for the RIGHTMOST restart entry <= key,
        //    comparing the stored UTF-8 in place (restarts have shared == 0)
        let (mut lo, mut hi) = (0isize, nrest as isize - 1);
        let mut r: isize = -1;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            let c = compare_restart(input, rest_base, blob_base, blob_len, mid, &key16)?;
            if c != Ordering::Greater {
                r = mid as isize;
                lo = mid as isize + 1;
            } else {
                hi = mid as isize - 1;
            }
        }
        if r < 0 {
            // key sorts below the first entry
            input.seek(end)?;
            return Ok(Err(0));
        }
        let r = r as usize;

        // 2. roll forward through interval r (<= K entries), reconstructing incrementally
        let first = r * RESTART_INTERVAL;
        let limit = (first + RESTART_INTERVAL).min(count);
        seek_restart(input, rest_base, blob_base, blob_len, r)?;
        let mut scratch: Vec<u8> = Vec::new();
        // default: key above the whole interval (and below restart r+1, if any)
        let mut result: SearchResult = Err(limit);
        for i in first..limit {
            read_entry(input, &mut scratch, end, i == first)?;
            let c = compare_utf8(&mut SliceInput::new(&scratch), scratch.len(), &key16)?;
            match c {
                Ordering::Equal => {
                    result = Ok(i);
                    break;
                }
                Ordering::Greater => {
                    result = Err(i);
                    break;
                }
                Ordering::Less => {}
            }
        }
        input.seek(end)?;
        Ok(result)
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<String> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt prefix group blobLen"));
        }
        let blob_len = blob_len as usize;
        if pos >= count {
            return Err(DbError::corrupt("prefix group index out of range"));
        }
        let nrest = n_restarts(count);
        let rest_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(rest_base, nrest, 4)?;
        let end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;

        let r = pos / RESTART_INTERVAL;
        seek_restart(input, rest_base, blob_base, blob_len, r)?;
        let mut scratch: Vec<u8> = Vec::new();
        let first = r * RESTART_INTERVAL;
        for i in first..=pos {
            read_entry(input, &mut scratch, end, i == first)?;
        }
        input.seek(end)?;
        Ok(String::from_utf8_lossy(&scratch).into_owned())
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = String> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}
