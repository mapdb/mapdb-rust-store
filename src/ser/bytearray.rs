//! Binary-capable group formats for `byte[]` keys (Java `ByteArrayFormat` and
//! `ByteArrayPrefixFormat`, rules R6/R7).
//!
//! Both use `Group = Vec<Vec<u8>>` and order by **UNSIGNED** lexicographic
//! (`memcmp`, `Arrays.compareUnsigned`) on BOTH sides — the element serializer is
//! [`ByteArrayUnsignedSer`](serializers::ByteArrayUnsignedSer), so
//! `element().compare == search order == binary_search order`. Slice `<[u8]>::cmp`
//! is exactly unsigned lexicographic, so the byte side compares stored bytes in
//! place with sign convention `stored - probe` (never via UTF-8 like the string
//! formats).
//!
//! [`ByteArrayFormat`] wire: `i32 blobLen; i32 off[n]; blob[blobLen]` (blob+offset
//! table, mirrors `StringGroupFormat`). [`ByteArrayPrefixFormat`] wire:
//! `i32 blobLen; i32 restartOff[ceil(n/K)]; blob` where the blob holds entries
//! `packInt(shared) packInt(suffixLen) suffix`, front-coded with RESTART every
//! `K = RESTART_INTERVAL` entries (mirrors `StringPrefixFormat`).

use super::util::common_prefix_len;
use super::{serializers, GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

/// Restart every K entries; matches `StringPrefixFormat` / LevelDB default.
const RESTART_INTERVAL: usize = 16;

/// Checked offset math: a torn/oversize node must fail fast rather than wrap (D4).
#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("byte[] group seek overflow")
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

/// Unsigned binary search over a materialized group (`Arrays.compareUnsigned`).
fn bsearch_unsigned(g: &[Vec<u8>], key: &[u8]) -> SearchResult {
    let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
    while lo <= hi {
        let mid = ((lo + hi) as usize) >> 1;
        match g[mid].as_slice().cmp(key) {
            Ordering::Equal => return Ok(mid),
            Ordering::Less => lo = mid as isize + 1,
            Ordering::Greater => hi = mid as isize - 1,
        }
    }
    Err(lo as usize)
}

// ---- shared object-side copy-on-write helpers ----

fn obj_insert(g: &[Vec<u8>], pos: usize, v: Vec<u8>) -> Vec<Vec<u8>> {
    let mut r = Vec::with_capacity(g.len() + 1);
    r.extend_from_slice(&g[..pos]);
    r.push(v);
    r.extend_from_slice(&g[pos..]);
    r
}

fn obj_set(g: &[Vec<u8>], pos: usize, v: Vec<u8>) -> Vec<Vec<u8>> {
    let mut r = g.to_vec();
    r[pos] = v;
    r
}

fn obj_delete(g: &[Vec<u8>], pos: usize) -> Vec<Vec<u8>> {
    let mut r = Vec::with_capacity(g.len() - 1);
    r.extend_from_slice(&g[..pos]);
    r.extend_from_slice(&g[pos + 1..]);
    r
}

// =====================================================================
// ByteArrayFormat — blob + fixed-width i32 offset table
// =====================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct ByteArrayFormat;

pub static BYTE_ARRAY_FORMAT: ByteArrayFormat = ByteArrayFormat;

/// Unsigned-lexicographic compare of stored element `idx` against `probe`, sign
/// convention `stored - probe`, comparing in place — no allocation.
fn compare_stored_to(
    input: &mut dyn DataInput2,
    off_base: usize,
    blob_base: usize,
    blob_len: usize,
    count: usize,
    idx: usize,
    probe: &[u8],
) -> Result<Ordering> {
    input.seek(elem_off(off_base, idx, 4)?)?;
    let s = input.read_i32()?;
    let e = if idx + 1 < count {
        input.read_i32()?
    } else {
        blob_len as i32
    };
    if s < 0 || e < s || e as usize > blob_len {
        return Err(DbError::corrupt("corrupt byte[] group offsets"));
    }
    let stored_len = (e - s) as usize;
    input.seek(
        blob_base
            .checked_add(s as usize)
            .ok_or_else(seek_overflow)?,
    )?;
    let n = stored_len.min(probe.len());
    for &p in probe.iter().take(n) {
        let c = input.read_u8()? as i32 - p as i32;
        if c != 0 {
            return Ok(if c < 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
    }
    Ok(stored_len.cmp(&probe.len()))
}

impl GroupFormat for ByteArrayFormat {
    type Elem = Vec<u8>;
    type Group = Vec<Vec<u8>>;

    fn element(&self) -> &dyn Serializer<Vec<u8>> {
        &serializers::BYTE_ARRAY_UNSIGNED
    }
    fn empty(&self) -> Vec<Vec<u8>> {
        Vec::new()
    }
    fn size(&self, g: &Vec<Vec<u8>>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<Vec<u8>>, pos: usize) -> Vec<u8> {
        g[pos].clone()
    }
    fn search(&self, g: &Vec<Vec<u8>>, key: &Vec<u8>) -> SearchResult {
        bsearch_unsigned(g, key)
    }
    fn insert(&self, g: &Vec<Vec<u8>>, pos: usize, v: Vec<u8>) -> Vec<Vec<u8>> {
        obj_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<Vec<u8>>, pos: usize, v: Vec<u8>) -> Vec<Vec<u8>> {
        obj_set(g, pos, v)
    }
    fn delete(&self, g: &Vec<Vec<u8>>, pos: usize) -> Vec<Vec<u8>> {
        obj_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<Vec<u8>>, from: usize, to: usize) -> Vec<Vec<u8>> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[Vec<u8>]) -> Vec<Vec<u8>> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<Vec<u8>>) {
        let blob_len: usize = g.iter().map(|b| b.len()).sum();
        out.write_i32(blob_len as i32);
        let mut off = 0i32;
        for b in g {
            out.write_i32(off);
            off += b.len() as i32;
        }
        for b in g {
            out.write_all(b);
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<Vec<u8>>> {
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt byte[] group blobLen"));
        }
        let blob_len = blob_len as usize;
        let mut off = Vec::new();
        off.try_reserve(count)?;
        for _ in 0..count {
            off.push(input.read_i32()?);
        }
        let mut blob = Vec::new();
        blob.try_reserve(blob_len)?;
        blob.resize(blob_len, 0);
        input.read_fully(&mut blob)?;
        let mut r = Vec::new();
        r.try_reserve(count)?;
        for i in 0..count {
            let s = off[i];
            let e = if i + 1 < count {
                off[i + 1]
            } else {
                blob_len as i32
            };
            if s < 0 || e < s || e as usize > blob_len {
                return Err(DbError::corrupt("corrupt byte[] group offsets"));
            }
            r.push(blob[s as usize..e as usize].to_vec());
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &Vec<u8>,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt byte[] group blobLen"));
        }
        let blob_len = blob_len as usize;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("byte[] group count too large"));
        }
        let off_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(off_base, count, 4)?;
        let blob_end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            let c = compare_stored_to(input, off_base, blob_base, blob_len, count, mid, key)?;
            match c {
                Ordering::Equal => {
                    found = Some(mid);
                    break;
                }
                Ordering::Less => lo = mid as isize + 1,
                Ordering::Greater => hi = mid as isize - 1,
            }
        }
        input.seek(blob_end)?; // leave input at group end
        Ok(found.map(Ok).unwrap_or(Err(lo as usize)))
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<Vec<u8>> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt byte[] group blobLen"));
        }
        let blob_len = blob_len as usize;
        if pos >= count {
            return Err(DbError::corrupt("byte[] group index out of range"));
        }
        let off_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(off_base, count, 4)?;
        let blob_end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
        input.seek(elem_off(off_base, pos, 4)?)?;
        let s = input.read_i32()?;
        let e = if pos + 1 < count {
            input.read_i32()?
        } else {
            blob_len as i32
        };
        if s < 0 || e < s || e as usize > blob_len {
            return Err(DbError::corrupt("corrupt byte[] group offsets"));
        }
        input.seek(
            blob_base
                .checked_add(s as usize)
                .ok_or_else(seek_overflow)?,
        )?;
        let mut b = Vec::new();
        b.try_reserve((e - s) as usize)?;
        b.resize((e - s) as usize, 0);
        input.read_fully(&mut b)?;
        input.seek(blob_end)?; // leave input at group end
        Ok(b)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Vec<u8>> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}

// =====================================================================
// ByteArrayPrefixFormat — front-coded (prefix-compressed) with restarts
// =====================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct ByteArrayPrefixFormat;

pub static BYTE_ARRAY_PREFIX_FORMAT: ByteArrayPrefixFormat = ByteArrayPrefixFormat;

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
        return Err(DbError::corrupt(
            "corrupt byte[] prefix group restart offset",
        ));
    }
    input.seek(
        blob_base
            .checked_add(off as usize)
            .ok_or_else(seek_overflow)?,
    )?;
    Ok(())
}

/// Decode one entry into `scratch` (which holds the previous entry's bytes on
/// entry; restart entries must carry `shared == 0`). Every length is clamped:
/// garbage errors, allocation is bounded by the blob.
fn read_entry(
    input: &mut dyn DataInput2,
    scratch: &mut Vec<u8>,
    end: usize,
    restart: bool,
) -> Result<()> {
    let shared = input.unpack_int()?;
    if shared < 0
        || (if restart {
            shared != 0
        } else {
            shared as usize > scratch.len()
        })
    {
        return Err(DbError::corrupt("corrupt byte[] prefix group sharedLen"));
    }
    let shared = shared as usize;
    let suffix_len = input.unpack_int()?;
    // subtraction form: pos() + suffixLen could overflow on garbage. If pos > end
    // the remainder is "negative", so any suffixLen errors (matches Java).
    let rem = end.checked_sub(input.pos()).filter(|_| suffix_len >= 0);
    match rem {
        Some(rem) if (suffix_len as usize) <= rem => {}
        _ => return Err(DbError::corrupt("corrupt byte[] prefix group suffixLen")),
    }
    let suffix_len = suffix_len as usize;
    let new_len = shared + suffix_len; // bounded: shared <= scratch.len, suffixLen <= blob remainder
    scratch.truncate(shared); // keep the shared prefix from the previous entry
    scratch.try_reserve(suffix_len)?;
    scratch.resize(new_len, 0);
    input.read_fully(&mut scratch[shared..new_len])?;
    Ok(())
}

/// Unsigned-lexicographic compare of restart entry `r`'s stored bytes against
/// `key`, sign convention `stored - key`, in place — no copy. Restart entries
/// carry `shared == 0`, so the stored suffix IS the full key.
fn compare_restart(
    input: &mut dyn DataInput2,
    rest_base: usize,
    blob_base: usize,
    blob_len: usize,
    r: usize,
    key: &[u8],
) -> Result<Ordering> {
    seek_restart(input, rest_base, blob_base, blob_len, r)?;
    let shared = input.unpack_int()?;
    if shared != 0 {
        return Err(DbError::corrupt(
            "corrupt byte[] prefix group restart sharedLen",
        ));
    }
    let len = input.unpack_int()?;
    let end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
    let rem = end.checked_sub(input.pos()).filter(|_| len >= 0);
    match rem {
        Some(rem) if (len as usize) <= rem => {}
        _ => {
            return Err(DbError::corrupt(
                "corrupt byte[] prefix group restart suffixLen",
            ))
        }
    }
    let len = len as usize;
    let n = len.min(key.len());
    for &k in key.iter().take(n) {
        let c = input.read_u8()? as i32 - k as i32;
        if c != 0 {
            return Ok(if c < 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
    }
    Ok(len.cmp(&key.len()))
}

impl GroupFormat for ByteArrayPrefixFormat {
    type Elem = Vec<u8>;
    type Group = Vec<Vec<u8>>;

    fn element(&self) -> &dyn Serializer<Vec<u8>> {
        &serializers::BYTE_ARRAY_UNSIGNED
    }
    fn empty(&self) -> Vec<Vec<u8>> {
        Vec::new()
    }
    fn size(&self, g: &Vec<Vec<u8>>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<Vec<u8>>, pos: usize) -> Vec<u8> {
        g[pos].clone()
    }
    fn search(&self, g: &Vec<Vec<u8>>, key: &Vec<u8>) -> SearchResult {
        bsearch_unsigned(g, key)
    }
    fn insert(&self, g: &Vec<Vec<u8>>, pos: usize, v: Vec<u8>) -> Vec<Vec<u8>> {
        obj_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<Vec<u8>>, pos: usize, v: Vec<u8>) -> Vec<Vec<u8>> {
        obj_set(g, pos, v)
    }
    fn delete(&self, g: &Vec<Vec<u8>>, pos: usize) -> Vec<Vec<u8>> {
        obj_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<Vec<u8>>, from: usize, to: usize) -> Vec<Vec<u8>> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[Vec<u8>]) -> Vec<Vec<u8>> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<Vec<u8>>) {
        let n = g.len();
        let n_rest = n.div_ceil(RESTART_INTERVAL);
        let mut rest_off = vec![0i32; n_rest];
        let mut blob = DataOutput2::with_capacity((n * 8).max(16));
        let mut prev: &[u8] = &[];
        for (i, enc) in g.iter().enumerate() {
            let shared = if i % RESTART_INTERVAL == 0 {
                rest_off[i / RESTART_INTERVAL] = blob.pos() as i32;
                0
            } else {
                common_prefix_len(prev, enc)
            };
            blob.pack_int(shared as i32);
            blob.pack_int((enc.len() - shared) as i32);
            blob.write_all(&enc[shared..]);
            prev = enc;
        }
        out.write_i32(blob.pos() as i32);
        for off in &rest_off {
            out.write_i32(*off);
        }
        out.write_all(&blob.buf);
    }

    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<Vec<u8>>> {
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt byte[] prefix group blobLen"));
        }
        let blob_len = blob_len as usize;
        let n_rest = n_restarts(count);
        input.skip_bytes(n_rest.checked_mul(4).ok_or_else(seek_overflow)?)?; // sequential decode does not need the restart table
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
            r.push(cur.clone());
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &Vec<u8>,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt byte[] prefix group blobLen"));
        }
        let blob_len = blob_len as usize;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("byte[] prefix group count too large"));
        }
        let n_rest = n_restarts(count);
        let rest_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(rest_base, n_rest, 4)?;
        let end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;

        // 1. binary search the restarts for the RIGHTMOST restart entry <= key,
        //    comparing the stored bytes in place (restarts have shared == 0)
        let (mut lo, mut hi) = (0isize, n_rest as isize - 1);
        let mut r: isize = -1;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            let c = compare_restart(input, rest_base, blob_base, blob_len, mid, key)?;
            if c != Ordering::Greater {
                r = mid as isize;
                lo = mid as isize + 1;
            } else {
                hi = mid as isize - 1;
            }
        }
        if r < 0 {
            input.seek(end)?; // key sorts below the first entry
            return Ok(Err(0));
        }

        // 2. roll forward through interval r (<= K entries), reconstructing incrementally
        let r = r as usize;
        let first = r * RESTART_INTERVAL;
        let limit = (first + RESTART_INTERVAL).min(count);
        seek_restart(input, rest_base, blob_base, blob_len, r)?;
        let mut scratch: Vec<u8> = Vec::new();
        let mut result: SearchResult = Err(limit); // key above the whole interval
        for i in first..limit {
            read_entry(input, &mut scratch, end, i == first)?;
            match scratch.as_slice().cmp(key.as_slice()) {
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

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<Vec<u8>> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt byte[] prefix group blobLen"));
        }
        let blob_len = blob_len as usize;
        if pos >= count {
            return Err(DbError::corrupt("byte[] prefix group index out of range"));
        }
        let n_rest = n_restarts(count);
        let rest_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(rest_base, n_rest, 4)?;
        let end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;

        let r = pos / RESTART_INTERVAL;
        seek_restart(input, rest_base, blob_base, blob_len, r)?;
        let mut scratch: Vec<u8> = Vec::new();
        let first = r * RESTART_INTERVAL;
        for i in first..=pos {
            read_entry(input, &mut scratch, end, i == first)?;
        }
        input.seek(end)?;
        Ok(scratch)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Vec<u8>> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}
