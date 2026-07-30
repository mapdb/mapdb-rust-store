//! `StringGroupFormat` — length-prefixed blob with a fixed-width per-element
//! offset index; the blob+offset-table reference format (Java
//! `StringGroupFormat`). Byte side binary-searches over stored UTF-8 in place
//! via [`compare_utf8`](super::util::compare_utf8).
//!
//! Wire: `i32 blobLen; i32 off[n]; blob[blobLen]`.

use super::util::compare_utf8;
use super::{serializers, GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

/// Checked offset math: a torn/oversize node must fail fast rather than wrap (D4).
#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("string group seek overflow")
}

/// `base + idx * width`, checked against overflow.
#[inline]
fn elem_off(base: usize, idx: usize, width: usize) -> Result<usize> {
    idx.checked_mul(width)
        .and_then(|o| base.checked_add(o))
        .ok_or_else(seek_overflow)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StringGroupFormat;

pub static STRING_GROUP_FORMAT: StringGroupFormat = StringGroupFormat;

fn bsearch_str(g: &[String], key: &str) -> SearchResult {
    let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
    while lo <= hi {
        let mid = ((lo + hi) as usize) >> 1;
        match super::util::compare_utf16(&g[mid], key) {
            Ordering::Equal => return Ok(mid),
            Ordering::Less => lo = mid as isize + 1,
            Ordering::Greater => hi = mid as isize - 1,
        }
    }
    Err(lo as usize)
}

impl GroupFormat for StringGroupFormat {
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
        let enc: Vec<&[u8]> = g.iter().map(|s| s.as_bytes()).collect();
        let blob_len: usize = enc.iter().map(|e| e.len()).sum();
        out.write_i32(blob_len as i32);
        let mut off = 0i32;
        for e in &enc {
            out.write_i32(off);
            off += e.len() as i32;
        }
        for e in &enc {
            out.write_all(e);
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<String>> {
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt string group blobLen"));
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
                return Err(DbError::corrupt("corrupt string group offsets"));
            }
            r.push(String::from_utf8_lossy(&blob[s as usize..e as usize]).into_owned());
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
            return Err(DbError::corrupt("corrupt string group blobLen"));
        }
        let blob_len = blob_len as usize;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("string group count too large"));
        }
        let off_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(off_base, count, 4)?;
        let blob_end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
        let key16: Vec<u16> = key.encode_utf16().collect();
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            input.seek(elem_off(off_base, mid, 4)?)?;
            let s = input.read_i32()?;
            let e = if mid + 1 < count {
                input.read_i32()?
            } else {
                blob_len as i32
            };
            if s < 0 || e < s || e as usize > blob_len {
                return Err(DbError::corrupt("corrupt string group offsets"));
            }
            input.seek(
                blob_base
                    .checked_add(s as usize)
                    .ok_or_else(seek_overflow)?,
            )?;
            let c = compare_utf8(input, (e - s) as usize, &key16)?;
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

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<String> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt string group blobLen"));
        }
        let blob_len = blob_len as usize;
        if pos >= count {
            return Err(DbError::corrupt("string group index out of range"));
        }
        let off_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(off_base, count, 4)?;
        let blob_end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
        // element pos (offsets are contiguous, so `e` reads the next slot in place)
        input.seek(elem_off(off_base, pos, 4)?)?;
        let s = input.read_i32()?;
        let e = if pos + 1 < count {
            input.read_i32()?
        } else {
            blob_len as i32
        };
        if s < 0 || e < s || e as usize > blob_len {
            return Err(DbError::corrupt("corrupt string group offsets"));
        }
        input.seek(
            blob_base
                .checked_add(s as usize)
                .ok_or_else(seek_overflow)?,
        )?;
        let mut bytes = Vec::new();
        bytes.try_reserve((e - s) as usize)?;
        bytes.resize((e - s) as usize, 0);
        input.read_fully(&mut bytes)?;
        let out = String::from_utf8_lossy(&bytes).into_owned();
        input.seek(blob_end)?; // leave input at group end
        Ok(out)
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
