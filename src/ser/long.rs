//! `LongFormat` and `LongDeltaFormat` — the fixed-stride-binary and
//! sequential-delta reference group formats (Java `LongFormat`,
//! `LongDeltaFormat`). Object side is a `Vec<i64>` (Java `long[]`, no boxing).

use super::util::{unzigzag, zigzag};
use super::{serializers, GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};

/// Checked offset math: a torn/oversize node must fail fast rather than wrap (D4).
#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("long group seek overflow")
}

/// `base + idx * width`, checked against overflow.
#[inline]
fn elem_off(base: usize, idx: usize, width: usize) -> Result<usize> {
    idx.checked_mul(width)
        .and_then(|o| base.checked_add(o))
        .ok_or_else(seek_overflow)
}

/// Helper: insert into a cloned vec.
fn vec_insert(g: &[i64], pos: usize, v: i64) -> Vec<i64> {
    let mut r = Vec::with_capacity(g.len() + 1);
    r.extend_from_slice(&g[..pos]);
    r.push(v);
    r.extend_from_slice(&g[pos..]);
    r
}

fn vec_delete(g: &[i64], pos: usize) -> Vec<i64> {
    let mut r = Vec::with_capacity(g.len() - 1);
    r.extend_from_slice(&g[..pos]);
    r.extend_from_slice(&g[pos + 1..]);
    r
}

/// `Arrays.binarySearch(long[])` semantics over a sorted `Vec<i64>`.
fn bsearch_i64(g: &[i64], key: i64) -> SearchResult {
    let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
    while lo <= hi {
        let mid = ((lo + hi) as usize) >> 1;
        let v = g[mid];
        if v < key {
            lo = mid as isize + 1;
        } else if v > key {
            hi = mid as isize - 1;
        } else {
            return Ok(mid);
        }
    }
    Err(lo as usize)
}

/// Fixed 8-byte BE stride; O(log n) true binary search over serialized bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct LongFormat;

pub static LONG_FORMAT: LongFormat = LongFormat;

impl GroupFormat for LongFormat {
    type Elem = i64;
    type Group = Vec<i64>;

    fn element(&self) -> &dyn Serializer<i64> {
        &serializers::LONG
    }
    fn empty(&self) -> Vec<i64> {
        Vec::new()
    }
    fn size(&self, g: &Vec<i64>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<i64>, pos: usize) -> i64 {
        g[pos]
    }
    fn search(&self, g: &Vec<i64>, key: &i64) -> SearchResult {
        bsearch_i64(g, *key)
    }
    fn insert(&self, g: &Vec<i64>, pos: usize, v: i64) -> Vec<i64> {
        vec_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<i64>, pos: usize, v: i64) -> Vec<i64> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<i64>, pos: usize) -> Vec<i64> {
        vec_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<i64>, from: usize, to: usize) -> Vec<i64> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[i64]) -> Vec<i64> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<i64>) {
        for &v in g {
            out.write_i64(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<i64>> {
        let mut r = Vec::new();
        r.try_reserve(count)?;
        for _ in 0..count {
            r.push(input.read_i64()?);
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &i64,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let k = *key;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("long group count too large"));
        }
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            input.seek(elem_off(start, mid, 8)?)?;
            let v = input.read_i64()?;
            if v == k {
                found = Some(mid);
                break;
            } else if v < k {
                lo = mid as isize + 1;
            } else {
                hi = mid as isize - 1;
            }
        }
        input.seek(elem_off(start, count, 8)?)?;
        Ok(found.map(Ok).unwrap_or(Err(lo as usize)))
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<i64> {
        let start = input.pos();
        if pos >= count {
            return Err(DbError::corrupt("long group index out of range"));
        }
        input.seek(elem_off(start, pos, 8)?)?;
        let v = input.read_i64()?;
        input.seek(elem_off(start, count, 8)?)?;
        Ok(v)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = i64> + 'a>> {
        if from > to || to > count {
            return Err(crate::error::DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}

/// Delta-packed longs: `packLong(zigzag(k0))`, then `packLong(zigzag(ki-ki-1))`.
/// Object side identical to [`LongFormat`]; byte side is a sequential decode
/// with early exit.
#[derive(Debug, Clone, Copy, Default)]
pub struct LongDeltaFormat;

pub static LONG_DELTA_FORMAT: LongDeltaFormat = LongDeltaFormat;

impl GroupFormat for LongDeltaFormat {
    type Elem = i64;
    type Group = Vec<i64>;

    fn element(&self) -> &dyn Serializer<i64> {
        &serializers::LONG
    }
    fn empty(&self) -> Vec<i64> {
        Vec::new()
    }
    fn size(&self, g: &Vec<i64>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<i64>, pos: usize) -> i64 {
        g[pos]
    }
    fn search(&self, g: &Vec<i64>, key: &i64) -> SearchResult {
        bsearch_i64(g, *key)
    }
    fn insert(&self, g: &Vec<i64>, pos: usize, v: i64) -> Vec<i64> {
        vec_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<i64>, pos: usize, v: i64) -> Vec<i64> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<i64>, pos: usize) -> Vec<i64> {
        vec_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<i64>, from: usize, to: usize) -> Vec<i64> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[i64]) -> Vec<i64> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<i64>) {
        let mut prev = 0i64;
        for &v in g {
            out.pack_long(zigzag(v.wrapping_sub(prev)));
            prev = v;
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<i64>> {
        let mut r = Vec::new();
        r.try_reserve(count)?;
        let mut v = 0i64;
        for _ in 0..count {
            v = v.wrapping_add(unzigzag(input.unpack_long()?));
            r.push(v);
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &i64,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let k = *key;
        let mut v = 0i64;
        for i in 0..count {
            v = v.wrapping_add(unzigzag(input.unpack_long()?));
            if v >= k {
                input.unpack_long_skip(count - i - 1)?; // leave input at group end
                return Ok(if v == k { Ok(i) } else { Err(i) });
            }
        }
        Ok(Err(count))
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<i64> {
        if pos >= count {
            return Err(DbError::corrupt("long group index out of range"));
        }
        let mut v = 0i64;
        for _ in 0..=pos {
            v = v.wrapping_add(unzigzag(input.unpack_long()?));
        }
        input.unpack_long_skip(count - pos - 1)?; // leave input at group end
        Ok(v)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = i64> + 'a>> {
        if from > to || to > count {
            return Err(crate::error::DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(LongDeltaCursor {
            input,
            count,
            to,
            idx: from,
            started: false,
            decoded: 0,
            acc: 0,
            cur: 0,
            exhausted: false,
        }))
    }
}

/// Single forward pass over the zigzag stream (O(n) full scan).
struct LongDeltaCursor<'a> {
    input: &'a mut dyn DataInput2,
    count: usize,
    to: usize,
    idx: usize,
    started: bool,
    decoded: usize, // elements consumed from the stream
    acc: i64,       // running value; after decoding element k, acc == element[k]
    cur: i64,
    exhausted: bool,
}

impl<'a> LongDeltaCursor<'a> {
    fn consume_to(&mut self, count: usize) -> Result<()> {
        while self.decoded < count {
            self.acc = self.acc.wrapping_add(unzigzag(self.input.unpack_long()?));
            self.decoded += 1;
        }
        Ok(())
    }
}

impl<'a> GroupCursor for LongDeltaCursor<'a> {
    type Elem = i64;
    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }
        if self.started {
            self.idx += 1;
        } else {
            self.started = true;
        }
        if self.idx >= self.to {
            self.exhausted = true;
            self.consume_to(self.count)?; // drain to group end
            return Ok(false);
        }
        let target = self.idx + 1;
        self.consume_to(target)?;
        self.cur = self.acc;
        Ok(true)
    }
    fn index(&self) -> usize {
        self.idx
    }
    fn value(&self) -> i64 {
        self.cur
    }
}
