//! `ShortFormat`, `CharFormat`, `UUIDFormat` — three fixed-stride scalar group
//! formats (Java `ShortFormat`, `CharFormat`, `UUIDFormat`). Each is a stride
//! sibling of [`super::long::LongFormat`]: a packed array of fixed-width
//! big-endian elements giving O(log n) true binary search over serialized bytes.
//!
//! The three differ only in stride and ordering:
//! - `Short` → `i16`, 2-byte BE stride, **signed** order — the stored bytes must
//!   be DECODED (sign-extended) before comparing; raw BE bytes memcmp the
//!   negative half after the non-negative half, which is wrong.
//! - `Character` → `u16`, 2-byte BE stride, **unsigned** order — here raw BE
//!   bytes happen to memcmp-order correctly, but we still decode for uniformity.
//! - `UUID` → [`Uuid`] (`{msb:i64, lsb:i64}`), 16-byte BE stride, **signed**
//!   `(msb, then lsb)` order (Java `UUID.compareTo`) — each half is decoded as a
//!   signed long; the high-bit-set msb half sorts BELOW the clear-bit half.

use super::serializers::Uuid;
use super::{serializers, GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

/// Checked offset math: a torn/oversize node must fail fast rather than wrap (D4).
#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("scalar group seek overflow")
}

/// `base + idx * width`, checked against overflow.
#[inline]
fn elem_off(base: usize, idx: usize, width: usize) -> Result<usize> {
    idx.checked_mul(width)
        .and_then(|o| base.checked_add(o))
        .ok_or_else(seek_overflow)
}

/// `Arrays.binarySearch` semantics over a sorted slice using the type's `Ord`.
fn bsearch<T: Copy + Ord>(g: &[T], key: T) -> SearchResult {
    let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
    while lo <= hi {
        let mid = ((lo + hi) as usize) >> 1;
        match g[mid].cmp(&key) {
            Ordering::Less => lo = mid as isize + 1,
            Ordering::Greater => hi = mid as isize - 1,
            Ordering::Equal => return Ok(mid),
        }
    }
    Err(lo as usize)
}

fn vec_insert<T: Copy>(g: &[T], pos: usize, v: T) -> Vec<T> {
    let mut r = Vec::with_capacity(g.len() + 1);
    r.extend_from_slice(&g[..pos]);
    r.push(v);
    r.extend_from_slice(&g[pos..]);
    r
}

fn vec_delete<T: Copy>(g: &[T], pos: usize) -> Vec<T> {
    let mut r = Vec::with_capacity(g.len() - 1);
    r.extend_from_slice(&g[..pos]);
    r.extend_from_slice(&g[pos + 1..]);
    r
}

// ---------------------------------------------------------------------------
// ShortFormat: i16, 2-byte BE stride, SIGNED order.
// ---------------------------------------------------------------------------

/// Fixed 2-byte BE stride over signed shorts; O(log n) binary search over bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShortFormat;

pub static SHORT_FORMAT: ShortFormat = ShortFormat;

impl GroupFormat for ShortFormat {
    type Elem = i16;
    type Group = Vec<i16>;

    fn element(&self) -> &dyn Serializer<i16> {
        &serializers::SHORT
    }
    fn empty(&self) -> Vec<i16> {
        Vec::new()
    }
    fn size(&self, g: &Vec<i16>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<i16>, pos: usize) -> i16 {
        g[pos]
    }
    fn search(&self, g: &Vec<i16>, key: &i16) -> SearchResult {
        bsearch(g, *key)
    }
    fn insert(&self, g: &Vec<i16>, pos: usize, v: i16) -> Vec<i16> {
        vec_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<i16>, pos: usize, v: i16) -> Vec<i16> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<i16>, pos: usize) -> Vec<i16> {
        vec_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<i16>, from: usize, to: usize) -> Vec<i16> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[i16]) -> Vec<i16> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<i16>) {
        for &v in g {
            out.write_i16(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<i16>> {
        let mut r = Vec::new();
        r.try_reserve(count)?;
        for _ in 0..count {
            r.push(input.read_i16()?);
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &i16,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let k = *key;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("scalar group count too large"));
        }
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            input.seek(elem_off(start, mid, 2)?)?;
            let v = input.read_i16()?; // sign-extended, signed compare
            if v == k {
                found = Some(mid);
                break;
            } else if v < k {
                lo = mid as isize + 1;
            } else {
                hi = mid as isize - 1;
            }
        }
        input.seek(elem_off(start, count, 2)?)?;
        Ok(found.map(Ok).unwrap_or(Err(lo as usize)))
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<i16> {
        let start = input.pos();
        if pos >= count {
            return Err(DbError::corrupt("scalar group index out of range"));
        }
        input.seek(elem_off(start, pos, 2)?)?;
        let v = input.read_i16()?;
        input.seek(elem_off(start, count, 2)?)?;
        Ok(v)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = i16> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}

// ---------------------------------------------------------------------------
// CharFormat: u16, 2-byte BE stride, UNSIGNED order.
// ---------------------------------------------------------------------------

/// Fixed 2-byte BE stride over unsigned chars; O(log n) binary search over bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharFormat;

pub static CHAR_FORMAT: CharFormat = CharFormat;

impl GroupFormat for CharFormat {
    type Elem = u16;
    type Group = Vec<u16>;

    fn element(&self) -> &dyn Serializer<u16> {
        &serializers::CHAR
    }
    fn empty(&self) -> Vec<u16> {
        Vec::new()
    }
    fn size(&self, g: &Vec<u16>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<u16>, pos: usize) -> u16 {
        g[pos]
    }
    fn search(&self, g: &Vec<u16>, key: &u16) -> SearchResult {
        bsearch(g, *key)
    }
    fn insert(&self, g: &Vec<u16>, pos: usize, v: u16) -> Vec<u16> {
        vec_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<u16>, pos: usize, v: u16) -> Vec<u16> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<u16>, pos: usize) -> Vec<u16> {
        vec_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<u16>, from: usize, to: usize) -> Vec<u16> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[u16]) -> Vec<u16> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<u16>) {
        for &v in g {
            out.write_u16(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<u16>> {
        let mut r = Vec::new();
        r.try_reserve(count)?;
        for _ in 0..count {
            r.push(input.read_u16()?);
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &u16,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let k = *key;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("scalar group count too large"));
        }
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            input.seek(elem_off(start, mid, 2)?)?;
            let v = input.read_u16()?; // zero-extended, unsigned compare
            if v == k {
                found = Some(mid);
                break;
            } else if v < k {
                lo = mid as isize + 1;
            } else {
                hi = mid as isize - 1;
            }
        }
        input.seek(elem_off(start, count, 2)?)?;
        Ok(found.map(Ok).unwrap_or(Err(lo as usize)))
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<u16> {
        let start = input.pos();
        if pos >= count {
            return Err(DbError::corrupt("scalar group index out of range"));
        }
        input.seek(elem_off(start, pos, 2)?)?;
        let v = input.read_u16()?;
        input.seek(elem_off(start, count, 2)?)?;
        Ok(v)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = u16> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}

// ---------------------------------------------------------------------------
// UUIDFormat: Uuid{msb,lsb}, 16-byte BE stride, SIGNED (msb, then lsb) order.
// ---------------------------------------------------------------------------

/// Fixed 16-byte stride (`msb` then `lsb`, each a BE signed long); O(log n)
/// binary search over bytes. Order is signed `(msb, then lsb)` (`UUID.compareTo`).
#[derive(Debug, Clone, Copy, Default)]
pub struct UuidFormat;

pub static UUID_FORMAT: UuidFormat = UuidFormat;

impl GroupFormat for UuidFormat {
    type Elem = Uuid;
    type Group = Vec<Uuid>;

    fn element(&self) -> &dyn Serializer<Uuid> {
        &serializers::UUID
    }
    fn empty(&self) -> Vec<Uuid> {
        Vec::new()
    }
    fn size(&self, g: &Vec<Uuid>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<Uuid>, pos: usize) -> Uuid {
        g[pos]
    }
    fn search(&self, g: &Vec<Uuid>, key: &Uuid) -> SearchResult {
        bsearch(g, *key)
    }
    fn insert(&self, g: &Vec<Uuid>, pos: usize, v: Uuid) -> Vec<Uuid> {
        vec_insert(g, pos, v)
    }
    fn set(&self, g: &Vec<Uuid>, pos: usize, v: Uuid) -> Vec<Uuid> {
        let mut r = g.clone();
        r[pos] = v;
        r
    }
    fn delete(&self, g: &Vec<Uuid>, pos: usize) -> Vec<Uuid> {
        vec_delete(g, pos)
    }
    fn copy_range(&self, g: &Vec<Uuid>, from: usize, to: usize) -> Vec<Uuid> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[Uuid]) -> Vec<Uuid> {
        values.to_vec()
    }

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<Uuid>) {
        for u in g {
            out.write_i64(u.msb);
            out.write_i64(u.lsb);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<Uuid>> {
        let mut r = Vec::new();
        r.try_reserve(count)?;
        for _ in 0..count {
            let msb = input.read_i64()?;
            let lsb = input.read_i64()?;
            r.push(Uuid { msb, lsb });
        }
        Ok(r)
    }

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &Uuid,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let start = input.pos();
        let k = *key;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("scalar group count too large"));
        }
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            input.seek(elem_off(start, mid, 16)?)?;
            let msb = input.read_i64()?;
            let lsb = input.read_i64()?;
            let v = Uuid { msb, lsb };
            match v.cmp(&k) {
                Ordering::Equal => {
                    found = Some(mid);
                    break;
                }
                Ordering::Less => lo = mid as isize + 1,
                Ordering::Greater => hi = mid as isize - 1,
            }
        }
        input.seek(elem_off(start, count, 16)?)?;
        Ok(found.map(Ok).unwrap_or(Err(lo as usize)))
    }

    fn binary_get(&self, input: &mut dyn DataInput2, count: usize, pos: usize) -> Result<Uuid> {
        let start = input.pos();
        if pos >= count {
            return Err(DbError::corrupt("scalar group index out of range"));
        }
        input.seek(elem_off(start, pos, 16)?)?;
        let msb = input.read_i64()?;
        let lsb = input.read_i64()?;
        input.seek(elem_off(start, count, 16)?)?;
        Ok(Uuid { msb, lsb })
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Uuid> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(super::BinaryGetCursor::new(
            self, input, count, from, to,
        )))
    }
}
