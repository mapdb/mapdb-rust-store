//! `TupleFormat` — binary-capable group format for composite/tuple keys
//! (Java `TupleFormat` + `TupleComponent`). Each tuple is encoded to an
//! ORDER-PRESERVING (memcomparable) `Vec<u8>` by its [`TupleComponent`] schema,
//! so the UNSIGNED byte order of the encodings equals the logical tuple order.
//! The group is then stored and searched exactly like a `byte[][]` group
//! (blob + fixed-width i32 offset table, unsigned in-place compare with a
//! length tie-break, giving `(a) < (a,b)` for prefix tuples).
//!
//! ## Per-component encoding
//! - [`TupleComponent::Int`]/[`TupleComponent::Long`]: fixed-width big-endian
//!   with the sign bit flipped (`v ^ MIN`) — maps the signed range onto the
//!   unsigned range monotonically, so big-endian unsigned byte compare equals
//!   signed integer order; fixed width makes the component self-delimiting.
//! - [`TupleComponent::Str`]/[`TupleComponent::Bytes`]: escaped-terminated —
//!   each `0x00` payload byte becomes `0x00 0xFF`, every other byte verbatim,
//!   then a `0x00 0x00` terminator. Order-preserving under unsigned compare and
//!   prefix-free. STRING order is UTF-8 (code-point) order, which deliberately
//!   differs from UTF-16 order for supplementary characters.

use super::value::Value;
use super::{BinaryGetCursor, GroupCursor, GroupFormat, SearchResult, Serializer};
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2, SliceInput};
use std::cmp::Ordering;

/// Checked offset math: a torn/oversize node must fail fast rather than wrap (D4).
#[inline]
fn seek_overflow() -> DbError {
    DbError::corrupt("tuple group seek overflow")
}

/// `base + idx * width`, checked against overflow.
#[inline]
fn elem_off(base: usize, idx: usize, width: usize) -> Result<usize> {
    idx.checked_mul(width)
        .and_then(|o| base.checked_add(o))
        .ok_or_else(seek_overflow)
}

/// A typed component of a composite key providing an order-preserving
/// (memcomparable) per-component codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TupleComponent {
    /// Signed 32-bit int, 4-byte big-endian, sign bit flipped.
    Int,
    /// Signed 64-bit long, 8-byte big-endian, sign bit flipped.
    Long,
    /// UTF-8 string, escaped-terminated; UTF-8 unsigned (code-point) order.
    Str,
    /// Raw bytes, escaped-terminated; unsigned lexicographic order.
    Bytes,
}

impl TupleComponent {
    /// Append the memcomparable encoding of `value` to `out`. Panics on a value
    /// whose variant does not match this component (a caller bug, mirroring
    /// Java's `ClassCastException`).
    fn encode(self, out: &mut DataOutput2, value: &Value) {
        match (self, value) {
            (TupleComponent::Int, Value::Int(v)) => out.write_i32(v ^ i32::MIN),
            (TupleComponent::Long, Value::Long(v)) => out.write_i64(v ^ i64::MIN),
            (TupleComponent::Str, Value::Str(s)) => write_escaped(out, s.as_bytes()),
            (TupleComponent::Bytes, Value::Bytes(b)) => write_escaped(out, b),
            _ => panic!("tuple component type mismatch: {self:?} vs {value:?}"),
        }
    }

    /// Read exactly one component from `input`, which must not advance past
    /// `end` (the exclusive end of the tuple's encoded bytes).
    fn decode(self, input: &mut dyn DataInput2, end: usize) -> Result<Value> {
        match self {
            TupleComponent::Int => {
                if input.pos() + 4 > end {
                    return Err(DbError::corrupt("corrupt tuple: truncated int component"));
                }
                Ok(Value::Int(input.read_i32()? ^ i32::MIN))
            }
            TupleComponent::Long => {
                if input.pos() + 8 > end {
                    return Err(DbError::corrupt("corrupt tuple: truncated long component"));
                }
                Ok(Value::Long(input.read_i64()? ^ i64::MIN))
            }
            TupleComponent::Str => {
                let bytes = read_escaped(input, end)?;
                Ok(Value::Str(String::from_utf8_lossy(&bytes).into_owned()))
            }
            TupleComponent::Bytes => Ok(Value::Bytes(read_escaped(input, end)?)),
        }
    }

    /// Logical order of this component, equal to the unsigned byte order of
    /// [`encode`](Self::encode). Panics on a value whose variant does not match.
    fn compare(self, a: &Value, b: &Value) -> Ordering {
        match (self, a, b) {
            (TupleComponent::Int, Value::Int(x), Value::Int(y)) => x.cmp(y),
            (TupleComponent::Long, Value::Long(x), Value::Long(y)) => x.cmp(y),
            // UTF-8 unsigned byte order == Unicode code-point order.
            (TupleComponent::Str, Value::Str(x), Value::Str(y)) => x.as_bytes().cmp(y.as_bytes()),
            // Unsigned lexicographic (Vec<u8> Ord is unsigned + length).
            (TupleComponent::Bytes, Value::Bytes(x), Value::Bytes(y)) => x.cmp(y),
            _ => panic!("tuple component type mismatch in compare: {self:?}"),
        }
    }

    /// Logical (value-based) equality.
    fn equal_to(self, a: &Value, b: &Value) -> bool {
        self.compare(a, b) == Ordering::Equal
    }
}

// ---- escaped-terminated codec for variable-length components ----

fn write_escaped(out: &mut DataOutput2, payload: &[u8]) {
    for &b in payload {
        if b == 0x00 {
            out.write_u8(0x00);
            out.write_u8(0xFF);
        } else {
            out.write_u8(b);
        }
    }
    // terminator 0x00 0x00 (never produced by an escaped 0x00, which is 0x00 0xFF)
    out.write_u8(0x00);
    out.write_u8(0x00);
}

fn read_escaped(input: &mut dyn DataInput2, end: usize) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        if input.pos() >= end {
            return Err(DbError::corrupt("corrupt tuple: unterminated component"));
        }
        let b = input.read_u8()?;
        if b == 0x00 {
            if input.pos() >= end {
                return Err(DbError::corrupt("corrupt tuple: dangling escape"));
            }
            let b2 = input.read_u8()?;
            if b2 == 0x00 {
                break; // terminator
            }
            if b2 != 0xFF {
                return Err(DbError::corrupt("corrupt tuple: bad escape"));
            }
            buf.push(0x00); // 0x00 0xFF -> literal 0x00
        } else {
            buf.push(b);
        }
    }
    Ok(buf)
}

// ---- per-tuple memcomparable codec (free fns shared by format + serializer) ----

/// Append the memcomparable encoding of `tuple` to a fresh buffer. Panics if the
/// arity exceeds the schema (a caller bug, mirroring Java's `IllegalArgumentException`).
fn encode_tuple(schema: &[TupleComponent], tuple: &[Value]) -> Vec<u8> {
    assert!(
        tuple.len() <= schema.len(),
        "tuple arity {} exceeds schema {}",
        tuple.len(),
        schema.len()
    );
    let mut out = DataOutput2::new();
    for (i, v) in tuple.iter().enumerate() {
        schema[i].encode(&mut out, v);
    }
    out.into_vec()
}

/// Decode a tuple from its memcomparable bytes; recovers arity by consuming
/// components until the encoded bytes are exhausted.
fn decode_tuple(schema: &[TupleComponent], enc: &[u8]) -> Result<Vec<Value>> {
    let end = enc.len();
    let mut input = SliceInput::new(enc);
    let mut r: Vec<Value> = Vec::new();
    let mut i = 0usize;
    while input.pos() < end {
        if i == schema.len() {
            return Err(DbError::corrupt(
                "corrupt tuple: more components than schema",
            ));
        }
        r.push(schema[i].decode(&mut input, end)?);
        i += 1;
    }
    Ok(r)
}

/// Component-wise tuple order; on an equal shared prefix the SHORTER tuple is
/// smaller (`(a) < (a,b)`).
fn compare_tuple(schema: &[TupleComponent], a: &[Value], b: &[Value]) -> Ordering {
    let min = a.len().min(b.len());
    for i in 0..min {
        let c = schema[i].compare(&a[i], &b[i]);
        if c != Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

/// A tuple format over an ordered component schema (arity = length).
#[derive(Clone)]
pub struct TupleFormat {
    serializer: TupleSerializer,
}

impl TupleFormat {
    /// Build a tuple format over the given ordered component types. Panics on an
    /// empty schema (mirroring Java's `IllegalArgumentException`).
    pub fn of(components: &[TupleComponent]) -> TupleFormat {
        assert!(
            !components.is_empty(),
            "tuple schema must have at least one component"
        );
        TupleFormat {
            serializer: TupleSerializer {
                schema: components.to_vec(),
            },
        }
    }

    #[inline]
    fn schema_slice(&self) -> &[TupleComponent] {
        &self.serializer.schema
    }

    /// Defensive copy of this format's persisted component schema (Java
    /// `TupleFormat.schema()`). The internal `Vec` is not exposed, so callers
    /// cannot mutate the live schema.
    pub fn schema(&self) -> Vec<TupleComponent> {
        self.serializer.schema.clone()
    }
}

impl GroupFormat for TupleFormat {
    type Elem = Vec<Value>;
    /// Group holds each tuple's memcomparable encoding (== Java's `byte[][]`).
    type Group = Vec<Vec<u8>>;

    fn element(&self) -> &dyn Serializer<Vec<Value>> {
        &self.serializer
    }

    // ---- object side (group == encoded byte tuples) ----

    fn empty(&self) -> Vec<Vec<u8>> {
        Vec::new()
    }
    fn size(&self, g: &Vec<Vec<u8>>) -> usize {
        g.len()
    }
    fn get(&self, g: &Vec<Vec<u8>>, pos: usize) -> Vec<Value> {
        decode_tuple(self.schema_slice(), &g[pos]).expect("corrupt stored tuple in get")
    }
    fn search(&self, g: &Vec<Vec<u8>>, key: &Vec<Value>) -> SearchResult {
        let probe = encode_tuple(self.schema_slice(), key);
        let (mut lo, mut hi) = (0isize, g.len() as isize - 1);
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            match g[mid].cmp(&probe) {
                Ordering::Equal => return Ok(mid),
                Ordering::Less => lo = mid as isize + 1,
                Ordering::Greater => hi = mid as isize - 1,
            }
        }
        Err(lo as usize)
    }
    fn insert(&self, g: &Vec<Vec<u8>>, pos: usize, v: Vec<Value>) -> Vec<Vec<u8>> {
        let e = encode_tuple(self.schema_slice(), &v);
        let mut r = Vec::with_capacity(g.len() + 1);
        r.extend_from_slice(&g[..pos]);
        r.push(e);
        r.extend_from_slice(&g[pos..]);
        r
    }
    fn set(&self, g: &Vec<Vec<u8>>, pos: usize, v: Vec<Value>) -> Vec<Vec<u8>> {
        let mut r = g.clone();
        r[pos] = encode_tuple(self.schema_slice(), &v);
        r
    }
    fn delete(&self, g: &Vec<Vec<u8>>, pos: usize) -> Vec<Vec<u8>> {
        let mut r = Vec::with_capacity(g.len() - 1);
        r.extend_from_slice(&g[..pos]);
        r.extend_from_slice(&g[pos + 1..]);
        r
    }
    fn copy_range(&self, g: &Vec<Vec<u8>>, from: usize, to: usize) -> Vec<Vec<u8>> {
        g[from..to].to_vec()
    }
    fn from_slice(&self, values: &[Vec<Value>]) -> Vec<Vec<u8>> {
        values
            .iter()
            .map(|t| encode_tuple(self.schema_slice(), t))
            .collect()
    }

    // ---- wire (blob + fixed-width i32 offset table) ----

    fn serialize(&self, out: &mut DataOutput2, g: &Vec<Vec<u8>>) {
        let blob_len: usize = g.iter().map(|e| e.len()).sum();
        out.write_i32(blob_len as i32);
        let mut off = 0i32;
        for e in g {
            out.write_i32(off);
            off += e.len() as i32;
        }
        for e in g {
            out.write_all(e);
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Vec<Vec<u8>>> {
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt tuple group blobLen"));
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
                return Err(DbError::corrupt("corrupt tuple group offsets"));
            }
            r.push(blob[s as usize..e as usize].to_vec());
        }
        Ok(r)
    }

    // ---- byte side ----

    fn supports_binary(&self) -> bool {
        true
    }

    fn binary_search(
        &self,
        key: &Vec<Value>,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        let probe = encode_tuple(self.schema_slice(), key);
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt tuple group blobLen"));
        }
        let blob_len = blob_len as usize;
        if count > isize::MAX as usize {
            return Err(DbError::corrupt("tuple group count too large"));
        }
        let off_base = start.checked_add(4).ok_or_else(seek_overflow)?;
        let blob_base = elem_off(off_base, count, 4)?;
        let blob_end = blob_base.checked_add(blob_len).ok_or_else(seek_overflow)?;
        let (mut lo, mut hi) = (0isize, count as isize - 1);
        let mut found: Option<usize> = None;
        while lo <= hi {
            let mid = ((lo + hi) as usize) >> 1;
            let c = compare_stored_to(input, off_base, blob_base, blob_len, count, mid, &probe)?;
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

    fn binary_get(
        &self,
        input: &mut dyn DataInput2,
        count: usize,
        pos: usize,
    ) -> Result<Vec<Value>> {
        let start = input.pos();
        let blob_len = input.read_i32()?;
        if blob_len < 0 {
            return Err(DbError::corrupt("corrupt tuple group blobLen"));
        }
        let blob_len = blob_len as usize;
        if pos >= count {
            return Err(DbError::corrupt("tuple group index out of range"));
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
            return Err(DbError::corrupt("corrupt tuple group offsets"));
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
        input.seek(blob_end)?; // leave input at group end
        decode_tuple(self.schema_slice(), &bytes)
    }

    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Vec<Value>> + 'a>> {
        if from > to || to > count {
            return Err(DbError::corrupt("range_cursor bounds"));
        }
        Ok(Box::new(BinaryGetCursor::new(self, input, count, from, to)))
    }
}

/// Unsigned-lexicographic compare of stored element `idx` against `probe`, sign
/// convention `stored - probe`, comparing in place (no allocation). Offset
/// corruption clamps ported verbatim from `ByteArrayFormat`.
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
        return Err(DbError::corrupt("corrupt tuple group offsets"));
    }
    let stored_len = (e - s) as usize;
    input.seek(
        blob_base
            .checked_add(s as usize)
            .ok_or_else(seek_overflow)?,
    )?;
    let n = stored_len.min(probe.len());
    for &pb in probe.iter().take(n) {
        let sb = input.read_u8()?;
        if sb != pb {
            // unsigned compare (both u8)
            return Ok(sb.cmp(&pb));
        }
    }
    Ok(stored_len.cmp(&probe.len()))
}

/// Self-delimiting standalone codec for a single tuple: `packInt(len) + encoded`.
#[derive(Clone)]
struct TupleSerializer {
    schema: Vec<TupleComponent>,
}

impl Serializer<Vec<Value>> for TupleSerializer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<Value>) {
        let e = encode_tuple(&self.schema, value);
        out.pack_int(e.len() as i32);
        out.write_all(&e);
    }

    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<Value>> {
        let len = input.unpack_int()?;
        if len < 0 {
            return Err(DbError::corrupt("corrupt tuple length"));
        }
        let mut e = Vec::new();
        e.try_reserve(len as usize)?;
        e.resize(len as usize, 0);
        input.read_fully(&mut e)?;
        decode_tuple(&self.schema, &e)
    }

    fn compare(&self, a: &Vec<Value>, b: &Vec<Value>) -> Ordering {
        compare_tuple(&self.schema, a, b)
    }

    fn equals(&self, a: &Vec<Value>, b: &Vec<Value>) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .enumerate()
            .all(|(i, (x, y))| self.schema[i].equal_to(x, y))
    }

    fn equals_by_serialized_bytes(&self) -> bool {
        true // memcomparable encoding is canonical
    }

    fn natural_order(&self) -> bool {
        false // tuples are not natural-Comparable
    }
}
