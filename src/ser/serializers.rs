//! Built-in element serializers (Java `Serializers`, spec 01 §2).
//!
//! Java → Rust type map: `Short`→`i16`, `Character`→`u16`, `Integer`→`i32`,
//! `Long`→`i64`, `UUID`→[`Uuid`], `String`→`String`, `byte[]`→`Vec<u8>`.
//! All built-ins declare `equals_by_serialized_bytes() == true`.

use super::Serializer;
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use std::cmp::Ordering;

/// Read a packed length prefix and validate it against the record before it is
/// used to size an allocation (D4). Java `new byte[len]`
/// throws on a negative length rather than treating it as a huge positive one;
/// we reject `len < 0` and any `len` beyond the bytes left in the record, so a
/// tiny corrupt record cannot provoke a multi-gigabyte reservation.
fn read_framed_len(input: &mut dyn DataInput2) -> Result<usize> {
    let raw = input.unpack_int()?;
    if raw < 0 {
        return Err(DbError::corrupt("negative length prefix"));
    }
    let len = raw as usize;
    if len > input.remaining() {
        return Err(DbError::corrupt("length prefix exceeds record"));
    }
    Ok(len)
}

/// 16-byte UUID: `msb` then `lsb`, each a big-endian **signed** long. Order is
/// signed msb-then-lsb (Java `UUID.compareTo`), not unsigned/lexicographic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Uuid {
    pub msb: i64,
    pub lsb: i64,
}

impl Uuid {
    pub fn new(msb: i64, lsb: i64) -> Self {
        Self { msb, lsb }
    }
}

impl PartialOrd for Uuid {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Uuid {
    fn cmp(&self, other: &Self) -> Ordering {
        // signed msb then signed lsb
        self.msb
            .cmp(&other.msb)
            .then_with(|| self.lsb.cmp(&other.lsb))
    }
}

/// 2-byte BE signed short. Natural (signed) order.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShortSer;
/// 2-byte BE unsigned char. Natural (unsigned) order == wire order.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharSer;
/// 4-byte BE int.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntSer;
/// 8-byte BE long.
#[derive(Debug, Clone, Copy, Default)]
pub struct LongSer;
/// 16-byte UUID.
#[derive(Debug, Clone, Copy, Default)]
pub struct UuidSer;
/// `packInt(utf8len)` + UTF-8 bytes; UTF-16 code-unit order (`String.compareTo`).
#[derive(Debug, Clone, Copy, Default)]
pub struct StringSer;
/// `packInt(len)` + bytes; **signed** lexicographic order (`Arrays.compare`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteArraySer;
/// Same wire/equality as [`ByteArraySer`] but **unsigned** (`memcmp`) order.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteArrayUnsignedSer;

/// Static singleton instances (so `GroupFormat::element` can return a
/// `&'static dyn Serializer<_>`).
pub static SHORT: ShortSer = ShortSer;
pub static CHAR: CharSer = CharSer;
pub static INT: IntSer = IntSer;
pub static LONG: LongSer = LongSer;
pub static UUID: UuidSer = UuidSer;
pub static STRING: StringSer = StringSer;
pub static BYTE_ARRAY: ByteArraySer = ByteArraySer;
pub static BYTE_ARRAY_UNSIGNED: ByteArrayUnsignedSer = ByteArrayUnsignedSer;

impl Serializer<i16> for ShortSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i16) {
        out.write_i16(*value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i16> {
        input.read_i16()
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(2)
    }
    fn compare(&self, a: &i16, b: &i16) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &i16, b: &i16) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<u16> for CharSer {
    fn serialize(&self, out: &mut DataOutput2, value: &u16) {
        out.write_u16(*value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<u16> {
        input.read_u16()
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(2)
    }
    fn compare(&self, a: &u16, b: &u16) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &u16, b: &u16) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<i32> for IntSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i32) {
        out.write_i32(*value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i32> {
        input.read_i32()
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(4)
    }
    fn compare(&self, a: &i32, b: &i32) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &i32, b: &i32) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<i64> for LongSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i64) {
        out.write_i64(*value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i64> {
        input.read_i64()
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(8)
    }
    fn compare(&self, a: &i64, b: &i64) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &i64, b: &i64) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Uuid> for UuidSer {
    fn serialize(&self, out: &mut DataOutput2, value: &Uuid) {
        out.write_i64(value.msb);
        out.write_i64(value.lsb);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Uuid> {
        let msb = input.read_i64()?;
        let lsb = input.read_i64()?;
        Ok(Uuid { msb, lsb })
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(16)
    }
    fn compare(&self, a: &Uuid, b: &Uuid) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Uuid, b: &Uuid) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<String> for StringSer {
    fn serialize(&self, out: &mut DataOutput2, value: &String) {
        let b = value.as_bytes();
        out.pack_int(b.len() as i32);
        out.write_all(b);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<String> {
        let len = read_framed_len(input)?;
        let mut b = Vec::new();
        b.try_reserve(len)?;
        b.resize(len, 0);
        input.read_fully(&mut b)?;
        // Materialization path is LOSSY (Java `new String(bytes, UTF_8)`),
        // matching Java's replacement behavior (spec 01 §3). Rust `String`
        // cannot hold ill-formed data so the lossy encode path is unreachable
        // (D9.1); decode uses `from_utf8_lossy`.
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn compare(&self, a: &String, b: &String) -> Ordering {
        // Java `String.compareTo` is UTF-16 code-unit order. For well-formed
        // Rust strings this differs from `str` (code-point) order only for
        // supplementary characters; handled by `util::compare_utf16`.
        super::util::compare_utf16(a, b)
    }
    fn equals(&self, a: &String, b: &String) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<u8>> for ByteArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<u8>) {
        out.pack_int(value.len() as i32);
        out.write_all(value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<u8>> {
        let len = read_framed_len(input)?;
        let mut b = Vec::new();
        b.try_reserve(len)?;
        b.resize(len, 0);
        input.read_fully(&mut b)?;
        Ok(b)
    }
    fn compare(&self, a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
        // Java `Arrays.compare` — signed byte lexicographic.
        super::util::compare_signed_bytes(a, b)
    }
    fn equals(&self, a: &Vec<u8>, b: &Vec<u8>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<u8>> for ByteArrayUnsignedSer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<u8>) {
        BYTE_ARRAY.serialize(out, value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<Vec<u8>> {
        BYTE_ARRAY.deserialize(input, size)
    }
    fn compare(&self, a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
        // Java `Arrays.compareUnsigned` — memcmp.
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<u8>, b: &Vec<u8>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}
