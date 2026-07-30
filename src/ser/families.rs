//! Additional built-in serializer families ported from MapDB 3
//! (`org.mapdb.ser.Serializers` plus `ArraySerializer`/`CompressionSerializer`).
//!
//! Java → Rust type map for the new families:
//! `Boolean`→`bool`, `Byte`→`i8`, `Float`→`f32`, `Double`→`f64`,
//! `Integer`→`i32`, `Long`→`i64`, `char[]`→`Vec<u16>`, `short[]`→`Vec<i16>`,
//! `int[]`→`Vec<i32>`, `long[]`→`Vec<i64>`, `float[]`→`Vec<f32>`,
//! `double[]`→`Vec<f64>`, `boolean[]`→`Vec<bool>`, `BigInteger`→[`num_bigint::BigInt`],
//! `BigDecimal`→[`BigDecimal`], `Date`→[`Date`] (epoch millis).
//!
//! All encodings are byte-for-byte identical to Java (validated by
//! `SerializerParityTest`). Serialize-side validation that Java performs by
//! throwing (`RECID` > 0, `STRING_ASCII` in-range) is relaxed to a
//! `debug_assert!` + deserialize-side check, because the port's
//! [`Serializer::serialize`](super::Serializer) is infallible (deviation, see
//! `PORTING-GAPS.md`).

use super::Serializer;
use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
pub use num_bigint::BigInt;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Guard against a hostile length prefix sizing an allocation before the record
/// backs it (D4). Mirrors `serializers::read_framed_len` but for element counts:
/// each element needs at least one byte, so `count > remaining` is impossible in
/// a valid record.
fn checked_count(input: &mut dyn DataInput2) -> Result<usize> {
    let raw = input.unpack_int()?;
    if raw < 0 {
        return Err(DbError::corrupt("negative array length"));
    }
    let count = raw as usize;
    if count > input.remaining() {
        return Err(DbError::corrupt("array length exceeds record"));
    }
    Ok(count)
}

// ---- scalar families --------------------------------------------------------

/// 1-byte boolean (`0`/`1`); rejects other bytes on decode.
#[derive(Debug, Clone, Copy, Default)]
pub struct BooleanSer;
/// 1-byte signed byte.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteSer;
/// 4-byte IEEE-754 float (`floatToIntBits` big-endian).
#[derive(Debug, Clone, Copy, Default)]
pub struct FloatSer;
/// 8-byte IEEE-754 double (`doubleToLongBits` big-endian).
#[derive(Debug, Clone, Copy, Default)]
pub struct DoubleSer;
/// Packed two's-complement `i32` (`packInt`); negatives use five bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntegerPackedSer;
/// Packed two's-complement `i64` (`packLong`); negatives use ten bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct LongPackedSer;
/// Positive record id encoded as a packed long.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecidSer;

pub static BOOLEAN: BooleanSer = BooleanSer;
pub static BYTE: ByteSer = ByteSer;
pub static FLOAT: FloatSer = FloatSer;
pub static DOUBLE: DoubleSer = DoubleSer;
pub static INTEGER_PACKED: IntegerPackedSer = IntegerPackedSer;
pub static LONG_PACKED: LongPackedSer = LongPackedSer;
pub static RECID: RecidSer = RecidSer;

impl Serializer<bool> for BooleanSer {
    fn serialize(&self, out: &mut DataOutput2, value: &bool) {
        out.write_byte(if *value { 1 } else { 0 });
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<bool> {
        let v = input.read_unsigned_byte()?;
        if v > 1 {
            return Err(DbError::corrupt("invalid boolean byte"));
        }
        Ok(v != 0)
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(1)
    }
    fn compare(&self, a: &bool, b: &bool) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &bool, b: &bool) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<i8> for ByteSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i8) {
        out.write_byte(*value as i32);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i8> {
        input.read_i8()
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(1)
    }
    fn compare(&self, a: &i8, b: &i8) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &i8, b: &i8) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<f32> for FloatSer {
    fn serialize(&self, out: &mut DataOutput2, value: &f32) {
        out.write_i32(value.to_bits() as i32);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<f32> {
        Ok(f32::from_bits(input.read_i32()? as u32))
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(4)
    }
    fn compare(&self, a: &f32, b: &f32) -> Ordering {
        // Java `Float.compare`: total order (-0.0 < 0.0, NaN greatest).
        a.total_cmp(b)
    }
    fn equals(&self, a: &f32, b: &f32) -> bool {
        // Java `Float.equals` compares bits (NaN==NaN, -0.0 != 0.0).
        a.to_bits() == b.to_bits()
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<f64> for DoubleSer {
    fn serialize(&self, out: &mut DataOutput2, value: &f64) {
        out.write_i64(value.to_bits() as i64);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<f64> {
        Ok(f64::from_bits(input.read_i64()? as u64))
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(8)
    }
    fn compare(&self, a: &f64, b: &f64) -> Ordering {
        a.total_cmp(b)
    }
    fn equals(&self, a: &f64, b: &f64) -> bool {
        a.to_bits() == b.to_bits()
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<i32> for IntegerPackedSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i32) {
        out.pack_int(*value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i32> {
        input.unpack_int()
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

impl Serializer<i64> for LongPackedSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i64) {
        out.pack_long(*value as u64);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i64> {
        Ok(input.unpack_long()? as i64)
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

impl Serializer<i64> for RecidSer {
    fn serialize(&self, out: &mut DataOutput2, value: &i64) {
        debug_assert!(*value > 0, "recid must be positive");
        out.pack_long(*value as u64);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<i64> {
        let value = input.unpack_long()? as i64;
        if value <= 0 {
            return Err(DbError::corrupt("invalid recid"));
        }
        Ok(value)
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

// ---- no-size / ascii string families ---------------------------------------

/// Raw record bytes with no inner length prefix; requires a known record size.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteArrayNoSizeSer;
/// UTF-8 occupying the whole record, no inner length prefix.
#[derive(Debug, Clone, Copy, Default)]
pub struct StringNoSizeSer;
/// Seven-bit ASCII; rejects code points above `0x7F`.
#[derive(Debug, Clone, Copy, Default)]
pub struct StringAsciiSer;

pub static BYTE_ARRAY_NOSIZE: ByteArrayNoSizeSer = ByteArrayNoSizeSer;
pub static STRING_NOSIZE: StringNoSizeSer = StringNoSizeSer;
pub static STRING_ASCII: StringAsciiSer = StringAsciiSer;

impl Serializer<Vec<u8>> for ByteArrayNoSizeSer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<u8>) {
        out.write_all(value);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<Vec<u8>> {
        let size =
            size.ok_or_else(|| DbError::corrupt("BYTE_ARRAY_NOSIZE requires record size"))?;
        let mut b = Vec::new();
        b.try_reserve(size)?;
        b.resize(size, 0);
        input.read_fully(&mut b)?;
        Ok(b)
    }
    fn compare(&self, a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
        super::util::compare_signed_bytes(a, b)
    }
    fn equals(&self, a: &Vec<u8>, b: &Vec<u8>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<String> for StringNoSizeSer {
    fn serialize(&self, out: &mut DataOutput2, value: &String) {
        out.write_all(value.as_bytes());
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<String> {
        let size = size.ok_or_else(|| DbError::corrupt("STRING_NOSIZE requires record size"))?;
        let mut b = Vec::new();
        b.try_reserve(size)?;
        b.resize(size, 0);
        input.read_fully(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn compare(&self, a: &String, b: &String) -> Ordering {
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

impl Serializer<String> for StringAsciiSer {
    fn serialize(&self, out: &mut DataOutput2, value: &String) {
        // Java frames with the UTF-16 code-unit count; for an ASCII string this
        // equals the byte count. Non-ASCII is a precondition violation (Java
        // throws; the infallible Rust serialize cannot — see module doc).
        debug_assert!(value.is_ascii(), "STRING_ASCII requires 7-bit ASCII");
        let bytes = value.as_bytes();
        out.pack_int(bytes.len() as i32);
        for &c in bytes {
            out.write_byte((c & 0x7F) as i32);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<String> {
        let len = checked_count(input)?;
        let mut b = Vec::new();
        b.try_reserve(len)?;
        for _ in 0..len {
            let c = input.read_unsigned_byte()?;
            if c > 0x7F {
                return Err(DbError::corrupt("non-ASCII byte in STRING_ASCII"));
            }
            b.push(c as u8);
        }
        // All bytes < 0x80 → valid UTF-8/ASCII.
        String::from_utf8(b).map_err(|_| DbError::corrupt("invalid ASCII"))
    }
    fn compare(&self, a: &String, b: &String) -> Ordering {
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

// ---- primitive array families ----------------------------------------------

/// `packInt(len)` + `RECID`-encoded elements.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecidArraySer;
/// `packInt(len)` + big-endian 2-byte chars.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharArraySer;
/// `packInt(len)` + big-endian 2-byte shorts.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShortArraySer;
/// `packInt(len)` + 4-byte ints.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntArraySer;
/// `packInt(len)` + 8-byte longs.
#[derive(Debug, Clone, Copy, Default)]
pub struct LongArraySer;
/// `packInt(len)` + 4-byte float bits.
#[derive(Debug, Clone, Copy, Default)]
pub struct FloatArraySer;
/// `packInt(len)` + 8-byte double bits.
#[derive(Debug, Clone, Copy, Default)]
pub struct DoubleArraySer;
/// `packInt(len)` + LSB-first bit-packed booleans.
#[derive(Debug, Clone, Copy, Default)]
pub struct BooleanArraySer;

pub static RECID_ARRAY: RecidArraySer = RecidArraySer;
pub static CHAR_ARRAY: CharArraySer = CharArraySer;
pub static SHORT_ARRAY: ShortArraySer = ShortArraySer;
pub static INT_ARRAY: IntArraySer = IntArraySer;
pub static LONG_ARRAY: LongArraySer = LongArraySer;
pub static FLOAT_ARRAY: FloatArraySer = FloatArraySer;
pub static DOUBLE_ARRAY: DoubleArraySer = DoubleArraySer;
pub static BOOLEAN_ARRAY: BooleanArraySer = BooleanArraySer;

impl Serializer<Vec<i64>> for RecidArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<i64>) {
        out.pack_int(value.len() as i32);
        for &recid in value {
            RECID.serialize(out, &recid);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<i64>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(RECID.deserialize(input, None)?);
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<i64>, b: &Vec<i64>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<i64>, b: &Vec<i64>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<u16>> for CharArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<u16>) {
        out.pack_int(value.len() as i32);
        for &v in value {
            out.write_u16(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<u16>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(input.read_u16()?);
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<u16>, b: &Vec<u16>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<u16>, b: &Vec<u16>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<i16>> for ShortArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<i16>) {
        out.pack_int(value.len() as i32);
        for &v in value {
            out.write_i16(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<i16>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(input.read_i16()?);
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<i16>, b: &Vec<i16>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<i16>, b: &Vec<i16>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<i32>> for IntArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<i32>) {
        out.pack_int(value.len() as i32);
        for &v in value {
            out.write_i32(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<i32>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(input.read_i32()?);
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<i32>, b: &Vec<i32>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<i32>, b: &Vec<i32>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<i64>> for LongArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<i64>) {
        out.pack_int(value.len() as i32);
        for &v in value {
            out.write_i64(v);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<i64>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(input.read_i64()?);
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<i64>, b: &Vec<i64>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<i64>, b: &Vec<i64>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<f32>> for FloatArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<f32>) {
        out.pack_int(value.len() as i32);
        for &v in value {
            out.write_i32(v.to_bits() as i32);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<f32>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(f32::from_bits(input.read_i32()? as u32));
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<f32>, b: &Vec<f32>) -> Ordering {
        // Element-wise total order, then length (Arrays.compare over Float).
        for (x, y) in a.iter().zip(b.iter()) {
            let o = x.total_cmp(y);
            if o != Ordering::Equal {
                return o;
            }
        }
        a.len().cmp(&b.len())
    }
    fn equals(&self, a: &Vec<f32>, b: &Vec<f32>) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<f64>> for DoubleArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<f64>) {
        out.pack_int(value.len() as i32);
        for &v in value {
            out.write_i64(v.to_bits() as i64);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<f64>> {
        let len = checked_count(input)?;
        let mut v = Vec::new();
        v.try_reserve(len)?;
        for _ in 0..len {
            v.push(f64::from_bits(input.read_i64()? as u64));
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<f64>, b: &Vec<f64>) -> Ordering {
        for (x, y) in a.iter().zip(b.iter()) {
            let o = x.total_cmp(y);
            if o != Ordering::Equal {
                return o;
            }
        }
        a.len().cmp(&b.len())
    }
    fn equals(&self, a: &Vec<f64>, b: &Vec<f64>) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

impl Serializer<Vec<bool>> for BooleanArraySer {
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<bool>) {
        out.pack_int(value.len() as i32);
        let mut offset = 0;
        while offset < value.len() {
            let mut bits = 0i32;
            let mut bit = 0;
            while bit < 8 && offset + bit < value.len() {
                if value[offset + bit] {
                    bits |= 1 << bit;
                }
                bit += 1;
            }
            out.write_byte(bits);
            offset += 8;
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Vec<bool>> {
        // Element count is a BIT count; the packed byte run is ceil(len/8), so
        // the guard bounds the byte run (not the bit count) against the record.
        let raw = input.unpack_int()?;
        if raw < 0 {
            return Err(DbError::corrupt("negative array length"));
        }
        let len = raw as usize;
        let byte_run = len.div_ceil(8);
        if byte_run > input.remaining() {
            return Err(DbError::corrupt("array length exceeds record"));
        }
        let mut v = Vec::new();
        v.try_reserve(len)?;
        v.resize(len, false);
        let mut offset = 0;
        while offset < len {
            let bits = input.read_unsigned_byte()?;
            let mut bit = 0;
            while bit < 8 && offset + bit < len {
                v[offset + bit] = (bits & (1 << bit)) != 0;
                bit += 1;
            }
            offset += 8;
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<bool>, b: &Vec<bool>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<bool>, b: &Vec<bool>) -> bool {
        a == b
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

// ---- BigInteger / BigDecimal / Date ----------------------------------------

/// Big integer, wire-compatible with Java's
/// `BigInteger.toByteArray()` (minimal two's-complement, big-endian), framed by
/// `BYTE_ARRAY` (`packInt(len)` + bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct BigIntegerSer;
pub static BIG_INTEGER: BigIntegerSer = BigIntegerSer;

impl Serializer<BigInt> for BigIntegerSer {
    fn serialize(&self, out: &mut DataOutput2, value: &BigInt) {
        let bytes = value.to_signed_bytes_be();
        out.pack_int(bytes.len() as i32);
        out.write_all(&bytes);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<BigInt> {
        let raw = input.unpack_int()?;
        if raw < 0 {
            return Err(DbError::corrupt("negative BigInteger length"));
        }
        let len = raw as usize;
        if len > input.remaining() {
            return Err(DbError::corrupt("BigInteger length exceeds record"));
        }
        let mut b = Vec::new();
        b.try_reserve(len)?;
        b.resize(len, 0);
        input.read_fully(&mut b)?;
        Ok(BigInt::from_signed_bytes_be(&b))
    }
    fn compare(&self, a: &BigInt, b: &BigInt) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &BigInt, b: &BigInt) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

/// Arbitrary-precision decimal `unscaled * 10^-scale`, wire-compatible with
/// Java's `BigDecimal` codec: `BYTE_ARRAY(unscaledValue().toByteArray())` then
/// `packInt(scale())`.
///
/// [`PartialEq`]/[`Eq`] are **scale-sensitive** (Java `BigDecimal.equals`:
/// `1.0 != 1.00`), while [`BigDecimalSer::compare`] is **value-based**
/// (`BigDecimal.compareTo`: `1.0 == 1.00`), matching Java's deliberate
/// equals/compareTo inconsistency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigDecimal {
    pub unscaled: BigInt,
    pub scale: i32,
}

impl BigDecimal {
    pub fn new(unscaled: BigInt, scale: i32) -> Self {
        Self { unscaled, scale }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BigDecimalSer;
pub static BIG_DECIMAL: BigDecimalSer = BigDecimalSer;

impl Serializer<BigDecimal> for BigDecimalSer {
    fn serialize(&self, out: &mut DataOutput2, value: &BigDecimal) {
        let bytes = value.unscaled.to_signed_bytes_be();
        out.pack_int(bytes.len() as i32);
        out.write_all(&bytes);
        out.pack_int(value.scale);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<BigDecimal> {
        let raw = input.unpack_int()?;
        if raw < 0 {
            return Err(DbError::corrupt("negative BigDecimal unscaled length"));
        }
        let len = raw as usize;
        if len > input.remaining() {
            return Err(DbError::corrupt(
                "BigDecimal unscaled length exceeds record",
            ));
        }
        let mut b = Vec::new();
        b.try_reserve(len)?;
        b.resize(len, 0);
        input.read_fully(&mut b)?;
        let unscaled = BigInt::from_signed_bytes_be(&b);
        let scale = input.unpack_int()?;
        Ok(BigDecimal { unscaled, scale })
    }
    fn compare(&self, a: &BigDecimal, b: &BigDecimal) -> Ordering {
        // Value comparison, matching Java `BigDecimal.compareTo`: align to the
        // common (larger) scale by multiplying the smaller-scaled unscaled value
        // by 10^(scale difference), then compare the integers.
        if a.scale == b.scale {
            return a.unscaled.cmp(&b.unscaled);
        }
        let ten = BigInt::from(10);
        if a.scale < b.scale {
            let factor = num_traits::pow::pow(ten, (b.scale - a.scale) as usize);
            (&a.unscaled * factor).cmp(&b.unscaled)
        } else {
            let factor = num_traits::pow::pow(ten, (a.scale - b.scale) as usize);
            a.unscaled.cmp(&(&b.unscaled * factor))
        }
    }
    fn equals(&self, a: &BigDecimal, b: &BigDecimal) -> bool {
        // Java `BigDecimal.equals` is scale-sensitive.
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

/// A point in time as epoch milliseconds (Java `java.util.Date`, 8-byte BE long).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Date(pub i64);

#[derive(Debug, Clone, Copy, Default)]
pub struct DateSer;
pub static DATE: DateSer = DateSer;

impl Serializer<Date> for DateSer {
    fn serialize(&self, out: &mut DataOutput2, value: &Date) {
        out.write_i64(value.0);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, _size: Option<usize>) -> Result<Date> {
        Ok(Date(input.read_i64()?))
    }
    fn fixed_size(&self) -> Option<usize> {
        Some(8)
    }
    fn compare(&self, a: &Date, b: &Date) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Date, b: &Date) -> bool {
        a == b
    }
    fn natural_order(&self) -> bool {
        true
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        true
    }
}

// ---- generic wrapper serializers -------------------------------------------

/// Length-framed homogeneous array (Java `ArraySerializer<A>` for `A[]`);
/// Rust represents `A[]` as `Vec<A>`. Wire format: `packInt(len)` + each
/// element via the delegate. `equals_by_serialized_bytes` follows the element.
#[derive(Debug, Clone)]
pub struct ArraySerializer<A, S: Serializer<A>> {
    element: S,
    _marker: PhantomData<fn() -> A>,
}

/// Upper bound on array length (matches Java's `MAX_ARRAY_LENGTH`).
const MAX_ARRAY_LENGTH: usize = 16_000_000;

impl<A, S: Serializer<A>> ArraySerializer<A, S> {
    pub fn new(element: S) -> Self {
        Self {
            element,
            _marker: PhantomData,
        }
    }

    pub fn element_serializer(&self) -> &S {
        &self.element
    }
}

impl<A, S> Serializer<Vec<A>> for ArraySerializer<A, S>
where
    A: Clone,
    S: Serializer<A>,
{
    fn serialize(&self, out: &mut DataOutput2, value: &Vec<A>) {
        debug_assert!(
            value.len() <= MAX_ARRAY_LENGTH,
            "array length exceeds limit"
        );
        out.pack_int(value.len() as i32);
        for element in value {
            self.element.serialize(out, element);
        }
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<Vec<A>> {
        let start = input.pos();
        let raw = input.unpack_int()?;
        if raw < 0 || raw as usize > MAX_ARRAY_LENGTH {
            return Err(DbError::corrupt("invalid array length"));
        }
        let length = raw as usize;
        // Frame check for fixed-size elements, mirroring Java's guard.
        if let (Some(size), Some(fixed)) = (size, self.element.fixed_size()) {
            if fixed > 0 {
                let consumed = input.pos() - start;
                let remaining = (size as i64) - (consumed as i64);
                if remaining < 0 || (length as i64) * (fixed as i64) > remaining {
                    return Err(DbError::corrupt("array length exceeds record frame"));
                }
            }
        }
        // Each element consumes at least one byte, so a length exceeding the
        // bytes left in the record is corrupt — reject before allocating.
        if length > input.remaining() {
            return Err(DbError::corrupt("array length exceeds record"));
        }
        let mut v = Vec::new();
        v.try_reserve(length)?;
        for _ in 0..length {
            v.push(self.element.deserialize(input, None)?);
        }
        Ok(v)
    }
    fn compare(&self, a: &Vec<A>, b: &Vec<A>) -> Ordering {
        for (x, y) in a.iter().zip(b.iter()) {
            let o = self.element.compare(x, y);
            if o != Ordering::Equal {
                return o;
            }
        }
        a.len().cmp(&b.len())
    }
    fn equals(&self, a: &Vec<A>, b: &Vec<A>) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| self.element.equals(x, y))
    }
    fn equals_by_serialized_bytes(&self) -> bool {
        self.element.equals_by_serialized_bytes()
    }
}

/// Deterministic DEFLATE (zlib) wrapper over any element serializer
/// (Java `CompressionSerializer<A>`). Wire format:
/// `packInt(plainLen)` + `packInt(compressedLen)` + zlib-compressed bytes.
///
/// **Byte-compatibility note.** The stored bytes use the *standard zlib DEFLATE*
/// stream that Java's `DeflaterOutputStream` also writes, so a record written by
/// either language decompresses in the other. The *exact* compressor output is
/// NOT guaranteed identical (miniz_oxide vs. zlib), which is why
/// `equals_by_serialized_bytes` is `false` here — matching Java (DEFLATE output
/// is non-canonical across zlib versions/levels). See
/// `PORTING-GAPS.md`.
#[derive(Debug, Clone)]
pub struct CompressionSerializer<A, S: Serializer<A>> {
    delegate: S,
    level: i32,
    _marker: PhantomData<fn() -> A>,
}

const MAX_PLAIN_LENGTH: usize = 256 * 1024 * 1024;
const MAX_COMPRESSED_LENGTH: usize = MAX_PLAIN_LENGTH + 1024 * 1024;
/// Java `Deflater.DEFAULT_COMPRESSION`.
pub const DEFAULT_COMPRESSION: i32 = -1;
/// Java `Deflater.BEST_COMPRESSION`.
pub const BEST_COMPRESSION: i32 = 9;

impl<A, S: Serializer<A>> CompressionSerializer<A, S> {
    /// Wrap `delegate` at the default compression level.
    pub fn new(delegate: S) -> Self {
        Self::with_level(delegate, DEFAULT_COMPRESSION)
    }

    /// Wrap `delegate` at an explicit level (`-1`, or `0..=9`).
    pub fn with_level(delegate: S, level: i32) -> Self {
        debug_assert!(
            (DEFAULT_COMPRESSION..=BEST_COMPRESSION).contains(&level),
            "invalid compression level"
        );
        Self {
            delegate,
            level,
            _marker: PhantomData,
        }
    }

    pub fn delegate(&self) -> &S {
        &self.delegate
    }

    pub fn level(&self) -> i32 {
        self.level
    }

    /// miniz_oxide level (`0..=10`); Java's `-1` default maps to `6`.
    fn miniz_level(&self) -> u8 {
        if self.level < 0 {
            6
        } else {
            self.level as u8
        }
    }
}

impl<A, S> Serializer<A> for CompressionSerializer<A, S>
where
    A: Clone,
    S: Serializer<A>,
{
    fn serialize(&self, out: &mut DataOutput2, value: &A) {
        let mut plain_out = DataOutput2::with_capacity(self.delegate.size_hint().max(16));
        self.delegate.serialize(&mut plain_out, value);
        let plain = plain_out.into_vec();
        debug_assert!(
            plain.len() <= MAX_PLAIN_LENGTH,
            "uncompressed value too large"
        );
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, self.miniz_level());
        debug_assert!(
            compressed.len() <= MAX_COMPRESSED_LENGTH,
            "compressed value too large"
        );
        out.pack_int(plain.len() as i32);
        out.pack_int(compressed.len() as i32);
        out.write_all(&compressed);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<A> {
        let start = input.pos();
        let plain_len = input.unpack_int()?;
        let compressed_len = input.unpack_int()?;
        if plain_len < 0
            || plain_len as usize > MAX_PLAIN_LENGTH
            || compressed_len < 0
            || compressed_len as usize > MAX_COMPRESSED_LENGTH
        {
            return Err(DbError::corrupt("invalid compressed frame length"));
        }
        let plain_len = plain_len as usize;
        let compressed_len = compressed_len as usize;
        if let Some(size) = size {
            let consumed = input.pos() - start;
            let remaining = (size as i64) - (consumed as i64);
            if remaining < 0 || compressed_len as i64 > remaining {
                return Err(DbError::corrupt("compressed length exceeds record frame"));
            }
        }
        if compressed_len > input.remaining() {
            return Err(DbError::corrupt("compressed length exceeds record"));
        }
        let mut compressed = Vec::new();
        compressed.try_reserve(compressed_len)?;
        compressed.resize(compressed_len, 0);
        input.read_fully(&mut compressed)?;
        let plain = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&compressed, plain_len)
            .map_err(|_| DbError::corrupt("invalid compressed data"))?;
        if plain.len() != plain_len {
            return Err(DbError::corrupt("invalid compressed frame length"));
        }
        let mut plain_input = crate::io::SliceInput::new(&plain);
        self.delegate.deserialize(&mut plain_input, Some(plain_len))
    }
    fn compare(&self, a: &A, b: &A) -> Ordering {
        self.delegate.compare(a, b)
    }
    fn equals(&self, a: &A, b: &A) -> bool {
        self.delegate.equals(a, b)
    }
    fn natural_order(&self) -> bool {
        self.delegate.natural_order()
    }
    // DEFLATE output is not canonical, so byte comparison is
    // unsafe regardless of the delegate; do NOT delegate here.
    fn equals_by_serialized_bytes(&self) -> bool {
        false
    }
}
