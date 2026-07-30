//! `io` layer — `DataOutput2` / `DataInput2` and the packed-varint wire
//! format. Ported byte-for-byte from `org.mapdb.io` (spec 01 §1).
//!
//! Wire primitives:
//! - multi-byte integers are **big-endian**;
//! - **packed long** (mapdb lineage varint): 7 bits per byte, most-significant
//!   group first, the terminating byte has bit `0x80` set; non-negative only.
//!
//! Every read is bounds-checked and returns `Err(DbError::DataCorruption)` on
//! overrun — it never panics (spec 01 §1, decision D4).

use crate::error::{DbError, Result};

/// Growable serialization buffer. `pos == buf.len()` always (append-only),
/// so no separate position field is needed (spec 01 §1 Rust shape).
#[derive(Debug, Clone, Default)]
pub struct DataOutput2 {
    pub buf: Vec<u8>,
}

impl DataOutput2 {
    /// Default 128-byte hint, matching Java's `DataOutput2()`.
    pub fn new() -> Self {
        Self::with_capacity(128)
    }

    /// Initial-capacity hint; floored at 16 like Java's `Math.max(16, sizeHint)`.
    pub fn with_capacity(size_hint: usize) -> Self {
        Self {
            buf: Vec::with_capacity(size_hint.max(16)),
        }
    }

    /// Current write position (== number of bytes written).
    #[inline]
    pub fn pos(&self) -> usize {
        self.buf.len()
    }

    #[inline]
    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Java `writeByte(int)` — writes the low 8 bits.
    #[inline]
    pub fn write_byte(&mut self, v: i32) {
        self.buf.push(v as u8);
    }

    #[inline]
    pub fn write_all(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    #[inline]
    pub fn write_i16(&mut self, v: i16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    #[inline]
    pub fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    #[inline]
    pub fn write_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    #[inline]
    pub fn write_i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Packed long. Value must be non-negative (`u64`). Wire format per module
    /// docs. Faithful transcription of Java `packLong`.
    pub fn pack_long(&mut self, value: u64) {
        // shift = 63 - numberOfLeadingZeros(value); shift -= shift % 7
        // For value==0, leading_zeros==64 so shift starts at -1, then
        // -1 - (-1 % 7) == 0 (Rust `%` truncates toward zero like Java), so
        // the loop is skipped and a single terminator byte 0x80 is emitted.
        let mut shift: i32 = 63 - value.leading_zeros() as i32;
        shift -= shift % 7;
        while shift != 0 {
            self.buf.push(((value >> shift) & 0x7F) as u8);
            shift -= 7;
        }
        self.buf.push(((value & 0x7F) | 0x80) as u8);
    }

    /// `packInt(v) == packLong(v as u32 as u64)` — Java masks to 32 bits.
    #[inline]
    pub fn pack_int(&mut self, value: i32) {
        self.pack_long(value as u32 as u64);
    }

    /// Copy of the written bytes, exact length (Java `copyBytes`).
    pub fn copy_bytes(&self) -> Vec<u8> {
        self.buf.clone()
    }

    /// Consume and return the buffer.
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }
}

/// Maximum bytes a valid packed `u64` occupies (`ceil(64/7)`). Torn-safe
/// decoders reject longer runs so garbage terminates quickly (spec 01 §1, D4).
const MAX_PACKED_LONG_BYTES: usize = 10;
const MAX_PACKED_INT_BYTES: usize = 5;

/// Positioned, seekable read cursor over record bytes. The Java "valid only for
/// the duration of the call" contract becomes a lifetime on `SliceInput`.
///
/// Repositioning via [`set_pos`](DataInput2::set_pos) is unchecked (matching
/// Java's `pos(int)`); the following read is what bounds-checks. Torn-safe
/// decode paths use [`seek`](DataInput2::seek) instead, which validates
/// eagerly (decision D4).
pub trait DataInput2 {
    /// Total length of the underlying byte range.
    fn len(&self) -> usize;
    /// Current read position (may exceed `len` after a raw `set_pos`; the next
    /// read then errors).
    fn pos(&self) -> usize;
    /// Reposition without validation (Java `pos(int)`). Reads still bounds-check.
    fn set_pos(&mut self, pos: usize);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes remaining from the current position (saturating).
    fn remaining(&self) -> usize {
        self.len().saturating_sub(self.pos())
    }

    /// Read one byte, advancing. `Err` on overrun.
    fn read_u8(&mut self) -> Result<u8>;

    /// Fill `dst` fully, advancing by `dst.len()`. `Err` on overrun.
    fn read_fully(&mut self, dst: &mut [u8]) -> Result<()>;

    /// Java `readUnsignedByte()`.
    #[inline]
    fn read_unsigned_byte(&mut self) -> Result<i32> {
        Ok(self.read_u8()? as i32)
    }

    /// Java `readByte()` returning a signed byte.
    #[inline]
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    /// Skip `n` bytes, checked.
    fn skip_bytes(&mut self, n: usize) -> Result<()> {
        self.seek(self.pos().checked_add(n).ok_or_else(overflow)?)
    }

    /// Checked reposition (torn-safe subset). `Err` if `pos > len`.
    fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.len() {
            return Err(DbError::corrupt("seek out of range"));
        }
        self.set_pos(pos);
        Ok(())
    }

    /// 2-byte big-endian signed (Short).
    fn read_i16(&mut self) -> Result<i16> {
        let hi = self.read_u8()? as u16;
        let lo = self.read_u8()? as u16;
        Ok(((hi << 8) | lo) as i16)
    }

    /// 2-byte big-endian unsigned (Char).
    fn read_u16(&mut self) -> Result<u16> {
        let hi = self.read_u8()? as u16;
        let lo = self.read_u8()? as u16;
        Ok((hi << 8) | lo)
    }

    /// 4-byte big-endian (Java `readInt`).
    fn read_i32(&mut self) -> Result<i32> {
        let mut r: u32 = 0;
        for _ in 0..4 {
            r = (r << 8) | self.read_u8()? as u32;
        }
        Ok(r as i32)
    }

    /// 8-byte big-endian (Java `readLong`).
    fn read_i64(&mut self) -> Result<i64> {
        let mut r: u64 = 0;
        for _ in 0..8 {
            r = (r << 8) | self.read_u8()? as u64;
        }
        Ok(r as i64)
    }

    /// Raw big-endian `u64` (DirTree bitmap words, long-stack chunk headers).
    fn read_u64(&mut self) -> Result<u64> {
        Ok(self.read_i64()? as u64)
    }

    /// Decode a packed varint, rejecting a run longer than `max_bytes`
    /// (`10` for a `u64`, `5` for the 32-bit form). A valid value never needs
    /// more; an over-long run is corruption, not a hang.
    #[inline]
    fn unpack_capped(&mut self, max_bytes: usize) -> Result<u64> {
        let mut ret: u64 = 0;
        for _ in 0..max_bytes {
            let v = self.read_u8()?;
            ret = (ret << 7) | (v & 0x7F) as u64;
            if v & 0x80 != 0 {
                return Ok(ret);
            }
        }
        Err(DbError::corrupt("packed varint too long"))
    }

    /// Decode a packed long. Capped at 10 bytes (a valid `u64` never needs
    /// more); an over-long run is corruption.
    fn unpack_long(&mut self) -> Result<u64> {
        self.unpack_capped(MAX_PACKED_LONG_BYTES)
    }

    /// Java `unpackInt()` == `(int) unpackLong()`. Capped at 5 bytes (D4): a
    /// 32-bit value never needs more, so `unpack_int` and `unpack_long_skip`
    /// stay consistent on over-long runs instead of one accepting what the
    /// other rejects.
    #[inline]
    fn unpack_int(&mut self) -> Result<i32> {
        Ok(self.unpack_capped(MAX_PACKED_INT_BYTES)? as u32 as i32)
    }

    /// Skip `count` packed longs without decoding, scanning for terminator
    /// bytes (bit `0x80`). Load-bearing for DirTree slot skip and delta formats.
    /// Each skipped value is capped at 10 bytes, matching [`Self::unpack_long`],
    /// so a value the decoder would reject cannot be silently skipped instead.
    fn unpack_long_skip(&mut self, mut count: usize) -> Result<()> {
        while count > 0 {
            let mut run = 0usize;
            loop {
                let terminated = self.read_u8()? & 0x80 != 0;
                run += 1;
                if terminated {
                    break;
                }
                if run >= MAX_PACKED_LONG_BYTES {
                    return Err(DbError::corrupt("packed varint too long"));
                }
            }
            count -= 1;
        }
        Ok(())
    }

    /// Compare the next `expected.len()` bytes against `expected`. The position
    /// **always** advances by `expected.len()`, match or not. `Err` on overrun.
    fn match_bytes(&mut self, expected: &[u8]) -> Result<bool> {
        let p = self.pos();
        let end = p.checked_add(expected.len()).ok_or_else(overflow)?;
        if end > self.len() {
            return Err(DbError::corrupt("match_bytes out of range"));
        }
        let mut m = true;
        for &e in expected {
            if self.read_u8()? != e {
                m = false;
            }
        }
        self.set_pos(end);
        Ok(m)
    }
}

#[inline]
fn overflow() -> DbError {
    DbError::corrupt("offset overflow")
}

/// A view over a contiguous byte slice — covers Java's `ByteArray` and
/// `ByteBuf` (both are views over contiguous bytes). The mmap volume hands out
/// a `SliceInput` borrowed from a slice; the borrow cannot escape the read
/// action (spec 01 §1).
#[derive(Debug, Clone)]
pub struct SliceInput<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceInput<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    /// Borrow the underlying slice (for byte-side in-place compares).
    pub fn slice(&self) -> &'a [u8] {
        self.buf
    }

    /// A subslice `[start, start+len)` if fully in range; else corruption.
    /// Used by byte-side group formats reading blobs in place.
    pub fn subslice(&self, start: usize, len: usize) -> Result<&'a [u8]> {
        let end = start.checked_add(len).ok_or_else(overflow)?;
        self.buf
            .get(start..end)
            .ok_or_else(|| DbError::corrupt("subslice out of range"))
    }
}

impl<'a> DataInput2 for SliceInput<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.buf.len()
    }
    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }
    #[inline]
    fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| DbError::corrupt("read past end"))?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn read_fully(&mut self, dst: &mut [u8]) -> Result<()> {
        let end = self.pos.checked_add(dst.len()).ok_or_else(overflow)?;
        let src = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| DbError::corrupt("read_fully past end"))?;
        dst.copy_from_slice(src);
        self.pos = end;
        Ok(())
    }

    #[inline]
    fn read_i32(&mut self) -> Result<i32> {
        let end = self.pos.checked_add(4).ok_or_else(overflow)?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| DbError::corrupt("read_i32 past end"))?;
        let v = i32::from_be_bytes([s[0], s[1], s[2], s[3]]);
        self.pos = end;
        Ok(v)
    }

    #[inline]
    fn read_i64(&mut self) -> Result<i64> {
        let end = self.pos.checked_add(8).ok_or_else(overflow)?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| DbError::corrupt("read_i64 past end"))?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        self.pos = end;
        Ok(i64::from_be_bytes(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independently-computed packed-long vectors (spec 05 §1 boundaries).
    fn packed(v: u64) -> Vec<u8> {
        let mut o = DataOutput2::new();
        o.pack_long(v);
        o.into_vec()
    }

    #[test]
    fn pack_long_boundaries() {
        assert_eq!(packed(0), vec![0x80]);
        assert_eq!(packed(1), vec![0x81]);
        assert_eq!(packed(127), vec![0xFF]);
        // 128 = 0b1000_0000 -> groups: high 0x01, low 0x00|term
        assert_eq!(packed(128), vec![0x01, 0x80]);
        assert_eq!(packed(300), vec![0x02, 0xAC]); // 300 = 0b1_0010_1100
        assert_eq!(packed(16383), vec![0x7F, 0xFF]); // 2^14-1
        assert_eq!(packed(16384), vec![0x01, 0x00, 0x80]);
        // i64::MAX = 0x7FFF_FFFF_FFFF_FFFF -> 9 groups of 7 bits
        let m = packed(i64::MAX as u64);
        assert_eq!(m.len(), 9);
        assert_eq!(*m.last().unwrap(), 0xFF);
        // u64::MAX -> 10 bytes
        assert_eq!(packed(u64::MAX).len(), 10);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        for v in [
            0u64,
            1,
            63,
            64,
            127,
            128,
            129,
            255,
            256,
            16383,
            16384,
            1 << 20,
            1 << 35,
            i64::MAX as u64,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let bytes = packed(v);
            let mut inp = SliceInput::new(&bytes);
            assert_eq!(inp.unpack_long().unwrap(), v, "roundtrip {v}");
            assert_eq!(inp.pos(), bytes.len());
        }
    }

    #[test]
    fn pack_int_masks_32() {
        // packInt(-1) stores 0xFFFFFFFF (32-bit), unpack_int reinterprets to -1.
        let mut o = DataOutput2::new();
        o.pack_int(-1);
        let bytes = o.into_vec();
        let mut inp = SliceInput::new(&bytes);
        assert_eq!(inp.unpack_int().unwrap(), -1);

        let mut o2 = DataOutput2::new();
        o2.pack_int(5);
        assert_eq!(o2.into_vec(), packed(5));
    }

    #[test]
    fn big_endian_ints() {
        let mut o = DataOutput2::new();
        o.write_i32(0x01020304);
        o.write_i64(0x0102030405060708);
        o.write_i16(-2);
        let b = o.into_vec();
        assert_eq!(&b[0..4], &[1, 2, 3, 4]);
        assert_eq!(&b[4..12], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&b[12..14], &[0xFF, 0xFE]);
        let mut inp = SliceInput::new(&b);
        assert_eq!(inp.read_i32().unwrap(), 0x01020304);
        assert_eq!(inp.read_i64().unwrap(), 0x0102030405060708);
        assert_eq!(inp.read_i16().unwrap(), -2);
    }

    #[test]
    fn unpack_long_skip_matches_decode() {
        let mut o = DataOutput2::new();
        let vals = [0u64, 200, 128, 99999, 7];
        for &v in &vals {
            o.pack_long(v);
        }
        let b = o.into_vec();
        let mut inp = SliceInput::new(&b);
        inp.unpack_long_skip(3).unwrap();
        assert_eq!(inp.unpack_long().unwrap(), 99999);
        assert_eq!(inp.unpack_long().unwrap(), 7);
    }

    #[test]
    fn match_bytes_always_advances() {
        let b = b"MDB5.SD1extra";
        let mut inp = SliceInput::new(b);
        assert!(inp.match_bytes(b"MDB5.SD1").unwrap());
        assert_eq!(inp.pos(), 8);
        let mut inp2 = SliceInput::new(b);
        assert!(!inp2.match_bytes(b"XXXX.SD1").unwrap());
        assert_eq!(inp2.pos(), 8); // advanced despite mismatch
    }

    #[test]
    fn reads_error_never_panic() {
        let b = [0u8; 2];
        let mut inp = SliceInput::new(&b);
        assert!(inp.read_i32().is_err());
        let mut inp = SliceInput::new(&b);
        assert!(inp.read_i64().is_err());
        // over-long packed run
        let bad = [0u8; 12];
        let mut inp = SliceInput::new(&bad);
        assert!(inp.unpack_long().is_err());
        // seek past end
        let mut inp = SliceInput::new(&b);
        assert!(inp.seek(3).is_err());
        assert!(inp.seek(2).is_ok());
    }
}
