//! Bit-parity encodings for on-volume pointers/counters (Java `Parity`, spec 02
//! §5). The low N bits of the stored long carry a checksum of the payload bits:
//! `p1` for 2-aligned payloads, `p4` for 16-aligned, `p16` for 1 MiB-aligned.
//!
//! A raw stored value of 0 always FAILS its parity check, so "never written /
//! lost update" is distinguishable from every legitimately-stored value
//! (including the encoded 0 used for empty links — `p1set(0) == 1`).

use crate::error::{DbError, Result};

/// `v` must have bit 0 clear; result has an odd total bit count.
#[inline]
pub fn p1set(v: u64) -> u64 {
    debug_assert!(v & 1 == 0, "parity1 payload uses bit 0");
    v | (((v.count_ones() + 1) & 1) as u64)
}

/// Validate and strip parity1.
#[inline]
pub fn p1get(v: u64) -> Result<u64> {
    if v.count_ones() & 1 != 1 {
        return Err(DbError::corrupt("parity1 broken"));
    }
    Ok(v & !1)
}

/// `v` must have the low 4 bits clear.
#[inline]
pub fn p4set(v: u64) -> u64 {
    debug_assert!(v & 0xF == 0, "parity4 payload uses low 4 bits");
    v | (((v.count_ones() + 1) & 0xF) as u64)
}

#[inline]
pub fn p4get(v: u64) -> Result<u64> {
    let x = v & !0xF;
    if (v & 0xF) != (((x.count_ones() + 1) & 0xF) as u64) {
        return Err(DbError::corrupt("parity4 broken"));
    }
    Ok(x)
}

/// `v` must have the low 16 bits clear.
#[inline]
pub fn p16set(v: u64) -> u64 {
    debug_assert!(v & 0xFFFF == 0, "parity16 payload uses low 16 bits");
    v | (((v.count_ones() + 1) & 0xFFFF) as u64)
}

#[inline]
pub fn p16get(v: u64) -> Result<u64> {
    let x = v & !0xFFFF;
    if (v & 0xFFFF) != (((x.count_ones() + 1) & 0xFFFF) as u64) {
        return Err(DbError::corrupt("parity16 broken"));
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_zero_fails() {
        for &v in &[0u64, 16, 0x10, 0xFF0, 1u64 << 20, 0x0000_FFFF_FFFF_FFF0] {
            assert_eq!(p1get(p1set(v)).unwrap(), v);
        }
        for &v in &[0u64, 16, 0x10, 0xFF0, 1u64 << 20, 0x0000_FFFF_FFFF_FFF0] {
            assert_eq!(p4get(p4set(v)).unwrap(), v);
        }
        for &v in &[0u64, 1u64 << 16, 1u64 << 20, 1u64 << 40] {
            assert_eq!(p16get(p16set(v)).unwrap(), v);
        }
        // raw 0 fails every parity check (the "never written" guarantee)
        assert!(p1get(0).is_err());
        assert!(p4get(0).is_err());
        assert!(p16get(0).is_err());
        // p*set(0) is a valid non-zero encoding of 0
        assert_eq!(p1set(0), 1);
        assert_eq!(p1get(p1set(0)).unwrap(), 0);
    }

    #[test]
    fn corrupt_bit_flip_detected() {
        let good = p4set(0x1230);
        assert!(p4get(good ^ 0x100).is_err()); // flip a payload bit
    }
}
