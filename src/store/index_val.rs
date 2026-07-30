//! Capacity-based index-value encoding (Java `IndexVal`, spec 02 §5):
//!
//! ```text
//! bit 63..48  capacityUnits (16 bits) — capacity = capacityUnits * 16 bytes
//! bit 47..4   offset (44 bits, 16-aligned)
//! bit 3       linked   (oversize records as chunk chains)
//! bit 2       prealloc (P state)
//! bit 1       archive  (reserved)
//! bit 0       parity   (parity1 over the whole slot once stored)
//! ```
//! Record data layout at offset: 4-byte used-length header, then content.

/// Mask of the 44-bit, 16-aligned offset field.
pub const MOFFSET: u64 = 0x0000_FFFF_FFFF_FFF0;

pub const FLAG_LINKED: u64 = 8;
pub const FLAG_PREALLOC: u64 = 4;
pub const FLAG_ARCHIVE: u64 = 2;

/// capacityUnits sentinel: record content is null (P state iff `FLAG_PREALLOC`).
pub const CAP_NULL: u32 = 0xFFFF;
/// capacityUnits sentinel: recid deleted (tombstone).
pub const CAP_DELETED: u32 = 0xFFFE;
pub const CAP_MAX_UNITS: u32 = 0xFFFD;
/// Max plain-record capacity incl. 4-byte header: ~1 MiB − 48.
pub const MAX_CAPACITY: usize = CAP_MAX_UNITS as usize * 16;

#[inline]
pub fn compose(cap_units: u32, offset: u64, flags: u64) -> u64 {
    debug_assert!(
        offset & !MOFFSET == 0,
        "offset not 16-aligned or out of range"
    );
    ((cap_units as u64) << 48) | offset | flags
}

#[inline]
pub fn cap_units(iv: u64) -> u32 {
    (iv >> 48) as u32
}

#[inline]
pub fn offset(iv: u64) -> u64 {
    iv & MOFFSET
}

#[inline]
pub fn is_prealloc(iv: u64) -> bool {
    iv & FLAG_PREALLOC != 0
}

#[inline]
pub fn is_linked(iv: u64) -> bool {
    iv & FLAG_LINKED != 0
}

#[inline]
pub fn round_up16(n: usize) -> usize {
    (n + 15) & !15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_decompose() {
        let iv = compose(3, 0x1_0000, FLAG_LINKED);
        assert_eq!(cap_units(iv), 3);
        assert_eq!(offset(iv), 0x1_0000);
        assert!(is_linked(iv));
        assert!(!is_prealloc(iv));

        let p = compose(CAP_NULL, 0, FLAG_PREALLOC);
        assert_eq!(cap_units(p), CAP_NULL);
        assert_eq!(offset(p), 0);
        assert!(is_prealloc(p));
    }

    #[test]
    fn round_up() {
        assert_eq!(round_up16(0), 0);
        assert_eq!(round_up16(1), 16);
        assert_eq!(round_up16(16), 16);
        assert_eq!(round_up16(17), 32);
    }
}
