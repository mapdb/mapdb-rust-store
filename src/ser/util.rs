//! Shared ser helpers: zigzag, Java-exact comparisons, in-place UTF-8 compare.
//!
//! Order coherence (top-risk #3): each comparison here mirrors an exact Java
//! semantic. `compare_utf16` = `String.compareTo` (UTF-16 code-unit order);
//! `compare_signed_bytes` = `Arrays.compare` (signed); `compare_utf8` matches
//! `String.compareTo` against stored UTF-8 in place (spec 01 §3, `Utf8`).

use crate::error::{DbError, Result};
use crate::io::DataInput2;
use std::cmp::Ordering;

/// `(v<<1) ^ (v>>63)` — signed → unsigned zigzag.
#[inline]
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// `(v>>>1) ^ -(v&1)` — inverse of [`zigzag`].
#[inline]
pub fn unzigzag(v: u64) -> i64 {
    ((v >> 1) as i64) ^ (-((v & 1) as i64))
}

/// `String.compareTo`: UTF-16 code-unit lexicographic order. Differs from Rust
/// `str` (code-point) order only for supplementary characters.
#[inline]
pub fn compare_utf16(a: &str, b: &str) -> Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// `Arrays.compare(byte[], byte[])`: signed-byte lexicographic, shorter-is-less
/// on a shared prefix.
#[inline]
pub fn compare_signed_bytes(a: &[u8], b: &[u8]) -> Ordering {
    let n = a.len().min(b.len());
    for i in 0..n {
        let c = (a[i] as i8).cmp(&(b[i] as i8));
        if c != Ordering::Equal {
            return c;
        }
    }
    a.len().cmp(&b.len())
}

/// Sign of `stored.compareTo(key)` where `stored` is the UTF-8 string spanning
/// exactly `byte_len` bytes at `input`'s position and `key` is given as its
/// UTF-16 code units. Consumes at most `byte_len` bytes (fewer on early
/// difference; caller re-seeks). Zero allocation.
///
/// STRICT: malformed/torn UTF-8 → `Err(DataCorruption)` (RFC 3629
/// well-formedness), never a plausible-looking key (spec 01 §3, `Utf8`).
pub fn compare_utf8(
    input: &mut dyn DataInput2,
    byte_len: usize,
    key_utf16: &[u16],
) -> Result<Ordering> {
    let mut rem = byte_len;
    let mut ci = 0usize;
    let key_len = key_utf16.len();

    // Compare one produced UTF-16 unit against the key; returns Some(ord) to
    // stop, None to continue.
    macro_rules! cmp_unit {
        ($unit:expr) => {{
            if ci == key_len {
                return Ok(Ordering::Greater); // key is a strict prefix of stored
            }
            let c = ($unit as i32) - (key_utf16[ci] as i32);
            ci += 1;
            if c != 0 {
                return Ok(if c < 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                });
            }
        }};
    }

    while rem > 0 {
        let b0 = input.read_u8()? as u32;
        rem -= 1;
        let (mut cp, need): (u32, usize) = if b0 < 0x80 {
            (b0, 0)
        } else if b0 & 0xE0 == 0xC0 {
            (b0 & 0x1F, 1)
        } else if b0 & 0xF0 == 0xE0 {
            (b0 & 0x0F, 2)
        } else if b0 & 0xF8 == 0xF0 {
            (b0 & 0x07, 3)
        } else {
            return Err(DbError::corrupt("corrupt utf8 lead byte"));
        };
        if need > rem {
            return Err(DbError::corrupt("corrupt utf8 truncated sequence"));
        }
        for _ in 0..need {
            let b = input.read_u8()? as u32;
            if b & 0xC0 != 0x80 {
                return Err(DbError::corrupt("corrupt utf8 continuation byte"));
            }
            cp = (cp << 6) | (b & 0x3F);
        }
        rem -= need;
        // RFC 3629 well-formedness.
        match need {
            1 => {
                if cp < 0x80 {
                    return Err(DbError::corrupt("corrupt utf8 overlong sequence"));
                }
            }
            2 => {
                if cp < 0x800 || (0xD800..=0xDFFF).contains(&cp) {
                    return Err(DbError::corrupt("corrupt utf8 overlong or surrogate"));
                }
            }
            3 => {
                if !(0x10000..=0x10FFFF).contains(&cp) {
                    return Err(DbError::corrupt("corrupt utf8 overlong or out-of-range"));
                }
            }
            _ => {}
        }
        if cp < 0x10000 {
            cmp_unit!(cp as u16);
        } else {
            let v = cp - 0x10000;
            let hi_sur = 0xD800u16 | (v >> 10) as u16;
            let lo_sur = 0xDC00u16 | (v & 0x3FF) as u16;
            cmp_unit!(hi_sur);
            cmp_unit!(lo_sur);
        }
    }
    // stored exhausted: equal, or stored is a strict prefix of key
    Ok(if ci == key_len {
        Ordering::Equal
    } else {
        Ordering::Less
    })
}

/// Length of the common **byte** prefix of two byte slices.
#[inline]
pub fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}
