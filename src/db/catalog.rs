//! The DB name catalog — a `Map<String,String>` stored at recid 1, byte-for-byte
//! compatible with Java `DB.CATALOG_SER`.
//!
//! ## Wire format ("MDBC" v1)
//!
//! ```text
//! magic:   u32 big-endian  0x4D444243 ("MDBC")
//! version: u32 big-endian  1
//! repr:    u8              0            (REPR_INLINE)
//! count:   packInt         number of entries
//! entries: count × (key, value)         sorted ascending by key
//!   each string = packInt(utf8-byte-len) ++ utf8 bytes  (Serializers.STRING)
//! ```
//!
//! The `packInt`/`packLong` varint puts 7 data bits per byte, most-significant
//! group first, and sets the high bit `0x80` on the **terminal** byte (NOT
//! LEB128). `packInt(0)` is the single byte `0x80`. An empty catalog is exactly
//! the 10 bytes `4D 44 42 43 00 00 00 01 00 80`.
//!
//! Entries are kept in a [`BTreeMap`], whose ascending key order matches Java's
//! `TreeMap` natural (`String.compareTo`, UTF-16 code-unit) order because all
//! catalog keys are restricted to ASCII (`[A-Za-z0-9._-]` name + `#` + ASCII
//! parameter suffix), for which byte order equals UTF-16 order.
//!
//! The decoder is defensive: it rejects records shorter than
//! 10 bytes, a bad magic/version/representation, a count over `MAX_CATALOG_ENTRIES`,
//! a string length crossing the record end, invalid UTF-8 in a string, duplicate
//! keys, and any trailing or short bytes.

use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use crate::ser::Serializer;
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Reserved recid holding the name catalog (Java `DB.RECID_CATALOG`).
pub const RECID_CATALOG: u64 = 1;

const CATALOG_MAGIC: u32 = 0x4D44_4243; // "MDBC"
const CATALOG_VERSION: i32 = 1;
const REPR_INLINE: u8 = 0;
const MAX_CATALOG_ENTRIES: u64 = 10_000_000;
/// magic(4) + version(4) + repr(1) + at least one count byte.
const MIN_CATALOG_LEN: usize = 10;

/// The name catalog: sorted `name#param -> value` string pairs. A `BTreeMap`
/// so serialization is deterministic and byte-compatible with Java's `TreeMap`.
pub type NameCatalog = BTreeMap<String, String>;

/// The catalog codec (Java `DB.CATALOG_SER`). Not a registered serializer; only
/// the DB facade uses it, at recid 1.
#[derive(Debug, Clone, Copy, Default)]
pub struct CatalogSer;

/// The single shared instance.
pub static CATALOG_SER: CatalogSer = CatalogSer;

#[inline]
fn write_string(out: &mut DataOutput2, s: &str) {
    let b = s.as_bytes();
    out.pack_int(b.len() as i32);
    out.write_all(b);
}

/// Read a `packInt`-framed UTF-8 string, bounded by `end` (a record offset).
fn read_bounded_string(input: &mut dyn DataInput2, end: usize) -> Result<String> {
    let len = read_bounded_packed(input, end)?;
    let len: usize = len
        .try_into()
        .map_err(|_| DbError::corrupt("catalog string length overflow"))?;
    // The bytes must not cross the record end.
    if input.pos().checked_add(len).map_or(true, |p| p > end) {
        return Err(DbError::corrupt("catalog string crosses record end"));
    }
    let mut b = Vec::new();
    b.try_reserve(len)?;
    b.resize(len, 0);
    input.read_fully(&mut b)?;
    // Strict UTF-8 decode: malformed bytes are catalog corruption, not something
    // to silently repair. Java's `Serializers.STRING` encoder
    // only ever emits valid UTF-8, so a legitimately-written catalog always
    // decodes cleanly; rejecting invalid UTF-8 loses no Java wire compatibility.
    // (A lossy decode would additionally diverge from Java's read behavior — e.g.
    // an overlong surrogate `ED A0 80` yields one U+FFFD on the JVM but three via
    // `from_utf8_lossy` — so it is not even faithful corrupt-input parity.)
    String::from_utf8(b).map_err(|_| DbError::corrupt("catalog string is not valid UTF-8"))
}

/// A valid packed `u64` occupies at most `ceil(64/7) == 10` bytes. A longer run
/// is corruption, not a value whose high bits we may silently discard.
const MAX_PACKED_BYTES: usize = 10;

/// Decode a packed varint, bounded so it cannot read past `end` (Java
/// `DB.readBoundedPackedLong`). Same bit layout as `unpackLong`, but rejects a
/// non-canonical / overlong run instead of shifting indefinitely (a release
/// build would otherwise discard the overflowed high bits).
fn read_bounded_packed(input: &mut dyn DataInput2, end: usize) -> Result<u64> {
    let mut ret: u64 = 0;
    for i in 0..MAX_PACKED_BYTES {
        if input.pos() >= end {
            return Err(DbError::corrupt(
                "catalog packed value runs past record end",
            ));
        }
        let v = input.read_u8()?;
        let group = (v & 0x7F) as u64;
        let terminal = (v & 0x80) != 0;
        // Canonicality: an MSB-first packed value never begins with an all-zero
        // group unless the value IS zero (a single terminal byte). A leading zero
        // group followed by more bytes is a non-canonical encoding, not a value.
        if i == 0 && group == 0 && !terminal {
            return Err(DbError::corrupt(
                "catalog packed value has a non-canonical leading zero",
            ));
        }
        // Checked accumulation on EVERY byte (this encoding is MSB-first, so the
        // overflow can land on any byte, not just the 10th): a run whose value
        // exceeds 64 bits is corruption, never a value whose high bits we discard.
        ret = ret
            .checked_mul(128)
            .and_then(|x| x.checked_add(group))
            .ok_or_else(|| DbError::corrupt("catalog packed value overflows 64 bits"))?;
        if terminal {
            return Ok(ret);
        }
    }
    Err(DbError::corrupt(
        "catalog packed value is overlong (unterminated)",
    ))
}

impl Serializer<NameCatalog> for CatalogSer {
    fn serialize(&self, out: &mut DataOutput2, cat: &NameCatalog) {
        out.write_i32(CATALOG_MAGIC as i32);
        out.write_i32(CATALOG_VERSION);
        out.write_u8(REPR_INLINE);
        out.pack_int(cat.len() as i32);
        // BTreeMap iterates ascending by key == Java TreeMap natural order.
        for (k, v) in cat.iter() {
            write_string(out, k);
            write_string(out, v);
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<NameCatalog> {
        let size = size.ok_or_else(|| DbError::corrupt("catalog record needs a known size"))?;
        if size < MIN_CATALOG_LEN {
            return Err(DbError::corrupt("catalog record shorter than 10 bytes"));
        }
        let start = input.pos();
        let end = start
            .checked_add(size)
            .ok_or_else(|| DbError::corrupt("catalog record end overflow"))?;

        let magic = input.read_i32()? as u32;
        if magic != CATALOG_MAGIC {
            return Err(DbError::corrupt(
                "catalog magic mismatch (not a MapDB catalog)",
            ));
        }
        let version = input.read_i32()?;
        if version != CATALOG_VERSION {
            return Err(DbError::corrupt("unsupported catalog version"));
        }
        let repr = input.read_u8()?;
        if repr != REPR_INLINE {
            return Err(DbError::corrupt("unsupported catalog representation"));
        }
        let count = read_bounded_packed(input, end)?;
        if count > MAX_CATALOG_ENTRIES {
            return Err(DbError::corrupt("catalog entry count exceeds maximum"));
        }
        let mut cat = NameCatalog::new();
        for _ in 0..count {
            let key = read_bounded_string(input, end)?;
            let value = read_bounded_string(input, end)?;
            if cat.insert(key, value).is_some() {
                return Err(DbError::corrupt("duplicate key in catalog"));
            }
        }
        // No trailing or short bytes (Java asserts pos == start+size).
        if input.pos() != end {
            return Err(DbError::corrupt("catalog has trailing bytes"));
        }
        Ok(cat)
    }

    fn compare(&self, _a: &NameCatalog, _b: &NameCatalog) -> Ordering {
        // The catalog is never used as a key; ordering is irrelevant.
        Ordering::Equal
    }

    fn equals(&self, a: &NameCatalog, b: &NameCatalog) -> bool {
        a == b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::SliceInput;

    fn encode(cat: &NameCatalog) -> Vec<u8> {
        let mut out = DataOutput2::new();
        CATALOG_SER.serialize(&mut out, cat);
        out.into_vec()
    }

    fn decode(bytes: &[u8]) -> Result<NameCatalog> {
        let mut input = SliceInput::new(bytes);
        CATALOG_SER.deserialize(&mut input, Some(bytes.len()))
    }

    #[test]
    fn empty_catalog_golden_bytes() {
        // The exact 10 bytes Java emits for an empty catalog.
        let cat = NameCatalog::new();
        assert_eq!(
            encode(&cat),
            vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x80]
        );
    }

    #[test]
    fn single_entry_golden_bytes() {
        // One entry {"al#type" -> "AtomicLong"}. Hand-computed Java encoding:
        //   header:  4D 44 42 43 | 00 00 00 01 | 00
        //   count 1: 81 (packInt(1) = 0x80 | 1)
        //   key "al#type" (7 bytes):   87  61 6C 23 74 79 70 65   (packInt(7)=0x87)
        //   val "AtomicLong" (10):     8A  41 74 6F 6D 69 63 4C 6F 6E 67 (packInt(10)=0x8A)
        let mut cat = NameCatalog::new();
        cat.insert("al#type".to_string(), "AtomicLong".to_string());
        let expected: Vec<u8> = vec![
            0x4D, 0x44, 0x42, 0x43, // magic
            0x00, 0x00, 0x00, 0x01, // version
            0x00, // repr
            0x81, // count = 1
            0x87, b'a', b'l', b'#', b't', b'y', b'p', b'e', // key
            0x8A, b'A', b't', b'o', b'm', b'i', b'c', b'L', b'o', b'n', b'g', // value
        ];
        assert_eq!(encode(&cat), expected);
        // Round-trips.
        assert_eq!(decode(&expected).unwrap(), cat);
    }

    #[test]
    fn entries_serialize_in_sorted_order() {
        // Insertion order is irrelevant; bytes are ascending by key.
        let mut a = NameCatalog::new();
        a.insert("b".to_string(), "2".to_string());
        a.insert("a".to_string(), "1".to_string());
        let mut b = NameCatalog::new();
        b.insert("a".to_string(), "1".to_string());
        b.insert("b".to_string(), "2".to_string());
        assert_eq!(encode(&a), encode(&b));
        // "a" (0x61) sorts before "b" (0x62) — first key byte after count.
        let bytes = encode(&a);
        // header(9) + count(1) + packInt(1)=0x81 for "a" len 1 then 'a'
        assert_eq!(bytes[10], 0x81);
        assert_eq!(bytes[11], b'a');
    }

    #[test]
    fn round_trips_a_realistic_treemap_row() {
        let mut cat = NameCatalog::new();
        cat.insert("t#type".into(), "TreeMap".into());
        cat.insert("t#keySerializer".into(), "LONG".into());
        cat.insert("t#valueSerializer".into(), "STRING".into());
        cat.insert("t#rootRecidRecid".into(), "2".into());
        cat.insert("t#maxNodeSize".into(), "32".into());
        cat.insert("t#counterRecid".into(), "0".into());
        cat.insert("t#valueInline".into(), "true".into());
        let bytes = encode(&cat);
        assert_eq!(decode(&bytes).unwrap(), cat);
    }

    #[test]
    fn rejects_short_record() {
        assert!(matches!(
            decode(&[0x4D, 0x44, 0x42]),
            Err(DbError::DataCorruption(_))
        ));
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x80];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_bad_version() {
        let bytes = vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x09, 0x00, 0x80];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x80];
        bytes.push(0xFF); // extra trailing byte
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_string_crossing_record_end() {
        // count 1, key length claims 50 bytes but record ends immediately.
        let bytes = vec![
            0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x81, // count 1
            0xB2, // packInt(50)
        ];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_overlong_packed_count() {
        // header + 11 continuation bytes (high bit clear) with no terminator.
        let mut bytes = vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00];
        bytes.extend(std::iter::repeat(0x00).take(11));
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_overflowing_packed_count() {
        // 10 bytes where the leading byte forces >64 bits: first byte 0x02 (>0x01)
        // on the terminal 10th position path. Build 9 zero-continuation + terminal.
        let mut bytes = vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00];
        // 0x7F * 9 continuation then a terminal byte -> way over 64 bits.
        bytes.extend(std::iter::repeat(0x7F).take(9));
        bytes.push(0xFF); // terminal, high bit set, value bits set
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_unterminated_string_length() {
        // count 1, then an unterminated packed key length (high bit never set).
        let mut bytes = vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x81];
        bytes.extend(std::iter::repeat(0x00).take(11));
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_non_canonical_leading_zero_count() {
        // count encoded as [0x00, 0x81]: a leading zero group with a continuation
        // is non-canonical (the canonical zero is a single 0x80) — R2.
        let bytes = vec![
            0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x81,
        ];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_2_pow_64_with_zero_terminal_group() {
        // count = [0x02, 0x00×8, 0x80] = 2 * 128^9 = 2^64. The old terminal-byte
        // guard wrapped this to 0 and accepted it; checked accumulation rejects it
        // as overflow (R2).
        let mut bytes = vec![0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00];
        bytes.push(0x02);
        bytes.extend(std::iter::repeat(0x00).take(8));
        bytes.push(0x80);
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_duplicate_keys() {
        // Two entries with the same key "a".
        let bytes = vec![
            0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x82, // count 2
            0x81, b'a', 0x81, b'x', // ("a","x")
            0x81, b'a', 0x81, b'y', // ("a","y") duplicate key
        ];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_invalid_utf8_in_a_string() {
        // count 1, key length 1 with a lone continuation byte 0x80 (invalid UTF-8).
        // Java's encoder never emits this; a corrupt record decodes strictly to an
        // error rather than being silently repaired to U+FFFD.
        let bytes = vec![
            0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x81, // count 1
            0x81, 0x80, // key: len 1, byte 0x80 (invalid UTF-8 start)
            0x81, b'x', // value "x"
        ];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }

    #[test]
    fn rejects_overlong_surrogate_that_a_lossy_decode_would_accept() {
        // `ED A0 80` is the (invalid) UTF-8 of surrogate U+D800. `from_utf8_lossy`
        // would turn it into replacement chars and accept the record; strict decode
        // rejects it as corruption.
        let bytes = vec![
            0x4D, 0x44, 0x42, 0x43, 0x00, 0x00, 0x00, 0x01, 0x00, 0x81, // count 1
            0x83, 0xED, 0xA0, 0x80, // key: len 3, surrogate bytes
            0x81, b'x', // value "x"
        ];
        assert!(matches!(decode(&bytes), Err(DbError::DataCorruption(_))));
    }
}
