//! Codec descriptor strings — the stable, Java-byte-compatible wire identifiers
//! persisted in the name catalog for every codec.
//!
//! Because the port monomorphizes every collection over its concrete
//! `GroupFormat` / `Serializer` (decisions D1/D2), there is no runtime class to
//! reflect on. Instead each built-in codec type implements [`GroupDescriptor`] /
//! [`SerDescriptor`], returning the exact Java-registered identifier (`LONG`,
//! `INT`, `STRING`, `OBJECT_ARRAY:<b64url>`, `DEFLATE:<level>:<b64url>`, …). A
//! codec whose wire identity the port cannot reproduce returns `None`, which the
//! catalog stores as the opaque marker [`CUSTOM`] and which reopen can only match
//! against another custom codec (never against a known descriptor).
//!
//! Unlike Java, the port never *reconstructs* a codec from its descriptor string
//! (that would require the erased dispatch D1 forbids). Typed opens always supply
//! the concrete codec; verification is a pure string comparison of the supplied
//! codec's descriptor against the stored one (see [`verify_group`] /
//! [`verify_ser`]). The tooling-only `inspect_catalog` decodes metadata but
//! constructs nothing.

use crate::ser::bytearray::{ByteArrayFormat, ByteArrayPrefixFormat};
use crate::ser::columnar::{ColumnType, ColumnarValueFormat};
use crate::ser::families::{
    BigDecimalSer, BigIntegerSer, BooleanArraySer, BooleanSer, ByteArrayNoSizeSer, ByteSer,
    CharArraySer, CompressionSerializer, DateSer, DoubleArraySer, DoubleSer, FloatArraySer,
    FloatSer, IntArraySer, IntegerPackedSer, LongArraySer, LongPackedSer, RecidArraySer, RecidSer,
    ShortArraySer, StringAsciiSer, StringNoSizeSer,
};
use crate::ser::int::{IntDeltaFormat, IntFormat};
use crate::ser::long::{LongDeltaFormat, LongFormat};
use crate::ser::object_array::ObjectArrayFormat;
use crate::ser::scalar::{CharFormat, ShortFormat, UuidFormat};
use crate::ser::serializers::{
    ByteArraySer, ByteArrayUnsignedSer, CharSer, IntSer, LongSer, ShortSer, StringSer, UuidSer,
};
use crate::ser::string_group::StringGroupFormat;
use crate::ser::string_prefix::StringPrefixFormat;
use crate::ser::tuple::{TupleComponent, TupleFormat};
use crate::ser::Serializer;

use crate::error::{DbError, Result};

/// Opaque marker written for a codec whose exact wire identity the port cannot
/// reproduce (Java writes `CUSTOM:<fqcn>`; the port has no class name, so it
/// writes the bare marker and requires the codec to be re-supplied on reopen).
pub const CUSTOM: &str = "CUSTOM";

/// A group format that can name itself for the catalog.
pub trait GroupDescriptor {
    /// The stable Java-compatible descriptor, or `None` for a custom codec.
    fn group_descriptor(&self) -> Option<String>;
}

/// An element serializer that can name itself for the catalog.
pub trait SerDescriptor {
    /// The stable Java-compatible descriptor, or `None` for a custom codec.
    fn ser_descriptor(&self) -> Option<String>;
}

/// The descriptor to persist for a group format (custom → [`CUSTOM`]).
pub fn group_descriptor_or_custom<F: GroupDescriptor>(f: &F) -> String {
    f.group_descriptor().unwrap_or_else(|| CUSTOM.to_string())
}

/// The descriptor to persist for an element serializer (custom → [`CUSTOM`]).
pub fn ser_descriptor_or_custom<S: SerDescriptor>(s: &S) -> String {
    s.ser_descriptor().unwrap_or_else(|| CUSTOM.to_string())
}

#[inline]
fn is_custom_marker(stored: &str) -> bool {
    stored == CUSTOM || stored.starts_with("CUSTOM:")
}

/// Verify a supplied group format against the stored catalog descriptor.
pub fn verify_group<F: GroupDescriptor>(stored: &str, supplied: &F) -> Result<()> {
    verify(
        stored,
        supplied.group_descriptor(),
        is_valid_group_descriptor(stored),
    )
}

/// Verify a supplied element serializer against the stored catalog descriptor.
pub fn verify_ser<S: SerDescriptor>(stored: &str, supplied: &S) -> Result<()> {
    verify(
        stored,
        supplied.ser_descriptor(),
        is_valid_ser_descriptor(stored),
    )
}

/// `stored_valid` = the stored descriptor parses as a known built-in / valid
/// recursive-grammar / `CUSTOM` marker. A mismatch against a VALID stored
/// descriptor is wrong configuration; a mismatch against a MALFORMED stored
/// descriptor is catalog corruption.
fn verify(stored: &str, supplied: Option<String>, stored_valid: bool) -> Result<()> {
    match supplied {
        Some(d) if d == stored => Ok(()),
        Some(d) => {
            if stored_valid {
                Err(DbError::wrong_config(format!(
                    "codec descriptor mismatch: catalog has '{stored}' but the supplied codec is '{d}'"
                )))
            } else {
                Err(DbError::corrupt_msg(format!(
                    "unknown/malformed stored codec descriptor '{stored}'"
                )))
            }
        }
        // A custom (unreproducible) codec matches only a stored custom marker.
        None if is_custom_marker(stored) => Ok(()),
        None if stored_valid => Err(DbError::wrong_config(format!(
            "a custom codec was supplied but the catalog stored the known descriptor '{stored}'; \
             re-supply the exact codec used at create time"
        ))),
        None => Err(DbError::corrupt_msg(format!(
            "unknown/malformed stored codec descriptor '{stored}'"
        ))),
    }
}

// ---- descriptor grammar validation (defensive: a stored descriptor the port
// cannot classify is corruption, not a configuration mismatch) ----

const BUILTIN_GROUPS: &[&str] = &[
    "LONG",
    "INT",
    "SHORT",
    "CHAR",
    "UUID",
    "STRING",
    "STRING_PREFIX",
    "BYTE_ARRAY",
    "BYTE_ARRAY_PREFIX",
    "INT_DELTA",
    "LONG_DELTA",
];

const BUILTIN_SERS: &[&str] = &[
    "LONG",
    "INTEGER",
    "SHORT",
    "CHAR",
    "UUID",
    "STRING",
    "BYTE_ARRAY",
    "BYTE_ARRAY_UNSIGNED",
    "BOOLEAN",
    "BYTE",
    "FLOAT",
    "DOUBLE",
    "INTEGER_PACKED",
    "LONG_PACKED",
    "BYTE_ARRAY_NOSIZE",
    "STRING_NOSIZE",
    "STRING_ASCII",
    "STRING_INTERN",
    "RECID",
    "RECID_ARRAY",
    "BOOLEAN_ARRAY",
    "CHAR_ARRAY",
    "SHORT_ARRAY",
    "INT_ARRAY",
    "LONG_ARRAY",
    "FLOAT_ARRAY",
    "DOUBLE_ARRAY",
    "BIG_INTEGER",
    "BIG_DECIMAL",
    "DATE",
    "CLASS",
    "JAVA",
];

const TUPLE_COMPONENTS: &[&str] = &["INT", "LONG", "STRING", "BYTES"];
const COLUMN_TYPES: &[&str] = &["LONG", "INT", "SHORT", "BYTE"];

/// True if `s` is a valid group-format descriptor (built-in, recursive grammar,
/// or the `CUSTOM` marker).
pub fn is_valid_group_descriptor(s: &str) -> bool {
    if BUILTIN_GROUPS.contains(&s) || is_custom_marker(s) {
        return true;
    }
    if let Some(b64) = s.strip_prefix("OBJECT_ARRAY:") {
        return match b64url_decode(b64) {
            Some(nested) => is_valid_ser_descriptor(&nested),
            None => false,
        };
    }
    if let Some(list) = s.strip_prefix("TUPLE:") {
        return !list.is_empty() && list.split(',').all(|c| TUPLE_COMPONENTS.contains(&c));
    }
    if let Some(list) = s.strip_prefix("COLUMNAR:") {
        return !list.is_empty() && list.split(',').all(|c| COLUMN_TYPES.contains(&c));
    }
    false
}

/// True if `s` is a valid element-serializer descriptor.
pub fn is_valid_ser_descriptor(s: &str) -> bool {
    if BUILTIN_SERS.contains(&s) || is_custom_marker(s) {
        return true;
    }
    if let Some(rest) = s.strip_prefix("DEFLATE:") {
        let mut it = rest.splitn(2, ':');
        let level = it.next();
        let nested_b64 = it.next();
        return match (level, nested_b64) {
            (Some(l), Some(b64)) => {
                // Java `CompressionSerializer` (Deflater) only accepts -1..=9;
                // any other level makes the registry return null → catalog
                // corruption, so it is NOT a valid descriptor (R8).
                l.parse::<i32>()
                    .map_or(false, |lvl| (-1..=9).contains(&lvl))
                    && b64url_decode(b64)
                        .map(|n| is_valid_ser_descriptor(&n))
                        .unwrap_or(false)
            }
            _ => false,
        };
    }
    if let Some(rest) = s.strip_prefix("ARRAY:") {
        // ARRAY:<b64(component-class)>:<b64(nested descriptor)>
        let mut it = rest.splitn(2, ':');
        let comp = it.next();
        let nested_b64 = it.next();
        return match (comp, nested_b64) {
            (Some(c), Some(b64)) => {
                b64url_decode(c).is_some()
                    && b64url_decode(b64)
                        .map(|n| is_valid_ser_descriptor(&n))
                        .unwrap_or(false)
            }
            _ => false,
        };
    }
    false
}

// =========================== base64url (RFC 4648, no padding) ===========================

const B64URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// RFC 4648 URL-safe base64 without padding, over UTF-8 (Java
/// `Base64.getUrlEncoder().withoutPadding()`).
pub fn b64url_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64URL[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64URL[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64URL[((n >> 6) & 0x3F) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(B64URL[(n & 0x3F) as usize] as char);
        }
    }
    out
}

/// Decode RFC 4648 URL-safe base64 without padding back to a UTF-8 string.
/// Returns `None` on an illegal alphabet, illegal length group, or non-UTF-8.
pub fn b64url_decode(s: &str) -> Option<String> {
    #[inline]
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'-' => Some(62),
            b'_' => Some(63),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() / 4 * 3 + 2);
    for chunk in bytes.chunks(4) {
        if chunk.len() == 1 {
            return None; // an illegal trailing group of 1
        }
        let mut n: u32 = 0;
        for &c in chunk {
            n = (n << 6) | val(c)?;
        }
        // left-align to a multiple of 24 bits
        n <<= 6 * (4 - chunk.len());
        out.push((n >> 16) as u8);
        if chunk.len() >= 3 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() >= 4 {
            out.push(n as u8);
        }
    }
    String::from_utf8(out).ok()
}

/// `OBJECT_ARRAY:<b64url(nested-serializer-descriptor)>`.
fn object_array_descriptor(nested: Option<String>) -> Option<String> {
    let nested = nested?;
    Some(format!("OBJECT_ARRAY:{}", b64url_encode(nested.as_bytes())))
}

// =========================== element serializer descriptors ===========================

macro_rules! ser_desc {
    ($ty:ty, $id:literal) => {
        impl SerDescriptor for $ty {
            fn ser_descriptor(&self) -> Option<String> {
                Some($id.to_string())
            }
        }
    };
}

ser_desc!(LongSer, "LONG");
ser_desc!(IntSer, "INTEGER");
ser_desc!(ShortSer, "SHORT");
ser_desc!(CharSer, "CHAR");
ser_desc!(UuidSer, "UUID");
ser_desc!(StringSer, "STRING");
ser_desc!(ByteArraySer, "BYTE_ARRAY");
ser_desc!(ByteArrayUnsignedSer, "BYTE_ARRAY_UNSIGNED");
ser_desc!(BooleanSer, "BOOLEAN");
ser_desc!(ByteSer, "BYTE");
ser_desc!(FloatSer, "FLOAT");
ser_desc!(DoubleSer, "DOUBLE");
ser_desc!(IntegerPackedSer, "INTEGER_PACKED");
ser_desc!(LongPackedSer, "LONG_PACKED");
ser_desc!(ByteArrayNoSizeSer, "BYTE_ARRAY_NOSIZE");
ser_desc!(StringNoSizeSer, "STRING_NOSIZE");
ser_desc!(StringAsciiSer, "STRING_ASCII");
ser_desc!(RecidSer, "RECID");
ser_desc!(RecidArraySer, "RECID_ARRAY");
ser_desc!(BooleanArraySer, "BOOLEAN_ARRAY");
ser_desc!(CharArraySer, "CHAR_ARRAY");
ser_desc!(ShortArraySer, "SHORT_ARRAY");
ser_desc!(IntArraySer, "INT_ARRAY");
ser_desc!(LongArraySer, "LONG_ARRAY");
ser_desc!(FloatArraySer, "FLOAT_ARRAY");
ser_desc!(DoubleArraySer, "DOUBLE_ARRAY");
ser_desc!(BigIntegerSer, "BIG_INTEGER");
ser_desc!(BigDecimalSer, "BIG_DECIMAL");
ser_desc!(DateSer, "DATE");

/// `DEFLATE:<level>:<b64url(nested-descriptor)>` (Java `CompressionSerializer`).
impl<A, S: Serializer<A> + SerDescriptor> SerDescriptor for CompressionSerializer<A, S> {
    fn ser_descriptor(&self) -> Option<String> {
        let nested = self.delegate().ser_descriptor()?;
        Some(format!(
            "DEFLATE:{}:{}",
            self.level(),
            b64url_encode(nested.as_bytes())
        ))
    }
}

// NOTE: `ArraySerializer` intentionally has NO descriptor impl. Java's `ARRAY`
// descriptor embeds the Java component class name (`ARRAY:<b64(class)>:<b64(nested)>`),
// which the Rust port cannot reproduce (there is no Java class). It is therefore
// treated as a custom codec (requires re-supply on reopen). See PORTING-GAPS.

// =========================== group format descriptors ===========================

macro_rules! group_desc {
    ($ty:ty, $id:literal) => {
        impl GroupDescriptor for $ty {
            fn group_descriptor(&self) -> Option<String> {
                Some($id.to_string())
            }
        }
    };
}

group_desc!(LongFormat, "LONG");
group_desc!(IntFormat, "INT");
group_desc!(ShortFormat, "SHORT");
group_desc!(CharFormat, "CHAR");
group_desc!(UuidFormat, "UUID");
group_desc!(StringGroupFormat, "STRING");
group_desc!(StringPrefixFormat, "STRING_PREFIX");
group_desc!(ByteArrayFormat, "BYTE_ARRAY");
group_desc!(ByteArrayPrefixFormat, "BYTE_ARRAY_PREFIX");
group_desc!(IntDeltaFormat, "INT_DELTA");
group_desc!(LongDeltaFormat, "LONG_DELTA");

/// `OBJECT_ARRAY:<b64url(element-serializer-descriptor)>`.
impl<A, S> GroupDescriptor for ObjectArrayFormat<A, S>
where
    A: Clone + Send + Sync + 'static,
    S: Serializer<A> + SerDescriptor + Send + Sync + 'static,
{
    fn group_descriptor(&self) -> Option<String> {
        object_array_descriptor(self.element_serializer().ser_descriptor())
    }
}

fn tuple_component_name(c: TupleComponent) -> &'static str {
    match c {
        TupleComponent::Int => "INT",
        TupleComponent::Long => "LONG",
        TupleComponent::Str => "STRING",
        TupleComponent::Bytes => "BYTES",
    }
}

/// `TUPLE:<comp>[,<comp>...]` (Java `TupleFormat`; component = `TupleComponent.name()`).
impl GroupDescriptor for TupleFormat {
    fn group_descriptor(&self) -> Option<String> {
        let names: Vec<&str> = self
            .schema()
            .into_iter()
            .map(tuple_component_name)
            .collect();
        Some(format!("TUPLE:{}", names.join(",")))
    }
}

fn column_type_name(c: ColumnType) -> &'static str {
    match c {
        ColumnType::Long => "LONG",
        ColumnType::Int => "INT",
        ColumnType::Short => "SHORT",
        ColumnType::Byte => "BYTE",
    }
}

/// `COLUMNAR:<coltype>[,<coltype>...]` (Java `ColumnarValueFormat`).
impl GroupDescriptor for ColumnarValueFormat {
    fn group_descriptor(&self) -> Option<String> {
        let names: Vec<&str> = (0..self.column_count())
            .map(|i| column_type_name(self.column_type(i)))
            .collect();
        Some(format!("COLUMNAR:{}", names.join(",")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_group_descriptors() {
        assert_eq!(LongFormat.group_descriptor().unwrap(), "LONG");
        assert_eq!(IntFormat.group_descriptor().unwrap(), "INT");
        assert_eq!(StringGroupFormat.group_descriptor().unwrap(), "STRING");
        assert_eq!(ByteArrayFormat.group_descriptor().unwrap(), "BYTE_ARRAY");
    }

    #[test]
    fn builtin_ser_descriptors() {
        assert_eq!(LongSer.ser_descriptor().unwrap(), "LONG");
        assert_eq!(IntSer.ser_descriptor().unwrap(), "INTEGER");
        assert_eq!(BooleanSer.ser_descriptor().unwrap(), "BOOLEAN");
        assert_eq!(StringSer.ser_descriptor().unwrap(), "STRING");
    }

    #[test]
    fn tuple_descriptor_uses_java_component_names() {
        let f = TupleFormat::of(&[
            TupleComponent::Str,
            TupleComponent::Long,
            TupleComponent::Int,
        ]);
        assert_eq!(f.group_descriptor().unwrap(), "TUPLE:STRING,LONG,INT");
    }

    #[test]
    fn columnar_descriptor_uses_java_column_names() {
        let f = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Int]);
        assert_eq!(f.group_descriptor().unwrap(), "COLUMNAR:LONG,INT");
    }

    #[test]
    fn object_array_wraps_nested_serializer() {
        let f = ObjectArrayFormat::new(StringSer);
        // b64url("STRING") with no padding.
        let expected = format!("OBJECT_ARRAY:{}", b64url_encode(b"STRING"));
        assert_eq!(f.group_descriptor().unwrap(), expected);
    }

    #[test]
    fn deflate_descriptor_embeds_level_and_nested() {
        let s = CompressionSerializer::with_level(StringSer, 6);
        let expected = format!("DEFLATE:6:{}", b64url_encode(b"STRING"));
        assert_eq!(s.ser_descriptor().unwrap(), expected);
    }

    #[test]
    fn deflate_descriptor_rejects_out_of_range_level() {
        let nested = b64url_encode(b"STRING");
        // Java `CompressionSerializer` (Deflater) accepts only -1..=9.
        assert!(is_valid_ser_descriptor(&format!("DEFLATE:-1:{nested}")));
        assert!(is_valid_ser_descriptor(&format!("DEFLATE:0:{nested}")));
        assert!(is_valid_ser_descriptor(&format!("DEFLATE:9:{nested}")));
        // Out of range → registry returns null → catalog corruption (R8).
        assert!(!is_valid_ser_descriptor(&format!("DEFLATE:-2:{nested}")));
        assert!(!is_valid_ser_descriptor(&format!("DEFLATE:10:{nested}")));
        // i32 overflow in the level field is likewise invalid.
        assert!(!is_valid_ser_descriptor(&format!(
            "DEFLATE:99999999999:{nested}"
        )));
    }

    #[test]
    fn b64url_matches_known_vectors() {
        // Java Base64.getUrlEncoder().withoutPadding()
        assert_eq!(b64url_encode(b"STRING"), "U1RSSU5H");
        assert_eq!(b64url_encode(b""), "");
        assert_eq!(b64url_encode(b"f"), "Zg");
        assert_eq!(b64url_encode(b"fo"), "Zm8");
        assert_eq!(b64url_encode(b"foo"), "Zm9v");
    }

    #[test]
    fn verify_accepts_match_rejects_mismatch() {
        assert!(verify_group("LONG", &LongFormat).is_ok());
        assert!(verify_group("STRING", &LongFormat).is_err());
    }

    #[test]
    fn verify_custom_only_matches_custom_marker() {
        struct CustomSer;
        impl SerDescriptor for CustomSer {
            fn ser_descriptor(&self) -> Option<String> {
                None
            }
        }
        assert!(verify_ser(CUSTOM, &CustomSer).is_ok());
        assert!(verify_ser("STRING", &CustomSer).is_err());
    }
}
