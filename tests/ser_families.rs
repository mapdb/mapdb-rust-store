//! Port of `SerializerParityTest.java` — round-trip coverage for the serializer
//! families (scalars, packed ints, primitive/object arrays,
//! BigInteger/BigDecimal/Date, ArraySerializer, CompressionSerializer) plus the
//! hostile-length allocation guards. Catalog-reopen and JVM-only serializers
//! (CLASS/JAVA/STRING_INTERN) are out of scope (see PORTING-GAPS.md).

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::families::{
    BIG_DECIMAL, BIG_INTEGER, BOOLEAN, BOOLEAN_ARRAY, BYTE, BYTE_ARRAY_NOSIZE, CHAR_ARRAY, DATE,
    DOUBLE, DOUBLE_ARRAY, FLOAT, FLOAT_ARRAY, INTEGER_PACKED, INT_ARRAY, LONG_ARRAY, LONG_PACKED,
    RECID, RECID_ARRAY, SHORT_ARRAY, STRING_ASCII, STRING_NOSIZE,
};
use mapdb_rust_store::ser::{
    ArraySerializer, BigDecimal, BigInt, CompressionSerializer, Date, Serializer, StringSer,
};

fn round_trip<A, S: Serializer<A>>(ser: &S, value: &A) -> A {
    let mut out = DataOutput2::new();
    ser.serialize(&mut out, value);
    let bytes = out.copy_bytes();
    let mut input = SliceInput::new(&bytes);
    ser.deserialize(&mut input, Some(bytes.len()))
        .expect("deserialize")
}

#[test]
fn scalar_and_packed_round_trips() {
    assert!(round_trip(&BOOLEAN, &true));
    assert_eq!(round_trip(&BYTE, &-128i8), -128i8);
    // -0.0 must survive bit-exactly (not collapse to +0.0).
    assert_eq!(round_trip(&FLOAT, &-0.0f32).to_bits(), (-0.0f32).to_bits());
    assert!(round_trip(&DOUBLE, &f64::NAN).is_nan());
    for v in [i32::MIN, -1, 0, 1, 127, 128, i32::MAX] {
        assert_eq!(round_trip(&INTEGER_PACKED, &v), v);
    }
    for v in [i64::MIN, -1, 0, 1, 127, 128, i64::MAX] {
        assert_eq!(round_trip(&LONG_PACKED, &v), v);
    }
}

#[test]
fn primitive_array_round_trips() {
    let bools = vec![true, false, true, true, false, false, false, true, true];
    assert_eq!(round_trip(&BOOLEAN_ARRAY, &bools), bools);
    let chars = vec![0u16, b'x' as u16, u16::MAX];
    assert_eq!(round_trip(&CHAR_ARRAY, &chars), chars);
    let shorts = vec![i16::MIN, 0, i16::MAX];
    assert_eq!(round_trip(&SHORT_ARRAY, &shorts), shorts);
    let ints = vec![i32::MIN, 0, i32::MAX];
    assert_eq!(round_trip(&INT_ARRAY, &ints), ints);
    let longs = vec![i64::MIN, 0, i64::MAX];
    assert_eq!(round_trip(&LONG_ARRAY, &longs), longs);
    let floats = vec![-0.0f32, 1.5, f32::NAN];
    let got = round_trip(&FLOAT_ARRAY, &floats);
    assert_eq!(got[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(got[1], 1.5);
    assert!(got[2].is_nan());
    let doubles = vec![-0.0f64, 1.5, f64::NAN];
    let got = round_trip(&DOUBLE_ARRAY, &doubles);
    assert_eq!(got[0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(got[1], 1.5);
    assert!(got[2].is_nan());
}

#[test]
fn object_like_and_no_size_round_trips() {
    assert_eq!(
        round_trip(&BYTE_ARRAY_NOSIZE, &vec![1u8, 2, 3]),
        vec![1u8, 2, 3]
    );
    assert_eq!(
        round_trip(&STRING_NOSIZE, &"žluťoučký".to_string()),
        "žluťoučký"
    );
    assert_eq!(
        round_trip(&STRING_ASCII, &"plain ASCII".to_string()),
        "plain ASCII"
    );
    assert_eq!(round_trip(&RECID, &123456789i64), 123456789i64);
    let recids = vec![1i64, 2, i64::MAX];
    assert_eq!(round_trip(&RECID_ARRAY, &recids), recids);

    let big: BigInt = "-123456789012345678901234567890".parse().unwrap();
    assert_eq!(round_trip(&BIG_INTEGER, &big), big);

    // -1234567890.0012300 == unscaled -12345678900012300 at scale 7.
    let dec = BigDecimal::new("-12345678900012300".parse().unwrap(), 7);
    assert_eq!(round_trip(&BIG_DECIMAL, &dec), dec);

    assert_eq!(round_trip(&DATE, &Date(123456789)), Date(123456789));

    let array = ArraySerializer::new(StringSer);
    let strs = vec!["a".to_string(), "b".to_string()];
    assert_eq!(round_trip(&array, &strs), strs);
}

#[test]
fn compression_wrapper_round_trips_large_and_empty_values() {
    let strings = CompressionSerializer::new(StringSer);
    let value = "abcdefghij".repeat(10_000);
    assert_eq!(round_trip(&strings, &value), value);

    let raw = CompressionSerializer::new(BYTE_ARRAY_NOSIZE);
    assert_eq!(round_trip(&raw, &Vec::<u8>::new()), Vec::<u8>::new());
}

#[test]
fn compression_body_is_standard_zlib_framed() {
    // The compressed body carries the standard zlib CMF/FLG header (0x78 for the
    // default 32K window), so a Java `DeflaterOutputStream` record and this
    // record share the same wire format and decompress cross-language, even
    // though the exact compressor output is not byte-identical (PORTING-GAPS.md).
    let ser = CompressionSerializer::new(StringSer);
    let value = "the quick brown fox".to_string();
    let mut out = DataOutput2::new();
    ser.serialize(&mut out, &value);
    let bytes = out.copy_bytes();
    let mut input = SliceInput::new(&bytes);
    let _plain_len = input.unpack_int().unwrap();
    let clen = input.unpack_int().unwrap() as usize;
    let start = input.pos();
    let body = &bytes[start..start + clen];
    assert_eq!(body[0], 0x78, "zlib CMF header byte");
    // And it still round-trips through our own deserialize.
    assert_eq!(round_trip(&ser, &value), value);
}

#[test]
fn hostile_lengths_rejected_before_allocation() {
    // Array length = i32::MAX with no elements present.
    let mut array_frame = DataOutput2::new();
    array_frame.pack_int(i32::MAX);
    let array = ArraySerializer::new(StringSer);
    let bytes = array_frame.copy_bytes();
    let mut input = SliceInput::new(&bytes);
    assert!(array.deserialize(&mut input, Some(bytes.len())).is_err());

    // Compressed plain length = i32::MAX, compressed length = 0.
    let mut comp_frame = DataOutput2::new();
    comp_frame.pack_int(i32::MAX);
    comp_frame.pack_int(0);
    let comp = CompressionSerializer::new(StringSer);
    let bytes = comp_frame.copy_bytes();
    let mut input = SliceInput::new(&bytes);
    assert!(comp.deserialize(&mut input, Some(bytes.len())).is_err());
}

#[test]
fn equals_by_serialized_bytes_flags_match_java() {
    // 620fd6b: char[] canonical (true), compression non-canonical (false).
    assert!(CHAR_ARRAY.equals_by_serialized_bytes());
    assert!(!CompressionSerializer::new(StringSer).equals_by_serialized_bytes());
}

#[test]
fn big_decimal_compare_is_value_based_but_equals_is_scale_sensitive() {
    // 1.0 vs 1.00 : compareTo == 0 (value), equals == false (scale).
    let one_0 = BigDecimal::new("10".parse().unwrap(), 1); // 1.0
    let one_00 = BigDecimal::new("100".parse().unwrap(), 2); // 1.00
    assert_eq!(
        BIG_DECIMAL.compare(&one_0, &one_00),
        std::cmp::Ordering::Equal
    );
    assert!(!BIG_DECIMAL.equals(&one_0, &one_00));
    // 1.0 < 2.0
    let two_0 = BigDecimal::new("20".parse().unwrap(), 1);
    assert_eq!(
        BIG_DECIMAL.compare(&one_0, &two_0),
        std::cmp::Ordering::Less
    );
}
