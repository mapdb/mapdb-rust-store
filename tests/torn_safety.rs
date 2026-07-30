//! Torn/corrupt-input hardening (decision D4). Decoding
//! untrusted bytes must return `Err(DataCorruption)` and never panic, wrap, or
//! attempt an unbounded allocation. These pin the corresponding fixes.

use mapdb_rust_store::error::DbError;
use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::serializers::{ByteArraySer, IntSer};
use mapdb_rust_store::ser::Serializer;

fn is_corrupt<T: std::fmt::Debug>(r: mapdb_rust_store::error::Result<T>) -> bool {
    matches!(r, Err(DbError::DataCorruption(_)))
}

#[test]
fn framed_length_beyond_record_is_rejected_not_allocated() {
    // A packed length prefix that far exceeds the bytes actually present must be
    // rejected before it is used to size a reservation — no
    // multi-gigabyte allocation from a tiny corrupt record.
    let mut out = DataOutput2::new();
    out.pack_int(1_000_000); // claims a megabyte of content...
    out.write_all(&[1, 2, 3]); // ...but only 3 bytes follow
    let buf = out.into_vec();
    let mut input = SliceInput::new(&buf);
    assert!(is_corrupt(ByteArraySer.deserialize(&mut input, None)));
}

#[test]
fn negative_framed_length_is_rejected() {
    // Java `new byte[len]` throws on a negative length rather than reinterpreting
    // it as a huge positive u32; we reject it.
    let mut out = DataOutput2::new();
    out.pack_int(-1);
    let buf = out.into_vec();
    let mut input = SliceInput::new(&buf);
    assert!(is_corrupt(ByteArraySer.deserialize(&mut input, None)));
}

#[test]
fn overlong_packed_int_is_rejected_at_five_bytes() {
    // A 32-bit packed varint never needs more than 5 groups; an over-long run is
    // corruption, not a value to accept. Six non-terminated bytes.
    let buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x80];
    let mut input = SliceInput::new(&buf);
    assert!(is_corrupt(input.unpack_int()));
}

#[test]
fn packed_long_and_int_agree_on_overlong_runs() {
    // unpack_long caps at 10 bytes; unpack_long_skip must reject the same run
    // rather than silently skipping what the decoder rejects.
    let buf = [0u8; 11]; // 11 continuation bytes, never terminates
    let mut a = SliceInput::new(&buf);
    assert!(is_corrupt(a.unpack_long()));
    let mut b = SliceInput::new(&buf);
    assert!(is_corrupt(b.unpack_long_skip(1)));
}

#[test]
fn valid_encodings_still_round_trip() {
    // Hardening must not change acceptance of VALID data.
    for v in [0i32, 1, -1, i32::MAX, i32::MIN, 12345, -99999] {
        let mut out = DataOutput2::new();
        IntSer.serialize(&mut out, &v);
        let buf = out.into_vec();
        let mut input = SliceInput::new(&buf);
        assert_eq!(IntSer.deserialize(&mut input, None).unwrap(), v);
    }
    let payload = vec![7u8; 500];
    let mut out = DataOutput2::new();
    ByteArraySer.serialize(&mut out, &payload);
    let buf = out.into_vec();
    let mut input = SliceInput::new(&buf);
    assert_eq!(ByteArraySer.deserialize(&mut input, None).unwrap(), payload);
}
