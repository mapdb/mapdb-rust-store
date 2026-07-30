//! Validation of the three fixed-stride scalar group formats (`ShortFormat`,
//! `CharFormat`, `UuidFormat`): wire round-trip, object-side `search` vs
//! byte-side `binary_search` coherence over many probes, `binary_get == get`
//! for every position, the "leaves input at group end" contract, and the
//! range-cursor positioning contract. Includes the tricky cases: negative
//! shorts (signed order), chars > 0x7FFF (unsigned order), and UUIDs with
//! negative msb and/or lsb (signed msb-then-lsb order).

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::scalar::{CharFormat, ShortFormat, UuidFormat};
use mapdb_rust_store::ser::serializers::Uuid;
use mapdb_rust_store::ser::{GroupFormat, SearchResult};

/// Serialize a group, return the bytes.
fn ser<F: GroupFormat>(f: &F, g: &F::Group) -> Vec<u8> {
    let mut out = DataOutput2::new();
    f.serialize(&mut out, g);
    out.into_vec()
}

/// Object-side search vs byte-side binary_search must agree for every probe;
/// also checks binary_get==get and both positioning contracts.
fn check_coherence<F>(f: &F, g: &F::Group, probes: &[F::Elem])
where
    F: GroupFormat,
    F::Elem: std::fmt::Debug + PartialEq,
{
    let count = f.size(g);
    let bytes = ser(f, g);
    for key in probes {
        let obj: SearchResult = f.search(g, key);
        assert!(f.supports_binary());
        let mut inp = SliceInput::new(&bytes);
        let byte_res = f.binary_search(key, &mut inp, count).unwrap();
        assert_eq!(obj, byte_res, "coherence for key {key:?}");
        assert_eq!(inp.pos(), bytes.len(), "binary_search leaves at end");
        for pos in 0..count {
            let mut ig = SliceInput::new(&bytes);
            let v = f.binary_get(&mut ig, count, pos).unwrap();
            assert_eq!(v, f.get(g, pos), "binary_get pos {pos}");
            assert_eq!(ig.pos(), bytes.len(), "binary_get leaves at end");
        }
    }
}

/// Round-trip: serialize then deserialize equals original.
fn roundtrip<F>(f: &F, g: &F::Group)
where
    F: GroupFormat,
    F::Group: std::fmt::Debug + PartialEq,
{
    let bytes = ser(f, g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, f.size(g)).unwrap();
    assert_eq!(&back, g);
    assert_eq!(inp.pos(), bytes.len(), "deserialize consumes all bytes");
}

/// Full range-cursor scan yields get()-equal values in order and leaves input
/// at group end.
fn range_scan<F>(f: &F, g: &F::Group)
where
    F: GroupFormat,
    F::Elem: std::fmt::Debug + PartialEq,
{
    let count = f.size(g);
    let bytes = ser(f, g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, count, 0, count).unwrap();
        let mut i = 0;
        while cur.next().unwrap() {
            assert_eq!(cur.index(), i);
            assert_eq!(cur.value(), f.get(g, i), "cursor value pos {i}");
            i += 1;
        }
        assert_eq!(i, count, "cursor visited all elements");
    }
    assert_eq!(inp.pos(), bytes.len(), "cursor leaves input at group end");

    // sub-range [from, to) positioning + values
    if count >= 3 {
        let (from, to) = (1usize, count - 1);
        let mut inp2 = SliceInput::new(&bytes);
        {
            let mut cur = f.range_cursor(&mut inp2, count, from, to).unwrap();
            let mut i = from;
            while cur.next().unwrap() {
                assert_eq!(cur.index(), i);
                assert_eq!(cur.value(), f.get(g, i));
                i += 1;
            }
            assert_eq!(i, to);
        }
        assert_eq!(inp2.pos(), bytes.len(), "sub-range cursor leaves at end");
    }
}

#[test]
fn short_format_roundtrip_coherence_and_cursor() {
    let f = ShortFormat;
    // signed-ascending; the negative half sorts BEFORE the non-negative half.
    let g: Vec<i16> = vec![i16::MIN, -30000, -1000, -1, 0, 1, 42, 1000, 30000, i16::MAX];
    roundtrip(&f, &g);
    range_scan(&f, &g);
    let probes: Vec<i16> = vec![
        i16::MIN,
        -30001,
        -30000,
        -500,
        -1,
        0,
        1,
        41,
        42,
        999,
        30000,
        i16::MAX,
        // ensure a negative key does not accidentally sort after positives
        -12345,
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn char_format_roundtrip_coherence_and_cursor() {
    let f = CharFormat;
    // unsigned 0..65535; values > 0x7FFF must sort AFTER smaller ones.
    let g: Vec<u16> = vec![0, 1, 0x00FF, 0x0100, 0x7FFF, 0x8000, 0xABCD, 0xFFFE, 0xFFFF];
    roundtrip(&f, &g);
    range_scan(&f, &g);
    let probes: Vec<u16> = vec![
        0, 1, 0x0080, 0x00FF, 0x0100, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xABCD, 0xC000, 0xFFFE,
        0xFFFF,
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn uuid_format_roundtrip_coherence_and_cursor() {
    let f = UuidFormat;
    // signed (msb, then lsb): negative msb (high bit set) sorts BEFORE
    // non-negative msb; within equal msb, negative lsb sorts before positive.
    let mut g: Vec<Uuid> = vec![
        Uuid::new(i64::MIN, 0),
        Uuid::new(-1, i64::MIN),
        Uuid::new(-1, -1),
        Uuid::new(-1, 0),
        Uuid::new(-1, i64::MAX),
        Uuid::new(0, i64::MIN),
        Uuid::new(0, -1),
        Uuid::new(0, 0),
        Uuid::new(0, 1),
        Uuid::new(5, -100),
        Uuid::new(5, 100),
        Uuid::new(i64::MAX, i64::MAX),
    ];
    g.sort(); // Uuid: Ord == signed msb then lsb
              // sanity: the least element must have the most-negative msb
    assert_eq!(g[0].msb, i64::MIN);
    roundtrip(&f, &g);
    range_scan(&f, &g);
    let probes: Vec<Uuid> = vec![
        Uuid::new(i64::MIN, i64::MIN),
        Uuid::new(i64::MIN, 0),
        Uuid::new(-1, -1),
        Uuid::new(-1, 0),
        Uuid::new(-1, 50),
        Uuid::new(0, i64::MIN),
        Uuid::new(0, 0),
        Uuid::new(0, 2),
        Uuid::new(5, -100),
        Uuid::new(5, 0),
        Uuid::new(5, 100),
        Uuid::new(6, 0),
        Uuid::new(i64::MAX, i64::MAX),
        // a key strictly greater than everything
        Uuid::new(i64::MAX, i64::MIN),
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn empty_and_single_groups() {
    // empty groups: binary_search on any key returns Err(0), input at end (0).
    let sf = ShortFormat;
    let empty: Vec<i16> = vec![];
    let bytes = ser(&sf, &empty);
    assert!(bytes.is_empty());
    let mut inp = SliceInput::new(&bytes);
    assert_eq!(sf.binary_search(&5, &mut inp, 0).unwrap(), Err(0));
    assert_eq!(inp.pos(), 0);

    check_coherence(&sf, &vec![7i16], &[6, 7, 8, i16::MIN, i16::MAX]);
    check_coherence(&CharFormat, &vec![0x8000u16], &[0, 0x8000, 0xFFFF]);
    check_coherence(
        &UuidFormat,
        &vec![Uuid::new(-1, 5)],
        &[
            Uuid::new(-1, 4),
            Uuid::new(-1, 5),
            Uuid::new(-1, 6),
            Uuid::new(0, 0),
        ],
    );
}
