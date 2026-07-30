//! Validation of the `GroupFormat` trait design (spec D2) against the four
//! reference formats: fixed-stride binary (Long), sequential delta (LongDelta),
//! blob+offset (StringGroup), non-binary (ObjectArray). Round-trip, object vs
//! byte-side coherence, and the range-cursor positioning contract.

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::long::{LongDeltaFormat, LongFormat};
use mapdb_rust_store::ser::object_array::ObjectArrayFormat;
use mapdb_rust_store::ser::serializers::LongSer;
use mapdb_rust_store::ser::string_group::StringGroupFormat;
use mapdb_rust_store::ser::{GroupFormat, SearchResult};

/// Serialize a group, return the bytes.
fn ser<F: GroupFormat>(f: &F, g: &F::Group) -> Vec<u8> {
    let mut out = DataOutput2::new();
    f.serialize(&mut out, g);
    out.into_vec()
}

/// Object-side search vs byte-side binary_search must agree for every probe.
fn check_coherence<F>(f: &F, g: &F::Group, probes: &[F::Elem])
where
    F: GroupFormat,
    F::Elem: std::fmt::Debug + PartialEq,
{
    let count = f.size(g);
    let bytes = ser(f, g);
    for key in probes {
        let obj: SearchResult = f.search(g, key);
        if f.supports_binary() {
            let mut inp = SliceInput::new(&bytes);
            let byte_res = f.binary_search(key, &mut inp, count).unwrap();
            assert_eq!(obj, byte_res, "coherence for key {key:?}");
            // positioning contract: input left at group end
            assert_eq!(inp.pos(), bytes.len(), "binary_search leaves at end");
            // binary_get of every position round-trips get()
            for pos in 0..count {
                let mut ig = SliceInput::new(&bytes);
                let v = f.binary_get(&mut ig, count, pos).unwrap();
                assert_eq!(v, f.get(g, pos), "binary_get pos {pos}");
                assert_eq!(ig.pos(), bytes.len(), "binary_get leaves at end");
            }
        }
    }
}

#[test]
fn long_format_roundtrip_and_coherence() {
    let f = LongFormat;
    let g: Vec<i64> = vec![-100, -1, 0, 1, 2, 5, 42, 1000, i64::MAX];
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
    let probes = vec![-100, -50, 0, 3, 42, 999, i64::MAX, i64::MIN];
    check_coherence(&f, &g, &probes);
}

#[test]
fn long_delta_matches_long_values_and_coherence() {
    let f = LongDeltaFormat;
    let g: Vec<i64> = vec![-100, -1, 0, 1, 2, 5, 42, 1000, i64::MAX];
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
    let probes = vec![-100, -50, 0, 3, 42, 999, i64::MAX];
    check_coherence(&f, &g, &probes);
}

#[test]
fn string_group_roundtrip_and_coherence() {
    let f = StringGroupFormat;
    // sorted by UTF-16 code-unit order; include a supplementary char case
    let mut g: Vec<String> = vec![
        "".into(),
        "a".into(),
        "apple".into(),
        "banana".into(),
        "zebra".into(),
        "\u{FF61}".into(), // U+FF61, BMP, > surrogate range as code unit
    ];
    g.sort_by(|a, b| mapdb_rust_store::ser::util::compare_utf16(a, b));
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
    let probes: Vec<String> = vec![
        "".into(),
        "a".into(),
        "aardvark".into(),
        "apple".into(),
        "app".into(),
        "banana".into(),
        "zzz".into(),
        "\u{FF61}".into(),
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn utf16_vs_supplementary_order() {
    // U+FF61 (0xFF61 as one UTF-16 unit) sorts AFTER U+10000 (surrogate 0xD800..)
    // in UTF-16 order, unlike code-point order. Verify compare_utf16 + byte side.
    let f = StringGroupFormat;
    let supp = "\u{10000}".to_string(); // surrogate pair D800 DC00
    let bmp = "\u{FF61}".to_string();
    assert_eq!(
        mapdb_rust_store::ser::util::compare_utf16(&supp, &bmp),
        std::cmp::Ordering::Less
    );
    let g = vec![supp.clone(), bmp.clone()]; // already in UTF-16 order
    check_coherence(&f, &g, &[supp, bmp, "\u{FFFF}".into()]);
}

#[test]
fn object_array_non_binary() {
    let f = ObjectArrayFormat::new(LongSer);
    let g: Vec<i64> = vec![1, 2, 3, 10, 20];
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
    assert!(!f.supports_binary());
    let mut inp2 = SliceInput::new(&bytes);
    assert!(f.binary_search(&3, &mut inp2, g.len()).is_err());
    assert_eq!(f.search(&g, &10), Ok(3));
    assert_eq!(f.search(&g, &4), Err(3));
}

#[test]
fn range_cursor_positioning() {
    // full scan leaves input at group end; values match get()
    let f = LongDeltaFormat;
    let g: Vec<i64> = vec![10, 20, 30, 40, 50];
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, g.len(), 1, 4).unwrap();
        let mut seen = vec![];
        while cur.next().unwrap() {
            seen.push((cur.index(), cur.value()));
        }
        assert_eq!(seen, vec![(1, 20), (2, 30), (3, 40)]);
    }
    assert_eq!(inp.pos(), bytes.len(), "cursor leaves input at group end");
}
