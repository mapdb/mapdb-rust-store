//! Validation of `IntFormat` and `IntDeltaFormat` (the 32-bit mirror of the
//! Long formats): wire round-trip, object-side `search` vs byte-side
//! `binary_search` coherence, `binary_get` == `get`, and the "leaves input at
//! group end" positioning contract. Includes the tricky cases: negatives,
//! `i32::MIN`, `i32::MAX`, and deltas that overflow `i32` (wrapping arithmetic).

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::int::{IntDeltaFormat, IntFormat};
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

/// A sorted group covering negatives, zero, MIN, MAX and large gaps whose
/// deltas overflow i32 (MIN..0..MAX spans > 2^32).
fn tricky_group() -> Vec<i32> {
    vec![
        i32::MIN,
        i32::MIN + 1,
        -1_000_000,
        -100,
        -1,
        0,
        1,
        2,
        5,
        42,
        1000,
        1_000_000,
        i32::MAX - 1,
        i32::MAX,
    ]
}

fn tricky_probes() -> Vec<i32> {
    vec![
        i32::MIN,
        i32::MIN + 2,
        -1_000_001,
        -100,
        -50,
        0,
        3,
        42,
        999,
        1_000_000,
        i32::MAX - 2,
        i32::MAX,
    ]
}

#[test]
fn int_format_roundtrip_and_coherence() {
    let f = IntFormat;
    let g = tricky_group();
    let bytes = ser(&f, &g);
    // fixed 4-byte stride
    assert_eq!(bytes.len(), g.len() * 4);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g);
    assert_eq!(inp.pos(), bytes.len(), "deserialize leaves at end");
    check_coherence(&f, &g, &tricky_probes());
}

#[test]
fn int_delta_matches_int_values_and_coherence() {
    let f = IntDeltaFormat;
    let g = tricky_group();
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(
        back, g,
        "delta round-trip exact even across overflowing deltas"
    );
    assert_eq!(inp.pos(), bytes.len(), "deserialize leaves at end");
    check_coherence(&f, &g, &tricky_probes());
}

#[test]
fn int_delta_wire_equals_int_values() {
    // Both formats decode to the same object side; cross-check delta decode
    // against IntFormat decode for a non-monotonic (positional) group, which
    // exercises negative and overflow-wrapped deltas.
    let delta = IntDeltaFormat;
    let plain = IntFormat;
    let g: Vec<i32> = vec![0, i32::MAX, i32::MIN, -1, 7, i32::MIN, i32::MAX, 0];
    let db = ser(&delta, &g);
    let pb = ser(&plain, &g);
    let mut di = SliceInput::new(&db);
    let mut pi = SliceInput::new(&pb);
    assert_eq!(
        delta.deserialize(&mut di, g.len()).unwrap(),
        plain.deserialize(&mut pi, g.len()).unwrap()
    );
}

#[test]
fn int_delta_single_element_extremes() {
    // Single-element groups: first delta is the absolute value, zigzagged.
    for v in [i32::MIN, i32::MAX, -1, 0, 1] {
        let f = IntDeltaFormat;
        let g = vec![v];
        let bytes = ser(&f, &g);
        let mut inp = SliceInput::new(&bytes);
        assert_eq!(f.deserialize(&mut inp, 1).unwrap(), g);
        // binary_get / binary_search on the single element
        let mut ig = SliceInput::new(&bytes);
        assert_eq!(f.binary_get(&mut ig, 1, 0).unwrap(), v);
        assert_eq!(ig.pos(), bytes.len());
        let mut sb = SliceInput::new(&bytes);
        assert_eq!(f.binary_search(&v, &mut sb, 1).unwrap(), Ok(0));
        assert_eq!(sb.pos(), bytes.len());
    }
}

#[test]
fn empty_group_binary_search() {
    let fi = IntFormat;
    let fd = IntDeltaFormat;
    let g: Vec<i32> = vec![];
    let bi = ser(&fi, &g);
    let bd = ser(&fd, &g);
    assert!(bi.is_empty() && bd.is_empty());
    let mut ii = SliceInput::new(&bi);
    assert_eq!(fi.binary_search(&5, &mut ii, 0).unwrap(), Err(0));
    assert_eq!(ii.pos(), 0);
    let mut id = SliceInput::new(&bd);
    assert_eq!(fd.binary_search(&5, &mut id, 0).unwrap(), Err(0));
    assert_eq!(id.pos(), 0);
}

#[test]
fn range_cursor_positioning_int_format() {
    // IntFormat uses the shared BinaryGetCursor.
    let f = IntFormat;
    let g: Vec<i32> = vec![-30, -10, 0, 10, 30, 50];
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, g.len(), 1, 4).unwrap();
        let mut seen = vec![];
        while cur.next().unwrap() {
            seen.push((cur.index(), cur.value()));
        }
        assert_eq!(seen, vec![(1, -10), (2, 0), (3, 10)]);
    }
    assert_eq!(inp.pos(), bytes.len(), "cursor leaves input at group end");
}

#[test]
fn range_cursor_positioning_int_delta() {
    // IntDeltaFormat uses its single-pass delta cursor.
    let f = IntDeltaFormat;
    let g: Vec<i32> = vec![10, 20, 30, 40, 50];
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

#[test]
fn range_cursor_full_scan_with_extremes() {
    // Full [0, count) scan over overflowing-delta group leaves input at end.
    let f = IntDeltaFormat;
    let g = tricky_group();
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, g.len(), 0, g.len()).unwrap();
        let mut seen = vec![];
        while cur.next().unwrap() {
            seen.push(cur.value());
        }
        assert_eq!(seen, g);
    }
    assert_eq!(inp.pos(), bytes.len());
}
