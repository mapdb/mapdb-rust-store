//! Validation of `StringPrefixFormat` (front-coded / LevelDB block style):
//! wire round-trip, object-side `search` vs byte-side `binary_search` coherence,
//! `binary_get == get` for every position, and the "leaves input at group end"
//! positioning contract. Covers the tricky cases: long shared prefixes, the
//! restart-boundary sizes (15/16/17/33), a supplementary-plane string, and the
//! empty-string element.

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::string_prefix::StringPrefixFormat;
use mapdb_rust_store::ser::util::compare_utf16;
use mapdb_rust_store::ser::{GroupFormat, SearchResult};

fn ser<F: GroupFormat>(f: &F, g: &F::Group) -> Vec<u8> {
    let mut out = DataOutput2::new();
    f.serialize(&mut out, g);
    out.into_vec()
}

/// Object-side search vs byte-side binary_search must agree for every probe;
/// binary_get must match get for every position; both leave input at group end.
fn check_coherence(f: &StringPrefixFormat, g: &Vec<String>, probes: &[String]) {
    let count = f.size(g);
    let bytes = ser(f, g);

    // wire round-trip
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, count).unwrap();
    assert_eq!(&back, g, "deserialize round-trip");

    for key in probes {
        let obj: SearchResult = f.search(g, key);
        let mut inp = SliceInput::new(&bytes);
        let byte_res = f.binary_search(key, &mut inp, count).unwrap();
        assert_eq!(obj, byte_res, "coherence for key {key:?}");
        assert_eq!(
            inp.pos(),
            bytes.len(),
            "binary_search leaves at end (key {key:?})"
        );
    }

    for pos in 0..count {
        let mut ig = SliceInput::new(&bytes);
        let v = f.binary_get(&mut ig, count, pos).unwrap();
        assert_eq!(v, f.get(g, pos), "binary_get pos {pos}");
        assert_eq!(
            ig.pos(),
            bytes.len(),
            "binary_get leaves at end (pos {pos})"
        );
    }
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort_by(|a, b| compare_utf16(a, b));
    v.dedup();
    v
}

#[test]
fn shared_long_prefixes() {
    let f = StringPrefixFormat;
    let g = sorted(vec![
        "apple".into(),
        "application".into(),
        "apply".into(),
        "appliance".into(),
        "app".into(),
        "banana".into(),
    ]);
    let probes: Vec<String> = vec![
        "".into(),
        "app".into(),
        "appl".into(),
        "apple".into(),
        "applf".into(),
        "application".into(),
        "apply".into(),
        "appliance".into(),
        "aardvark".into(),
        "banana".into(),
        "zzz".into(),
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn empty_string_element() {
    let f = StringPrefixFormat;
    let g = sorted(vec!["".into(), "a".into(), "ab".into(), "b".into()]);
    let probes: Vec<String> = vec![
        "".into(),
        "a".into(),
        "ab".into(),
        "abc".into(),
        "b".into(),
        "c".into(),
    ];
    check_coherence(&f, &g, &probes);
}

#[test]
fn supplementary_plane() {
    let f = StringPrefixFormat;
    // U+10000 (surrogate pair D800 DC00) sorts BEFORE U+FF61 in UTF-16 order.
    let g = sorted(vec![
        "a".into(),
        "\u{10000}".into(),
        "\u{10000}x".into(),
        "\u{1F600}".into(), // emoji, another supplementary char
        "\u{FF61}".into(),
        "z".into(),
    ]);
    let probes: Vec<String> = vec![
        "a".into(),
        "\u{10000}".into(),
        "\u{10000}x".into(),
        "\u{1F600}".into(),
        "\u{FF61}".into(),
        "\u{FFFF}".into(),
        "z".into(),
    ];
    check_coherence(&f, &g, &probes);
}

/// Sizes straddling the restart interval K=16 exercise the restart table and the
/// roll-forward at interval boundaries (last entry of an interval, first of the next).
#[test]
fn restart_boundaries() {
    let f = StringPrefixFormat;
    for &n in &[0usize, 1, 15, 16, 17, 31, 32, 33, 48] {
        // build n sorted, distinct, shared-prefix-heavy keys
        let mut v: Vec<String> = (0..n).map(|i| format!("key{:05}", i)).collect();
        v = sorted(v);
        assert_eq!(v.len(), n, "distinct keys for n={n}");

        // probes: every key, every gap (key+"a"), and out-of-range ends
        let mut probes: Vec<String> = Vec::new();
        probes.push("".into());
        probes.push("zzzzzz".into());
        for k in &v {
            probes.push(k.clone());
            probes.push(format!("{k}a")); // just after k
        }
        check_coherence(&f, &v, &probes);
    }
}

#[test]
fn range_cursor_positioning() {
    let f = StringPrefixFormat;
    let g = sorted(vec![
        "alpha".into(),
        "alphabet".into(),
        "beta".into(),
        "gamma".into(),
        "gammon".into(),
    ]);
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, g.len(), 1, 4).unwrap();
        let mut seen: Vec<(usize, String)> = vec![];
        while cur.next().unwrap() {
            seen.push((cur.index(), cur.value()));
        }
        let expected: Vec<(usize, String)> = (1..4).map(|i| (i, g[i].clone())).collect();
        assert_eq!(seen, expected);
    }
    assert_eq!(inp.pos(), bytes.len(), "cursor leaves input at group end");
}

#[test]
fn empty_group() {
    let f = StringPrefixFormat;
    let g: Vec<String> = vec![];
    let bytes = ser(&f, &g);
    // deserialize
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, 0).unwrap();
    assert!(back.is_empty());
    // search on empty -> insertion point 0
    let mut inp = SliceInput::new(&bytes);
    assert_eq!(
        f.binary_search(&"x".to_string(), &mut inp, 0).unwrap(),
        Err(0)
    );
    assert_eq!(
        inp.pos(),
        bytes.len(),
        "binary_search leaves at end (empty)"
    );
}

/// A large group with many restart intervals and heavy shared prefixes; probes
/// interleave hits and misses across interval boundaries.
#[test]
fn large_group_many_intervals() {
    let f = StringPrefixFormat;
    let mut v: Vec<String> = (0..200)
        .map(|i| format!("prefix/shared/segment/{:04}/leaf", i))
        .collect();
    v = sorted(v);
    let mut probes: Vec<String> = Vec::new();
    for k in v.iter().step_by(7) {
        probes.push(k.clone());
        probes.push(format!("{k}Z"));
    }
    probes.push("prefix".into());
    probes.push("zzzz".into());
    check_coherence(&f, &v, &probes);
}
