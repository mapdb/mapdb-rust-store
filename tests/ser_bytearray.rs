//! `ByteArrayFormat` / `ByteArrayPrefixFormat` validation: wire round-trip,
//! object-side `search` vs byte-side `binary_search` coherence, `binary_get ==
//! get` at every position, the "leaves input at group end" contract, and the
//! UNSIGNED (`memcmp`) ordering the formats exist for.
//!
//! Tricky cases exercised: empty byte arrays, embedded `0x00`, high bytes
//! (`0x80+`) proving unsigned (not signed) order, restart boundaries at 15/16/17
//! elements, and long shared-prefix keys (the front-coding target).

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::bytearray::{ByteArrayFormat, ByteArrayPrefixFormat};
use mapdb_rust_store::ser::{GroupFormat, SearchResult};

fn ser<F: GroupFormat>(f: &F, g: &F::Group) -> Vec<u8> {
    let mut out = DataOutput2::new();
    f.serialize(&mut out, g);
    out.into_vec()
}

/// Object-side `search` vs byte-side `binary_search` must agree for every probe;
/// `binary_get` must equal `get`; both byte-side entries must leave the input at
/// group end. Mirrors `check_coherence` in `ser_reference.rs`.
fn check_coherence<F>(f: &F, g: &F::Group, probes: &[F::Elem])
where
    F: GroupFormat,
    F::Elem: std::fmt::Debug + PartialEq,
{
    let count = f.size(g);
    let bytes = ser(f, g);
    // binary_get of every position round-trips get() and leaves input at end.
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
    for key in probes {
        let obj: SearchResult = f.search(g, key);
        let mut inp = SliceInput::new(&bytes);
        let byte_res = f.binary_search(key, &mut inp, count).unwrap();
        assert_eq!(obj, byte_res, "coherence for key {key:?}");
        assert_eq!(
            inp.pos(),
            bytes.len(),
            "binary_search leaves at end for {key:?}"
        );
    }
}

fn roundtrip<F>(f: &F, g: &F::Group)
where
    F: GroupFormat,
    F::Elem: std::fmt::Debug + PartialEq,
    F::Group: std::fmt::Debug + PartialEq,
{
    let bytes = ser(f, g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, f.size(g)).unwrap();
    assert_eq!(&back, g, "wire round-trip");
    assert_eq!(
        inp.pos(),
        bytes.len(),
        "deserialize consumes exactly the group"
    );
}

fn b(s: &[u8]) -> Vec<u8> {
    s.to_vec()
}

/// A tricky sorted-by-unsigned group: empty, `0x00`, ascending, and high bytes.
fn tricky_group() -> Vec<Vec<u8>> {
    let mut g = vec![
        b(&[]),           // empty byte array
        b(&[0x00]),       // single NUL
        b(&[0x00, 0x00]), // NUL is a valid interior byte
        b(&[0x00, 0x01]),
        b(&[0x01]),
        b(&[0x7f]),
        b(&[0x80]), // high byte: unsigned puts 0x80 AFTER 0x7f (signed would flip)
        b(&[0x80, 0x00]),
        b(&[0xfe]),
        b(&[0xff]),
        b(&[0xff, 0x00]),
        b(&[0xff, 0xff]),
    ];
    g.sort(); // Vec<u8> Ord == unsigned lexicographic
    g
}

fn tricky_probes() -> Vec<Vec<u8>> {
    vec![
        b(&[]),
        b(&[0x00]),
        b(&[0x00, 0x00, 0x00]),
        b(&[0x01]),
        b(&[0x40]),
        b(&[0x7f]),
        b(&[0x80]),
        b(&[0x80, 0x00]),
        b(&[0x80, 0x01]),
        b(&[0xfe, 0xff]),
        b(&[0xff]),
        b(&[0xff, 0xff]),
        b(&[0xff, 0xff, 0xff]), // above everything
    ]
}

/// Long shared-prefix keys — the front-coding target. `n` distinct keys, sorted.
fn prefix_group(n: usize) -> Vec<Vec<u8>> {
    let mut g = Vec::new();
    for i in 0..n {
        // "common/prefix/key" + a 3-digit suffix; long shared prefix, plus a NUL
        let mut v = b(b"common/prefix/key");
        v.push(0x00);
        v.extend_from_slice(format!("{i:03}").as_bytes());
        g.push(v);
    }
    g.sort();
    g
}

// ---------------------------------------------------------------------------
// ByteArrayFormat
// ---------------------------------------------------------------------------

#[test]
fn byte_array_format_roundtrip_and_coherence() {
    let f = ByteArrayFormat;
    let g = tricky_group();
    roundtrip(&f, &g);
    check_coherence(&f, &g, &tricky_probes());
}

#[test]
fn byte_array_format_empty() {
    let f = ByteArrayFormat;
    let g: Vec<Vec<u8>> = Vec::new();
    roundtrip(&f, &g);
    // empty group: any probe is "not found" at insertion point 0.
    check_coherence(&f, &g, &[b(&[]), b(&[0x00]), b(&[0xff])]);
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    assert_eq!(f.binary_search(&b(&[0x05]), &mut inp, 0).unwrap(), Err(0));
    assert_eq!(inp.pos(), bytes.len());
}

#[test]
fn byte_array_format_unsigned_order() {
    // 0x7f < 0x80 under unsigned order; a signed comparator would rank 0x80
    // (as -128) BELOW 0x7f. Confirm both sides agree on the unsigned answer.
    let f = ByteArrayFormat;
    let g = vec![b(&[0x7f]), b(&[0x80])]; // already unsigned-sorted
    check_coherence(&f, &g, &[b(&[0x7f]), b(&[0x80]), b(&[0x00]), b(&[0xff])]);
    assert_eq!(f.search(&g, &b(&[0x80])), Ok(1));
    assert_eq!(f.search(&g, &b(&[0x7f])), Ok(0));
}

// ---------------------------------------------------------------------------
// ByteArrayPrefixFormat
// ---------------------------------------------------------------------------

#[test]
fn byte_array_prefix_roundtrip_and_coherence() {
    let f = ByteArrayPrefixFormat;
    let g = tricky_group();
    roundtrip(&f, &g);
    check_coherence(&f, &g, &tricky_probes());
}

#[test]
fn byte_array_prefix_empty() {
    let f = ByteArrayPrefixFormat;
    let g: Vec<Vec<u8>> = Vec::new();
    roundtrip(&f, &g);
    check_coherence(&f, &g, &[b(&[]), b(&[0x00]), b(&[0xff])]);
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    assert_eq!(f.binary_search(&b(&[0x05]), &mut inp, 0).unwrap(), Err(0));
    assert_eq!(inp.pos(), bytes.len());
}

#[test]
fn byte_array_prefix_restart_boundaries() {
    // K = 16: exercise the restart-interval edges at 15/16/17 elements.
    let f = ByteArrayPrefixFormat;
    for n in [1usize, 15, 16, 17, 32, 33, 50] {
        let g = prefix_group(n);
        roundtrip(&f, &g);
        // probes: every stored key, plus below/between/above and near boundaries.
        let mut probes = g.clone();
        probes.push(b(b"a")); // below everything
        probes.push(b(b"zzz")); // above everything
        probes.push(b(b"common/prefix/key")); // prefix of every key (no NUL) -> sorts before
                                              // synthetic near-boundary keys
        for i in [0usize, 14, 15, 16, 30] {
            let mut v = b(b"common/prefix/key");
            v.push(0x00);
            v.extend_from_slice(format!("{i:03}").as_bytes());
            v.push(0x00); // one byte longer than a stored key -> "just above" it
            probes.push(v);
        }
        check_coherence(&f, &g, &probes);
    }
}

#[test]
fn byte_array_prefix_unsigned_order_and_zeros() {
    let f = ByteArrayPrefixFormat;
    let mut g = vec![
        b(&[0x00]),
        b(&[0x00, 0x80]),
        b(&[0x00, 0xff]),
        b(&[0x7f, 0x00]),
        b(&[0x80]),
        b(&[0x80, 0x00, 0x00]),
        b(&[0xff, 0xff]),
    ];
    g.sort();
    roundtrip(&f, &g);
    check_coherence(
        &f,
        &g,
        &[
            b(&[]),
            b(&[0x00]),
            b(&[0x00, 0x00]),
            b(&[0x00, 0x81]),
            b(&[0x7f]),
            b(&[0x80]),
            b(&[0x80, 0x00]),
            b(&[0xff, 0xff, 0x00]),
        ],
    );
}

/// The two formats must agree element-for-element and search-for-search (same
/// group object, same unsigned order) — a cross-check of the shared invariant.
#[test]
fn both_formats_agree() {
    let fa = ByteArrayFormat;
    let fp = ByteArrayPrefixFormat;
    let g = prefix_group(40);
    for key in tricky_probes().iter().chain(g.iter()) {
        assert_eq!(
            fa.search(&g, key),
            fp.search(&g, key),
            "object search agree for {key:?}"
        );
    }
    let ba = ser(&fa, &g);
    let bp = ser(&fp, &g);
    for key in g.iter() {
        let mut ia = SliceInput::new(&ba);
        let mut ip = SliceInput::new(&bp);
        assert_eq!(
            fa.binary_search(key, &mut ia, g.len()).unwrap(),
            fp.binary_search(key, &mut ip, g.len()).unwrap(),
            "byte search agree for {key:?}"
        );
    }
}
