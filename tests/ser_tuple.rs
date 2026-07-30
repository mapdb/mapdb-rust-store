//! `TupleFormat` validation: wire round-trip, object-vs-byte-side coherence,
//! `binary_get == get`, the "leaves input at group end" positioning contract,
//! and the memcomparable invariant (`compare == encoded byte order`). Includes
//! the tricky cases: prefix tuples, negative int/long ordering via sign flip,
//! strings/bytes containing 0x00, unsigned string order vs a supplementary
//! char, and empty components.

use mapdb_rust_store::io::{DataInput2, DataOutput2, SliceInput};
use mapdb_rust_store::ser::tuple::{TupleComponent, TupleFormat};
use mapdb_rust_store::ser::value::Value;
use mapdb_rust_store::ser::{GroupFormat, SearchResult};
use std::cmp::Ordering;

fn ser<F: GroupFormat>(f: &F, g: &F::Group) -> Vec<u8> {
    let mut out = DataOutput2::new();
    f.serialize(&mut out, g);
    out.into_vec()
}

/// Build a sorted group (ascending tuple order) from a set of tuples.
fn sorted_group(f: &TupleFormat, tuples: &[Vec<Value>]) -> Vec<Vec<Value>> {
    let mut v = tuples.to_vec();
    v.sort_by(|a, b| f.compare(a, b));
    v.dedup_by(|a, b| f.compare(a, b) == Ordering::Equal);
    v
}

/// Object-side `search` vs byte-side `binary_search` must agree for every probe;
/// `binary_get` must equal `get` at every position; both byte-side methods must
/// leave the input at group end.
fn check_coherence(f: &TupleFormat, sorted: &[Vec<Value>], probes: &[Vec<Value>]) {
    let g = f.from_slice(sorted);
    let count = f.size(&g);
    let bytes = ser(f, &g);
    for key in probes {
        let obj: SearchResult = f.search(&g, key);
        let mut inp = SliceInput::new(&bytes);
        let byte_res = f.binary_search(key, &mut inp, count).unwrap();
        assert_eq!(obj, byte_res, "coherence for key {key:?}");
        assert_eq!(inp.pos(), bytes.len(), "binary_search leaves at end");
    }
    for pos in 0..count {
        let mut ig = SliceInput::new(&bytes);
        let v = f.binary_get(&mut ig, count, pos).unwrap();
        assert_eq!(v, f.get(&g, pos), "binary_get pos {pos}");
        assert_eq!(ig.pos(), bytes.len(), "binary_get leaves at end");
    }
}

fn s(x: &str) -> Value {
    Value::Str(x.to_string())
}
fn i(x: i32) -> Value {
    Value::Int(x)
}
fn l(x: i64) -> Value {
    Value::Long(x)
}
fn b(x: &[u8]) -> Value {
    Value::Bytes(x.to_vec())
}

#[test]
fn wire_roundtrip_group() {
    let f = TupleFormat::of(&[TupleComponent::Int, TupleComponent::Str]);
    let sorted = sorted_group(
        &f,
        &[
            vec![i(1), s("a")],
            vec![i(1), s("b")],
            vec![i(2), s("")],
            vec![i(-5), s("z")],
        ],
    );
    let g = f.from_slice(&sorted);
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    let back = f.deserialize(&mut inp, g.len()).unwrap();
    assert_eq!(back, g, "group encoding round-trips");
    assert_eq!(inp.pos(), bytes.len());
    // decoded tuples equal originals
    for (pos, t) in sorted.iter().enumerate() {
        assert_eq!(&f.get(&g, pos), t);
    }
}

#[test]
fn element_serializer_roundtrip() {
    let f = TupleFormat::of(&[
        TupleComponent::Long,
        TupleComponent::Bytes,
        TupleComponent::Int,
    ]);
    let ser_el = f.element();
    let cases = vec![
        vec![l(-1), b(&[0, 1, 2, 0]), i(7)],
        vec![l(i64::MIN), b(&[]), i(i32::MAX)],
        vec![l(0)],           // prefix tuple, arity 1
        vec![],               // empty prefix tuple, arity 0
        vec![l(9), b(b"hi")], // prefix tuple, arity 2
    ];
    for t in &cases {
        let mut out = DataOutput2::new();
        ser_el.serialize(&mut out, t);
        let bytes = out.into_vec();
        let mut inp = SliceInput::new(&bytes);
        let back = ser_el.deserialize(&mut inp, None).unwrap();
        assert_eq!(&back, t, "element round-trips {t:?}");
        assert_eq!(inp.pos(), bytes.len(), "element self-delimits");
        assert!(ser_el.equals(t, &back));
    }
    assert!(ser_el.equals_by_serialized_bytes());
    assert!(!ser_el.natural_order());
}

#[test]
fn prefix_tuples_shorter_is_smaller() {
    let f = TupleFormat::of(&[TupleComponent::Int, TupleComponent::Int]);
    // (a) < (a,b): prefix tuple sorts before the extended tuple
    assert_eq!(f.compare(&vec![i(5)], &vec![i(5), i(0)]), Ordering::Less);
    assert_eq!(f.compare(&vec![i(5), i(0)], &vec![i(5)]), Ordering::Greater);
    assert_eq!(f.compare(&vec![], &vec![i(5)]), Ordering::Less);
    let sorted = sorted_group(
        &f,
        &[
            vec![],
            vec![i(5)],
            vec![i(5), i(-1)],
            vec![i(5), i(0)],
            vec![i(6)],
        ],
    );
    let probes = vec![
        vec![],
        vec![i(5)],
        vec![i(5), i(-1)],
        vec![i(5), i(0)],
        vec![i(5), i(1)], // not present
        vec![i(6)],
        vec![i(7)], // not present
    ];
    check_coherence(&f, &sorted, &probes);
}

#[test]
fn negative_int_long_ordering_via_sign_flip() {
    let f = TupleFormat::of(&[TupleComponent::Int]);
    // signed order preserved through the sign-bit-flip encoding
    let vals = [i32::MIN, -1000, -1, 0, 1, 1000, i32::MAX];
    let sorted = sorted_group(&f, &vals.iter().map(|&v| vec![i(v)]).collect::<Vec<_>>());
    // already ascending signed
    let decoded: Vec<i32> = sorted.iter().map(|t| t[0].as_int().unwrap()).collect();
    assert_eq!(decoded, vals.to_vec(), "int order == signed order");
    let probes: Vec<Vec<Value>> = [i32::MIN, -500, -1, 0, 42, i32::MAX, 7]
        .iter()
        .map(|&v| vec![i(v)])
        .collect();
    check_coherence(&f, &sorted, &probes);

    let fl = TupleFormat::of(&[TupleComponent::Long]);
    let lvals = [i64::MIN, -1, 0, 1, i64::MAX];
    let lsorted = sorted_group(&fl, &lvals.iter().map(|&v| vec![l(v)]).collect::<Vec<_>>());
    let ldec: Vec<i64> = lsorted.iter().map(|t| t[0].as_long().unwrap()).collect();
    assert_eq!(ldec, lvals.to_vec(), "long order == signed order");
    let lprobes: Vec<Vec<Value>> = [i64::MIN, -100, 0, 5, i64::MAX]
        .iter()
        .map(|&v| vec![l(v)])
        .collect();
    check_coherence(&fl, &lsorted, &lprobes);
}

#[test]
fn strings_and_bytes_with_zero_bytes_escaping() {
    let f = TupleFormat::of(&[TupleComponent::Bytes, TupleComponent::Int]);
    // payloads containing 0x00 must round-trip and stay ordered after escaping;
    // a trailing int component proves the escape terminator delimits correctly.
    let tuples = vec![
        vec![b(&[]), i(0)],
        vec![b(&[0x00]), i(1)],
        vec![b(&[0x00, 0x00]), i(2)],
        vec![b(&[0x00, 0xFF]), i(3)],
        vec![b(&[0x00, 0xFF, 0x00]), i(4)],
        vec![b(&[0x01]), i(5)],
        vec![b(&[0xFF]), i(6)],
        vec![b(&[0xFF, 0x00]), i(7)],
    ];
    let sorted = sorted_group(&f, &tuples);
    // round-trip each through the byte group + verify escaping preserves payload
    let g = f.from_slice(&sorted);
    for (pos, t) in sorted.iter().enumerate() {
        assert_eq!(&f.get(&g, pos), t, "0x00 payload round-trips");
    }
    let probes = {
        let mut p = tuples.clone();
        p.push(vec![b(&[0x00, 0x01]), i(9)]); // not present
        p.push(vec![b(&[0x02]), i(0)]);
        p
    };
    check_coherence(&f, &sorted, &probes);

    // strings containing NUL characters
    let fs = TupleFormat::of(&[TupleComponent::Str, TupleComponent::Str]);
    let stuples = vec![
        vec![s("a\u{0}b"), s("x")],
        vec![s("a"), s("\u{0}")],
        vec![s(""), s("")],
        vec![s("a\u{0}"), s("y")],
    ];
    let ssorted = sorted_group(&fs, &stuples);
    let sg = fs.from_slice(&ssorted);
    for (pos, t) in ssorted.iter().enumerate() {
        assert_eq!(&fs.get(&sg, pos), t, "NUL-containing string round-trips");
    }
    check_coherence(&fs, &ssorted, &stuples);
}

#[test]
fn unsigned_utf8_string_order_vs_supplementary_char() {
    // STRING order is UTF-8 (code-point) order, NOT UTF-16 order.
    // U+FF61 (UTF-8 EF BD A1) < U+10000 (UTF-8 F0 90 80 80) in code-point order,
    // the reverse of UTF-16 code-unit order (surrogate 0xD800 < 0xFF61).
    let f = TupleFormat::of(&[TupleComponent::Str]);
    let bmp = "\u{FF61}".to_string();
    let supp = "\u{10000}".to_string();
    assert_eq!(
        f.compare(
            &vec![Value::Str(bmp.clone())],
            &vec![Value::Str(supp.clone())]
        ),
        Ordering::Less,
        "code-point order: U+FF61 < U+10000 (differs from UTF-16)"
    );
    let sorted = sorted_group(
        &f,
        &[
            vec![s("")],
            vec![s("a")],
            vec![s("z")],
            vec![Value::Str(bmp.clone())],
            vec![Value::Str(supp.clone())],
        ],
    );
    // ascending UTF-8 order places bmp before supp
    let last_two: Vec<&str> = sorted
        .iter()
        .rev()
        .take(2)
        .map(|t| t[0].as_str().unwrap())
        .collect();
    assert_eq!(last_two, vec![supp.as_str(), bmp.as_str()]);
    let probes = vec![
        vec![s("")],
        vec![s("a")],
        vec![Value::Str(bmp)],
        vec![Value::Str(supp)],
        vec![s("\u{FFFF}")],
    ];
    check_coherence(&f, &sorted, &probes);
}

#[test]
fn empty_components_and_empty_group() {
    let f = TupleFormat::of(&[TupleComponent::Str, TupleComponent::Bytes]);
    // empty group binary_search returns Err(0), leaves input at end
    let g = f.empty();
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    assert_eq!(f.binary_search(&vec![s("x")], &mut inp, 0).unwrap(), Err(0));
    assert_eq!(inp.pos(), bytes.len());

    // tuple of two empty components encodes to two terminators (4 bytes)
    let e = f.from_slice(&[vec![s(""), b(&[])]]);
    assert_eq!(e[0], vec![0, 0, 0, 0]);
    let sorted = sorted_group(
        &f,
        &[
            vec![s(""), b(&[])],
            vec![s(""), b(&[0])],
            vec![s("a"), b(&[])],
        ],
    );
    check_coherence(&f, &sorted, &sorted.clone());
}

#[test]
fn memcomparable_invariant_and_coherence_fuzz() {
    // Random tuples: assert compare(a,b) sign == encode(a).cmp(encode(b)) sign
    // (the memcomparable invariant), then run full object/byte coherence.
    let f = TupleFormat::of(&[
        TupleComponent::Int,
        TupleComponent::Str,
        TupleComponent::Long,
    ]);
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let rand_tuple = |next: &mut dyn FnMut() -> u64| -> Vec<Value> {
        let arity = (next() % 4) as usize; // 0..=3 prefix tuples
        let mut t = Vec::new();
        if arity >= 1 {
            // small int range so ties happen
            t.push(Value::Int(((next() % 7) as i32) - 3));
        }
        if arity >= 2 {
            let n = (next() % 3) as usize;
            let alphabet = [b'a', b'\x00', 0xC3]; // includes NUL + a UTF-8 lead-ish byte
            let mut sbytes = Vec::new();
            for _ in 0..n {
                sbytes.push(alphabet[(next() % 3) as usize]);
            }
            // keep it valid UTF-8: use from_utf8_lossy so bad seqs normalize
            t.push(Value::Str(String::from_utf8_lossy(&sbytes).into_owned()));
        }
        if arity >= 3 {
            t.push(Value::Long(((next() % 5) as i64) - 2));
        }
        t
    };

    let mut pool: Vec<Vec<Value>> = Vec::new();
    for _ in 0..300 {
        pool.push(rand_tuple(&mut next));
    }

    // memcomparable invariant over all pairs
    for a in &pool {
        for bb in &pool {
            let cmp = f.compare(a, bb);
            let ea = f.from_slice(std::slice::from_ref(a))[0].clone();
            let eb = f.from_slice(std::slice::from_ref(bb))[0].clone();
            let byte_cmp = ea.cmp(&eb);
            assert_eq!(
                cmp, byte_cmp,
                "memcomparable invariant: compare {a:?} vs {bb:?}"
            );
        }
    }

    let sorted = sorted_group(&f, &pool);
    check_coherence(&f, &sorted, &pool);
}

#[test]
fn range_cursor_positioning() {
    let f = TupleFormat::of(&[TupleComponent::Int, TupleComponent::Str]);
    let sorted = sorted_group(
        &f,
        &[
            vec![i(1), s("a")],
            vec![i(2), s("bb")],
            vec![i(3), s("")],
            vec![i(4), s("d\u{0}")],
            vec![i(5), s("e")],
        ],
    );
    let g = f.from_slice(&sorted);
    let bytes = ser(&f, &g);
    let mut inp = SliceInput::new(&bytes);
    {
        let mut cur = f.range_cursor(&mut inp, g.len(), 1, 4).unwrap();
        let mut seen = Vec::new();
        while cur.next().unwrap() {
            seen.push((cur.index(), cur.value()));
        }
        assert_eq!(
            seen,
            vec![
                (1, sorted[1].clone()),
                (2, sorted[2].clone()),
                (3, sorted[3].clone()),
            ]
        );
    }
    assert_eq!(inp.pos(), bytes.len(), "cursor leaves input at group end");
}

#[test]
fn corrupt_bytes_error_not_panic() {
    let f = TupleFormat::of(&[TupleComponent::Str]);
    // unterminated string component (no 0x00 0x00 terminator)
    let e = vec![b'a', b'b'];
    // decode via element serializer: packInt(len) + bytes
    let mut out = DataOutput2::new();
    out.pack_int(e.len() as i32);
    out.write_all(&e);
    let framed = out.into_vec();
    let mut fi = SliceInput::new(&framed);
    assert!(
        f.element().deserialize(&mut fi, None).is_err(),
        "unterminated component is corruption, not a panic"
    );
}
