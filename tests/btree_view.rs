//! Navigable view layer + columnar scan tests.

use std::sync::Arc;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::ser::columnar::{ColumnType, ColumnarValueFormat};
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::ser::Value;
use mapdb_rust_store::store::StoreOnHeap;

type Map = BTreeMap<StoreOnHeap, LongFormat, LongFormat>;

fn filled(max: usize, n: i64) -> Map {
    let store = Arc::new(StoreOnHeap::new(true));
    let map = BTreeMap::create(store, LongFormat, LongFormat, max).unwrap();
    for i in 0..n {
        map.put(i, i * 10).unwrap();
    }
    map
}

#[test]
fn navigation_entries() {
    let m = filled(4, 100);
    assert_eq!(m.first_entry().unwrap(), Some((0, 0)));
    assert_eq!(m.last_entry().unwrap(), Some((99, 990)));
    assert_eq!(m.floor_entry(&50).unwrap(), Some((50, 500)));
    assert_eq!(m.floor_entry(&999).unwrap(), Some((99, 990)));
    assert_eq!(m.ceiling_entry(&50).unwrap(), Some((50, 500)));
    assert_eq!(m.ceiling_entry(&999).unwrap(), None);
    assert_eq!(m.lower_entry(&50).unwrap(), Some((49, 490)));
    assert_eq!(m.higher_entry(&50).unwrap(), Some((51, 510)));
    assert_eq!(m.lower_entry(&0).unwrap(), None);
    assert_eq!(m.higher_entry(&99).unwrap(), None);
}

#[test]
fn navigation_on_gaps() {
    let store = Arc::new(StoreOnHeap::new(true));
    let m = BTreeMap::create(store, LongFormat, LongFormat, 4).unwrap();
    for i in [10i64, 20, 30, 40, 50] {
        m.put(i, i).unwrap();
    }
    assert_eq!(m.floor_entry(&25).unwrap(), Some((20, 20)));
    assert_eq!(m.ceiling_entry(&25).unwrap(), Some((30, 30)));
    assert_eq!(m.lower_entry(&30).unwrap(), Some((20, 20)));
    assert_eq!(m.higher_entry(&30).unwrap(), Some((40, 40)));
    assert_eq!(m.floor_entry(&5).unwrap(), None);
    assert_eq!(m.ceiling_entry(&55).unwrap(), None);
}

#[test]
fn sub_map_bounds() {
    let m = filled(4, 100);
    // [20, 30) half-open
    let r = m.range(20, 30);
    let ks: Vec<i64> = r.entries().unwrap().into_iter().map(|(k, _)| k).collect();
    assert_eq!(ks, (20..30).collect::<Vec<_>>());
    assert_eq!(r.size_long().unwrap(), 10);
    assert_eq!(r.first_entry().unwrap(), Some((20, 200)));
    assert_eq!(r.last_entry().unwrap(), Some((29, 290)));
    // out-of-range point ops
    assert_eq!(r.get(&15).unwrap(), None);
    assert_eq!(r.get(&25).unwrap(), Some(250));
    assert_eq!(r.get(&30).unwrap(), None); // exclusive upper
                                           // inclusive sub_map [20,30]
    let ri = m.sub_map(20, true, 30, true);
    assert_eq!(ri.last_entry().unwrap(), Some((30, 300)));
}

#[test]
fn head_tail_map() {
    let m = filled(4, 50);
    let head = m.head_map(10, false); // keys < 10
    assert_eq!(head.entries().unwrap().len(), 10);
    assert_eq!(head.last_entry().unwrap(), Some((9, 90)));
    let tail = m.tail_map(45, true); // keys >= 45
    let ks: Vec<i64> = tail
        .entries()
        .unwrap()
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    assert_eq!(ks, vec![45, 46, 47, 48, 49]);
}

#[test]
fn descending_view() {
    let m = filled(4, 20);
    let d = m.descending();
    let ks: Vec<i64> = d.entries().unwrap().into_iter().map(|(k, _)| k).collect();
    assert_eq!(ks, (0..20).rev().collect::<Vec<_>>());
    // orientation-mapped navigation
    assert_eq!(d.first_entry().unwrap(), Some((19, 190)));
    assert_eq!(d.last_entry().unwrap(), Some((0, 0)));
    // descending floor(10) == backing ceiling(10)
    assert_eq!(d.floor_entry(&10).unwrap(), Some((10, 100)));
    assert_eq!(d.ceiling_entry(&10).unwrap(), Some((10, 100)));
    assert_eq!(d.higher_entry(&10).unwrap(), Some((9, 90))); // next smaller
    assert_eq!(d.lower_entry(&10).unwrap(), Some((11, 110))); // prev larger
                                                              // descendingMap().descendingMap() == original
    let dd = d.descending();
    assert!(!dd.is_descending());
}

#[test]
fn descending_sub_map() {
    let m = filled(4, 100);
    // descending subMap(from=80, to=70): backing (70, 80]
    let d = m.descending();
    let sub = d.sub_map(80, true, 70, false);
    let ks: Vec<i64> = sub.entries().unwrap().into_iter().map(|(k, _)| k).collect();
    assert_eq!(ks, vec![80, 79, 78, 77, 76, 75, 74, 73, 72, 71]);
}

#[test]
fn nested_sub_map_never_widens() {
    let m = filled(4, 100);
    let outer = m.sub_map(20, true, 60, false); // [20,60)
                                                // inner tries to widen to [10,80) — must clamp to [20,60)
    let inner = outer.sub_map(20, true, 60, false);
    assert_eq!(inner.first_entry().unwrap(), Some((20, 200)));
    assert_eq!(inner.last_entry().unwrap(), Some((59, 590)));
}

#[test]
#[should_panic(expected = "out of submap range")]
fn sub_map_put_out_of_range_panics() {
    let m = filled(4, 100);
    let r = m.range(20, 30);
    r.put(50, 500).unwrap();
}

#[test]
fn poll_via_view() {
    let m = filled(4, 10);
    assert_eq!(m.pop_first().unwrap(), Some((0, 0)));
    assert_eq!(m.pop_last().unwrap(), Some((9, 90)));
    let r = m.sub_map(2, true, 8, true);
    assert_eq!(r.poll_first_entry().unwrap(), Some((2, 20)));
    assert_eq!(r.poll_last_entry().unwrap(), Some((8, 80)));
    // descending poll maps to opposite end
    let d = m.descending();
    assert_eq!(d.poll_first_entry().unwrap(), Some((7, 70))); // backing greatest remaining
}

#[test]
fn view_clear_bounded() {
    let m = filled(4, 50);
    let r = m.sub_map(10, true, 20, false);
    r.clear().unwrap();
    assert_eq!(m.get(&9).unwrap(), Some(90));
    assert_eq!(m.get(&10).unwrap(), None);
    assert_eq!(m.get(&19).unwrap(), None);
    assert_eq!(m.get(&20).unwrap(), Some(200));
    assert_eq!(m.size_long().unwrap(), 40);
}

#[test]
fn empty_and_inverted_ranges() {
    let m = filled(4, 20);
    // inverted range via sub_map with equal exclusive endpoints
    let r = m.sub_map(5, false, 5, false);
    assert!(r.is_empty().unwrap());
    assert_eq!(r.size_long().unwrap(), 0);
    assert_eq!(r.first_entry().unwrap(), None);
}

// ---------------- columnar scan ----------------

#[test]
fn columnar_single_column_scan() {
    let store = Arc::new(StoreOnHeap::new(true));
    let vf = ColumnarValueFormat::of(&[ColumnType::Long, ColumnType::Int, ColumnType::Short]);
    let map = BTreeMap::create(store, LongFormat, vf, 4).unwrap();
    for i in 0..200i64 {
        map.put(
            i,
            vec![
                Value::Long(i * 100),
                Value::Int(i as i32),
                Value::Short((i % 7) as i16),
            ],
        )
        .unwrap();
    }
    // scan column 1 (Int) over [50, 60]
    let mut got: Vec<(i64, i32)> = Vec::new();
    map.for_each_value_column(Some(50), true, Some(60), true, 1, |k, cell| {
        got.push((*k, cell.as_int().unwrap()));
    })
    .unwrap();
    let want: Vec<(i64, i32)> = (50..=60).map(|i| (i, i as i32)).collect();
    assert_eq!(got, want);

    // full scan of column 0 (Long)
    let mut sum = 0i64;
    map.for_each_value_column(None, true, None, true, 0, |_k, cell| {
        sum += cell.as_long().unwrap();
    })
    .unwrap();
    assert_eq!(sum, (0..200).map(|i| i * 100).sum::<i64>());

    // exclusive bounds
    let mut ks = Vec::new();
    map.for_each_value_column(Some(10), false, Some(14), false, 2, |k, _| ks.push(*k))
        .unwrap();
    assert_eq!(ks, vec![11, 12, 13]);
}
