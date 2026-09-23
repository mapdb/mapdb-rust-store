//! Regression guard for the byte-side point get (S1).
//!
//! On a format with `supports_binary()`, `GetAction::on_bytes` searches the key
//! group with `binary_search` and, on a hit, reads the value with `binary_get`.
//! `GroupFormat::deserialize` runs only when that format declares no binary
//! support. A present-key get of an inline value still does not deserialize the
//! node as objects.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use mapdb_rust_store::btree::BTreeMap;
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::long::LongFormat;
use mapdb_rust_store::ser::object_array::ObjectArrayFormat;
use mapdb_rust_store::ser::serializers::LongSer;
use mapdb_rust_store::ser::{GroupCursor, GroupFormat, SearchResult};
use mapdb_rust_store::store::StoreDirect;
use mapdb_rust_store::Result;

const KEYS: i64 = 16;
/// Large enough that 16 keys stay in the root leaf, so one get is one node.
const MAX_NODE: usize = 32;

#[test]
fn direct_binary_get_does_not_deserialize() {
    let store = Arc::new(StoreDirect::new_heap().unwrap());
    let deserialize = Arc::new(AtomicU64::new(0));
    let binary = Arc::new(AtomicU64::new(0));
    let keys = CountingFormat::new(LongFormat, Arc::clone(&deserialize), Arc::clone(&binary));
    let values = CountingFormat::new(LongFormat, Arc::clone(&deserialize), Arc::clone(&binary));
    let map = BTreeMap::create(store.clone(), keys, values, MAX_NODE).unwrap();
    put_keys(&map);
    // Heap StoreDirect writes the node on put. No separate commit, matching
    // the in-memory direct tests.
    assert_byte_side_get(&map, &deserialize, &binary);
}

#[test]
fn object_format_get_does_deserialize() {
    let store = Arc::new(StoreDirect::new_heap().unwrap());
    let deserialize = Arc::new(AtomicU64::new(0));
    let binary = Arc::new(AtomicU64::new(0));
    let keys = CountingFormat::new(
        ObjectArrayFormat::new(LongSer),
        Arc::clone(&deserialize),
        Arc::clone(&binary),
    );
    let map = BTreeMap::create(store, keys, LongFormat, MAX_NODE).unwrap();
    put_keys(&map);
    deserialize.store(0, Ordering::SeqCst);
    binary.store(0, Ordering::SeqCst);
    assert_eq!(map.get(&0).unwrap(), Some(100));
    assert_eq!(map.get(&-1).unwrap(), None);
    let d = deserialize.load(Ordering::SeqCst);
    assert!(
        d > 0,
        "objects-only key search must deserialize the key group; deserialize={d}"
    );
}

fn put_keys<KF, VF>(map: &BTreeMap<StoreDirect, KF, VF>)
where
    KF: GroupFormat<Elem = i64> + Send + Sync + 'static,
    VF: GroupFormat<Elem = i64> + Send + Sync + 'static,
{
    for k in 0..KEYS {
        assert_eq!(map.put(k, k + 100).unwrap(), None);
    }
}

fn assert_byte_side_get<KF, VF>(
    map: &BTreeMap<StoreDirect, KF, VF>,
    deserialize: &AtomicU64,
    binary: &AtomicU64,
) where
    KF: GroupFormat<Elem = i64> + Send + Sync + 'static,
    VF: GroupFormat<Elem = i64> + Send + Sync + 'static,
{
    deserialize.store(0, Ordering::SeqCst);
    binary.store(0, Ordering::SeqCst);
    assert_eq!(map.get(&0).unwrap(), Some(100));
    assert_eq!(map.get(&-1).unwrap(), None);
    assert_eq!(
        deserialize.load(Ordering::SeqCst),
        0,
        "binary key search must not deserialize the group"
    );
    let b = binary.load(Ordering::SeqCst);
    assert!(b > 0, "byte-side search must run; binary={b}");
}

/// Test double over a real [`GroupFormat`]. Counts `deserialize` against
/// `binary_search` and `binary_get`. No production counter.
struct CountingFormat<F> {
    inner: F,
    deserialize: Arc<AtomicU64>,
    binary: Arc<AtomicU64>,
}

impl<F> CountingFormat<F> {
    fn new(inner: F, deserialize: Arc<AtomicU64>, binary: Arc<AtomicU64>) -> Self {
        Self {
            inner,
            deserialize,
            binary,
        }
    }
}

impl<F> GroupFormat for CountingFormat<F>
where
    F: GroupFormat + Send + Sync + 'static,
{
    type Elem = F::Elem;
    type Group = F::Group;

    fn element(&self) -> &dyn mapdb_rust_store::ser::Serializer<Self::Elem> {
        self.inner.element()
    }
    fn empty(&self) -> Self::Group {
        self.inner.empty()
    }
    fn size(&self, g: &Self::Group) -> usize {
        self.inner.size(g)
    }
    fn get(&self, g: &Self::Group, pos: usize) -> Self::Elem {
        self.inner.get(g, pos)
    }
    fn search(&self, g: &Self::Group, key: &Self::Elem) -> SearchResult {
        self.inner.search(g, key)
    }
    fn compare(&self, a: &Self::Elem, b: &Self::Elem) -> std::cmp::Ordering {
        self.inner.compare(a, b)
    }
    fn natural_order(&self) -> bool {
        self.inner.natural_order()
    }
    fn insert(&self, g: &Self::Group, pos: usize, value: Self::Elem) -> Self::Group {
        self.inner.insert(g, pos, value)
    }
    fn set(&self, g: &Self::Group, pos: usize, value: Self::Elem) -> Self::Group {
        self.inner.set(g, pos, value)
    }
    fn delete(&self, g: &Self::Group, pos: usize) -> Self::Group {
        self.inner.delete(g, pos)
    }
    fn copy_range(&self, g: &Self::Group, from: usize, to: usize) -> Self::Group {
        self.inner.copy_range(g, from, to)
    }
    fn from_slice(&self, values: &[Self::Elem]) -> Self::Group {
        self.inner.from_slice(values)
    }
    fn serialize(&self, out: &mut DataOutput2, g: &Self::Group) {
        self.inner.serialize(out, g);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, count: usize) -> Result<Self::Group> {
        self.deserialize.fetch_add(1, Ordering::SeqCst);
        self.inner.deserialize(input, count)
    }
    fn supports_binary(&self) -> bool {
        self.inner.supports_binary()
    }
    fn binary_search(
        &self,
        key: &Self::Elem,
        input: &mut dyn DataInput2,
        count: usize,
    ) -> Result<SearchResult> {
        self.binary.fetch_add(1, Ordering::SeqCst);
        self.inner.binary_search(key, input, count)
    }
    fn binary_get(
        &self,
        input: &mut dyn DataInput2,
        count: usize,
        pos: usize,
    ) -> Result<Self::Elem> {
        self.binary.fetch_add(1, Ordering::SeqCst);
        self.inner.binary_get(input, count, pos)
    }
    fn supports_range_cursor(&self) -> bool {
        self.inner.supports_range_cursor()
    }
    fn range_cursor<'a>(
        &'a self,
        input: &'a mut dyn DataInput2,
        count: usize,
        from: usize,
        to: usize,
    ) -> Result<Box<dyn GroupCursor<Elem = Self::Elem> + 'a>> {
        self.inner.range_cursor(input, count, from, to)
    }
}
