//! In-crate btree tests that need the crate-internal `Node`/`NodeSerializer`
//! (golden wire-format vectors + crafted structural/store corruption). The
//! public-API behavioural suites live under `tests/btree_*.rs`.

use std::sync::Arc;

use super::node::{Node, NodeBody, NodeSerializer, DIR, LEFT, RIGHT};
use super::BTreeMap;
use crate::io::{DataInput2, DataOutput2, SliceInput};
use crate::ser::long::LongFormat;
use crate::ser::serializers::LONG;
use crate::ser::Serializer;
use crate::store::{Recid, Store, StoreOnHeap};

type LNode = Node<LongFormat, LongFormat>;

fn ser_bytes(node: &LNode) -> Vec<u8> {
    let ns = NodeSerializer::new(&LongFormat, &LongFormat, 16);
    let mut out = DataOutput2::new();
    ns.serialize(&mut out, node);
    out.into_vec()
}

fn be(v: i64) -> [u8; 8] {
    v.to_be_bytes()
}

// ================= golden wire-format vectors (spec 03 §8) =================
// Packed header/link/children bytes hand-computed from the MapDB packLong
// encoding (continuation bit on the LAST byte); fixed-width elements use
// big-endian i64. These pin the format so a change breaking cross-open /
// Java differential parity fails loudly.

#[test]
fn golden_empty_rightmost_leaf() {
    // flags LEFT|RIGHT (6), empty → header packInt(6)=0x86, nothing else.
    let node: LNode = Node {
        flags: LEFT | RIGHT,
        link: 0,
        keys: vec![],
        body: NodeBody::Leaf {
            values: vec![],
            fence: None,
        },
    };
    assert_eq!(ser_bytes(&node), vec![0x86]);
}

#[test]
fn golden_non_rightmost_leaf() {
    // flags LEFT (4), link 9, keys=[5,7], values=[50,70], fence=[7].
    // header packInt(36)=0xA4, link packLong(9)=0x89, then be groups, fence last.
    let node: LNode = Node {
        flags: LEFT,
        link: 9,
        keys: vec![5, 7],
        body: NodeBody::Leaf {
            values: vec![50, 70],
            fence: Some(vec![7]),
        },
    };
    let mut want = vec![0xA4u8, 0x89];
    want.extend_from_slice(&be(5));
    want.extend_from_slice(&be(7));
    want.extend_from_slice(&be(50));
    want.extend_from_slice(&be(70));
    want.extend_from_slice(&be(7));
    assert_eq!(ser_bytes(&node), want);
}

#[test]
fn golden_rightmost_dir() {
    // flags DIR|LEFT|RIGHT (14), keys=[5], children=[100,200] (childCount keysLen+1).
    // header packInt(30)=0x9E, no link, be(5), packLong(100)=0xE4, packLong(200)=0x01,0xC8.
    let node: LNode = Node {
        flags: DIR | LEFT | RIGHT,
        link: 0,
        keys: vec![5],
        body: NodeBody::Dir {
            children: vec![100, 200],
        },
    };
    let mut want = vec![0x9Eu8];
    want.extend_from_slice(&be(5));
    want.extend_from_slice(&[0xE4, 0x01, 0xC8]);
    assert_eq!(ser_bytes(&node), want);
}

#[test]
fn golden_non_rightmost_dir() {
    // flags DIR (8), link 42, keys=[3,9], children=[7,8] (childCount == keysLen).
    // header packInt(40)=0xA8, link packLong(42)=0xAA, be groups, packLong(7)=0x87, packLong(8)=0x88.
    let node: LNode = Node {
        flags: DIR,
        link: 42,
        keys: vec![3, 9],
        body: NodeBody::Dir {
            children: vec![7, 8],
        },
    };
    let mut want = vec![0xA8u8, 0xAA];
    want.extend_from_slice(&be(3));
    want.extend_from_slice(&be(9));
    want.extend_from_slice(&[0x87, 0x88]);
    assert_eq!(ser_bytes(&node), want);
}

#[test]
fn roundtrip_determinism() {
    let ns = NodeSerializer::new(&LongFormat, &LongFormat, 16);
    let nodes: Vec<LNode> = vec![
        Node {
            flags: LEFT | RIGHT,
            link: 0,
            keys: vec![],
            body: NodeBody::Leaf {
                values: vec![],
                fence: None,
            },
        },
        Node {
            flags: LEFT,
            link: 9,
            keys: vec![5, 7, 11],
            body: NodeBody::Leaf {
                values: vec![50, 70, 110],
                fence: Some(vec![11]),
            },
        },
        Node {
            flags: DIR | LEFT | RIGHT,
            link: 0,
            keys: vec![5, 9],
            body: NodeBody::Dir {
                children: vec![100, 200, 300],
            },
        },
        Node {
            flags: 0,
            link: 77,
            keys: vec![100],
            body: NodeBody::Leaf {
                values: vec![1000],
                fence: Some(vec![100]),
            },
        },
    ];
    for node in &nodes {
        let bytes = ser_bytes(node);
        let mut input = SliceInput::new(&bytes);
        let decoded = ns.deserialize(&mut input, Some(bytes.len())).unwrap();
        assert_eq!(ser_bytes(&decoded), bytes, "roundtrip flags={}", node.flags);
        assert_eq!(
            input.pos(),
            bytes.len(),
            "trailing bytes flags={}",
            node.flags
        );
    }
}

// ================= golden vectors: external values + size counter =================

fn ser_bytes_external(node: &LNode) -> Vec<u8> {
    // value_inline = false: leaf value slots hold value recids as LongFormat.
    let ns = NodeSerializer::new_mode(&LongFormat, &LongFormat, 16, false);
    let mut out = DataOutput2::new();
    ns.serialize(&mut out, node);
    out.into_vec()
}

#[test]
fn golden_external_leaf_encodes_recids_as_be_longs() {
    // External-value leaf: the value slots hold value RECIDS,
    // serialized as fixed-8-byte-BE longs (Java `nodeValueFormat = LongFormat`),
    // byte-IDENTICAL to an inline Long leaf. This pins external-map byte parity.
    let ext: LNode = Node {
        flags: LEFT,
        link: 9,
        keys: vec![5, 7],
        body: NodeBody::ExternalLeaf {
            recids: vec![50, 70],
            fence: Some(vec![7]),
        },
    };
    let mut want = vec![0xA4u8, 0x89];
    want.extend_from_slice(&be(5));
    want.extend_from_slice(&be(7));
    want.extend_from_slice(&be(50)); // value recid 50
    want.extend_from_slice(&be(70)); // value recid 70
    want.extend_from_slice(&be(7)); // fence last
    assert_eq!(ser_bytes_external(&ext), want);

    // Round-trips back to an ExternalLeaf under the external serializer.
    let ns = NodeSerializer::new_mode(&LongFormat, &LongFormat, 16, false);
    let mut input = SliceInput::new(&want);
    let back = ns.deserialize(&mut input, Some(want.len())).unwrap();
    match back.body {
        NodeBody::ExternalLeaf { recids, fence } => {
            assert_eq!(recids, vec![50, 70]);
            assert_eq!(fence, Some(vec![7]));
        }
        _ => panic!("expected ExternalLeaf"),
    }
    assert_eq!(input.pos(), want.len());
}

#[test]
fn golden_external_rightmost_leaf_no_fence() {
    let ext: LNode = Node {
        flags: LEFT | RIGHT,
        link: 0,
        keys: vec![3],
        body: NodeBody::ExternalLeaf {
            recids: vec![99],
            fence: None,
        },
    };
    // header packInt((1<<4)|6)=packInt(22)=0x96 (high bit on terminal byte),
    // no link, be(3), be(99), no fence.
    let mut want = vec![0x96u8];
    want.extend_from_slice(&be(3));
    want.extend_from_slice(&be(99));
    assert_eq!(ser_bytes_external(&ext), want);
}

#[test]
fn golden_counter_record_is_be_i64() {
    // The Feature-A size counter is a single `Long` record; the store persists it
    // via `Serializers.LONG` = fixed 8-byte BE. Verify a live counter map's record
    // holds the count in that exact encoding (byte-parity with Java).
    let store = Arc::new(StoreOnHeap::new(true));
    let map =
        BTreeMap::create_with_counter(store.clone(), LongFormat, LongFormat, 8, true).unwrap();
    for i in 0..5i64 {
        map.put(i, i).unwrap();
    }
    let cr = map.counter_recid();
    assert!(cr > 0);
    // stored value equals the count, and the LONG codec is fixed 8-byte BE.
    let v = store.get(Recid::new(cr).unwrap(), &LONG).unwrap().unwrap();
    assert_eq!(v, 5);
    let mut out = DataOutput2::new();
    LONG.serialize(&mut out, &5i64);
    assert_eq!(out.into_vec(), 5i64.to_be_bytes().to_vec());
    assert_eq!(map.size_long().unwrap(), 5);
}

// ================= deserialize-level structural corruption =================

fn deser(bytes: &[u8]) -> crate::Result<LNode> {
    let ns = NodeSerializer::new(&LongFormat, &LongFormat, 16);
    let mut input = SliceInput::new(bytes);
    ns.deserialize(&mut input, Some(bytes.len()))
}

fn is_corrupt<T>(r: crate::Result<T>) -> bool {
    matches!(r, Err(crate::DbError::DataCorruption(_)))
}

#[test]
fn corrupt_keyslen_exceeds_size() {
    let mut out = DataOutput2::new();
    out.pack_int((200 << 4) | (LEFT | RIGHT)); // claims 200 keys in 1 byte
    assert!(is_corrupt(deser(&out.into_vec())));
}

#[test]
fn corrupt_non_rightmost_node_zero_link() {
    let mut out = DataOutput2::new();
    out.pack_int((1 << 4) | LEFT);
    out.pack_long(0); // non-right node must carry a link
    assert!(is_corrupt(deser(&out.into_vec())));
}

#[test]
fn corrupt_directory_zero_child() {
    let mut out = DataOutput2::new();
    out.pack_int((1 << 4) | (DIR | LEFT | RIGHT));
    out.write_i64(5);
    out.pack_long(100);
    out.pack_long(0); // zero child recid
    assert!(is_corrupt(deser(&out.into_vec())));
}

#[test]
fn corrupt_empty_non_rightmost_dir() {
    let mut out = DataOutput2::new();
    out.pack_int(DIR); // non-right, 0 keys → 0 children
    out.pack_long(5);
    assert!(is_corrupt(deser(&out.into_vec())));
}

#[test]
fn corrupt_forged_keyslen_leaves_trailing_bytes() {
    // Valid rightmost leaf with 2 keys/values; forge the header keysLen 2→1.
    // The remaining bytes are still key1|key2|val1|val2, so the decoder would
    // read key1, then val1=key2, and ignore val1|val2 — returning the wrong
    // value. The leftover trailing bytes must trip exact-consumption.
    let node = Node {
        flags: LEFT | RIGHT,
        link: 0,
        keys: vec![1, 2],
        body: NodeBody::Leaf {
            values: vec![10, 20],
            fence: None,
        },
    };
    let mut bytes = ser_bytes(&node);
    // packLong sets the high bit on the terminal byte: header (2<<4)|6 = 38
    // encodes as 0xA6; forge keysLen 2→1 so it decodes as (1<<4)|6 = 22 = 0x96.
    assert_eq!(bytes[0], 0xA6, "unexpected header encoding");
    bytes[0] = 0x96;
    assert!(is_corrupt(deser(&bytes)));
}

#[test]
fn corrupt_unsorted_keys_rejected() {
    // keys descending [2,1] — search/routing assume strictly increasing.
    let node = Node {
        flags: LEFT | RIGHT,
        link: 0,
        keys: vec![2, 1],
        body: NodeBody::Leaf {
            values: vec![20, 10],
            fence: None,
        },
    };
    assert!(is_corrupt(deser(&ser_bytes(&node))));
}

#[test]
fn corrupt_duplicate_keys_rejected() {
    let node = Node {
        flags: LEFT | RIGHT,
        link: 0,
        keys: vec![5, 5],
        body: NodeBody::Leaf {
            values: vec![50, 51],
            fence: None,
        },
    };
    assert!(is_corrupt(deser(&ser_bytes(&node))));
}

#[test]
fn corrupt_leaf_fence_below_last_key() {
    // Non-rightmost leaf keys [1,2] with fence [1] < greatest key 2: a writer
    // replacing key 2 would treat it as beyond the leaf and move right, losing
    // the atomic replace.
    let node = Node {
        flags: LEFT, // non-right leaf
        link: 77,
        keys: vec![1, 2],
        body: NodeBody::Leaf {
            values: vec![10, 20],
            fence: Some(vec![1]),
        },
    };
    assert!(is_corrupt(deser(&ser_bytes(&node))));
}

// ================= store-level corruption =================

fn write(store: &StoreOnHeap, recid: Recid, node: &LNode) {
    let ns = NodeSerializer::new(&LongFormat, &LongFormat, 16);
    store.update(recid, Some(node), &ns).unwrap();
}

#[test]
fn zero_root_pointer_open_errors() {
    let store = Arc::new(StoreOnHeap::new(true));
    let rrr = store.put(&0i64, &LONG).unwrap();
    assert!(is_corrupt(BTreeMap::open(
        store,
        rrr.get(),
        LongFormat,
        LongFormat,
        8
    )));
}

#[test]
fn zero_root_recid_recid_open_errors() {
    // A caller passing root_recid_recid = 0 must be a clean error, not a nz() panic.
    let store = Arc::new(StoreOnHeap::new(true));
    assert!(BTreeMap::open(store, 0, LongFormat, LongFormat, 8).is_err());
}

#[test]
fn self_cyclic_root_dir_open_errors() {
    let store = Arc::new(StoreOnHeap::new(true));
    let d = store.preallocate().unwrap();
    let dir: LNode = Node {
        flags: DIR | LEFT | RIGHT,
        link: 0,
        keys: vec![5],
        body: NodeBody::Dir {
            children: vec![d.get(), d.get()],
        },
    };
    write(&store, d, &dir);
    let rrr = store.put(&(d.get() as i64), &LONG).unwrap();
    assert!(is_corrupt(BTreeMap::open(
        store,
        rrr.get(),
        LongFormat,
        LongFormat,
        8
    )));
}

/// Valid-looking two-leaf tree whose rightmost leaf links BACK to the left leaf
/// (a leaf-link cycle `open`'s leftmost-spine walk does not touch).
fn map_with_leaf_cycle() -> BTreeMap<StoreOnHeap, LongFormat, LongFormat> {
    let store = Arc::new(StoreOnHeap::new(true));
    let leaf_a = store.preallocate().unwrap();
    let leaf_b = store.preallocate().unwrap();
    let root = store.preallocate().unwrap();
    let a: LNode = Node {
        flags: LEFT,
        link: leaf_b.get(),
        keys: vec![1],
        body: NodeBody::Leaf {
            values: vec![10],
            fence: Some(vec![100]),
        },
    };
    let b: LNode = Node {
        flags: 0,
        link: leaf_a.get(), // cycle
        keys: vec![200],
        body: NodeBody::Leaf {
            values: vec![20],
            fence: Some(vec![200]),
        },
    };
    let r: LNode = Node {
        flags: DIR | LEFT | RIGHT,
        link: 0,
        keys: vec![100],
        body: NodeBody::Dir {
            children: vec![leaf_a.get(), leaf_b.get()],
        },
    };
    write(&store, leaf_a, &a);
    write(&store, leaf_b, &b);
    write(&store, root, &r);
    let rrr = store.put(&(root.get() as i64), &LONG).unwrap();
    BTreeMap::open(store, rrr.get(), LongFormat, LongFormat, 8).unwrap()
}

#[test]
fn get_through_leaf_cycle_errors_not_hangs() {
    let map = map_with_leaf_cycle();
    assert!(is_corrupt(map.get(&300))); // routed to leafB → move-right leafB→leafA→…
}

#[test]
fn iteration_through_leaf_cycle_errors_not_hangs() {
    let map = map_with_leaf_cycle();
    let mut saw_err = false;
    let mut count = 0u64;
    for e in map.iter().unwrap() {
        if e.is_err() {
            saw_err = true;
            break;
        }
        count += 1;
        assert!(count < 5_000_000, "iterator did not terminate");
    }
    assert!(saw_err, "leaf-link cycle must surface as an iterator error");
}

// ================= partial-failure poison =================

/// A store that delegates to `StoreOnHeap` but fails every `update` to a chosen
/// recid — used to fail the root-pointer update during a root-grow.
struct FailUpdateStore {
    inner: StoreOnHeap,
    fail_recid: std::sync::atomic::AtomicU64,
    lease_table: Arc<crate::store::LeaseTable>,
}

impl FailUpdateStore {
    fn new() -> Self {
        FailUpdateStore {
            inner: StoreOnHeap::new(true),
            fail_recid: std::sync::atomic::AtomicU64::new(0),
            lease_table: crate::store::LeaseTable::new(),
        }
    }
    fn arm(&self, recid: u64) {
        self.fail_recid
            .store(recid, std::sync::atomic::Ordering::SeqCst);
    }
    fn armed_for(&self, recid: Recid) -> bool {
        let f = self.fail_recid.load(std::sync::atomic::Ordering::SeqCst);
        f != 0 && f == recid.get()
    }
}

impl crate::store::StoreLease for FailUpdateStore {
    fn lease_table(&self) -> &Arc<crate::store::LeaseTable> {
        &self.lease_table
    }
}

impl Store for FailUpdateStore {
    fn preallocate(&self) -> crate::Result<Recid> {
        self.inner.preallocate()
    }
    fn put<R: crate::store::Record>(
        &self,
        value: &R,
        ser: &(impl Serializer<R> + Sync),
    ) -> crate::Result<Recid> {
        self.inner.put(value, ser)
    }
    fn get<R: crate::store::Record>(
        &self,
        recid: Recid,
        ser: &(impl Serializer<R> + Sync),
    ) -> crate::Result<Option<R>> {
        self.inner.get(recid, ser)
    }
    fn read(&self, recid: Recid, action: &mut dyn crate::store::RecordRead) -> crate::Result<i64> {
        self.inner.read(recid, action)
    }
    fn update<R: crate::store::Record>(
        &self,
        recid: Recid,
        value: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> crate::Result<()> {
        if self.armed_for(recid) {
            return Err(crate::DbError::corrupt("injected update failure"));
        }
        self.inner.update(recid, value, ser)
    }
    fn compare_and_swap<R: crate::store::Record>(
        &self,
        recid: Recid,
        expect: Option<&R>,
        new: Option<&R>,
        ser: &(impl Serializer<R> + Sync),
    ) -> crate::Result<bool> {
        self.inner.compare_and_swap(recid, expect, new, ser)
    }
    fn delete(&self, recid: Recid) -> crate::Result<()> {
        self.inner.delete(recid)
    }
    fn commit(&self) -> crate::Result<()> {
        self.inner.commit()
    }
    fn close(&self) -> crate::Result<()> {
        self.inner.close()
    }
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn verify(&self) -> crate::Result<()> {
        self.inner.verify()
    }
    fn get_all_recids(&self) -> crate::Result<Vec<Recid>> {
        self.inner.get_all_recids()
    }
}

#[test]
fn root_grow_failure_poisons_not_hangs() {
    let store = Arc::new(FailUpdateStore::new());
    let map = BTreeMap::create(store.clone(), LongFormat, LongFormat, 4).unwrap();
    // The root-pointer record is updated ONLY on a root grow → fail that update.
    store.arm(map.root_recid_recid());
    // Insert enough to overflow the single-leaf root (maxNodeSize 4) → root grow,
    // whose root-pointer update fails → propagation error.
    let mut hit_err = false;
    for i in 0..10i64 {
        if map.put(i, i).is_err() {
            hit_err = true;
            break;
        }
    }
    assert!(
        hit_err,
        "the root-grow update failure must surface as an error"
    );
    // The map is now poisoned: later ops must fail FAST (not park forever in
    // left_edge waiting for a level that will never be published).
    assert!(
        map.put(1000, 1000).is_err(),
        "poisoned map must reject writes"
    );
    assert!(map.get(&0).is_err(), "poisoned map must reject reads");

    // Round-3 residual: the poison is in-memory. REOPENING the same store after a
    // failed root-grow must not hang a later writer — the root pointer names a
    // now-LEFT-only node, which open detects as corruption rather than trusting.
    let rrr = map.root_recid_recid();
    drop(map);
    store
        .fail_recid
        .store(0, std::sync::atomic::Ordering::SeqCst); // disarm; the damage is already persisted
    let reopened = BTreeMap::open(store, rrr, LongFormat, LongFormat, 4);
    assert!(
        is_corrupt(reopened),
        "reopen after a failed root-grow must report corruption, not hang"
    );
}

#[test]
fn byte_path_rejects_empty_non_right_dir() {
    // A crafted checksum-valid non-right dir with 0 keys (→ 0 children) on a
    // routed path must be DataCorruption from get(), not a silent traversal
    // (GetAction::on_bytes must match deserialize's structural check).
    // Build it on StoreDirect so reads take the on_bytes push-down path.
    use crate::store::StoreDirect;
    let store = Arc::new(StoreDirect::new_heap().unwrap());
    let good_leaf = store.preallocate().unwrap();
    let bad_dir = store.preallocate().unwrap();
    let root = store.preallocate().unwrap();

    // good leftmost leaf (rightmost-shaped so open's spine walk is clean)... but
    // it must be non-right to carry a link to bad_dir. Give it a fence.
    let leaf: LNode = Node {
        flags: LEFT,
        link: bad_dir.get(),
        keys: vec![1],
        body: NodeBody::Leaf {
            values: vec![10],
            fence: Some(vec![50]),
        },
    };
    // crafted empty non-right dir: 0 keys → 0 children, nonzero link.
    let bad: LNode = Node {
        flags: DIR,
        link: good_leaf.get(),
        keys: vec![],
        body: NodeBody::Dir { children: vec![] },
    };
    // root dir routes keys > 50 to the bad dir (child index 1).
    let r: LNode = Node {
        flags: DIR | LEFT | RIGHT,
        link: 0,
        keys: vec![50],
        body: NodeBody::Dir {
            children: vec![good_leaf.get(), bad_dir.get()],
        },
    };
    write_direct(&store, good_leaf, &leaf);
    write_direct(&store, bad_dir, &bad);
    write_direct(&store, root, &r);
    let rrr = store.put(&(root.get() as i64), &LONG).unwrap();
    let map = BTreeMap::open(store, rrr.get(), LongFormat, LongFormat, 8).unwrap();
    // routing 100 → child[1] = bad dir → on_bytes sees child_count 0.
    assert!(is_corrupt(map.get(&100)));
}

fn write_direct(store: &crate::store::StoreDirect, recid: Recid, node: &LNode) {
    let ns = NodeSerializer::new(&LongFormat, &LongFormat, 16);
    store.update(recid, Some(node), &ns).unwrap();
}

#[test]
fn crafted_fake_root_descendant_does_not_replace_root() {
    // Round-4: a checksum-valid descendant leaf falsely flagged LEFT|RIGHT must
    // NOT be treated as the root when it later splits — doing so would grow a new
    // root from its halves alone and orphan the real root + siblings (data loss).
    // Root growth is now gated on authoritative root identity, not node flags.
    use crate::store::StoreDirect;
    let store = Arc::new(StoreDirect::new_heap().unwrap());
    let a = store.preallocate().unwrap();
    let c = store.preallocate().unwrap();
    let root = store.preallocate().unwrap();

    // A: crafted LEFT|RIGHT leaf (a "fake root"), full at maxNodeSize=4.
    let node_a: LNode = Node {
        flags: LEFT | RIGHT,
        link: 0,
        keys: vec![10, 20, 30, 40],
        body: NodeBody::Leaf {
            values: vec![100, 200, 300, 400],
            fence: None,
        },
    };
    // C: real rightmost sibling holding data that must NOT be lost.
    let node_c: LNode = Node {
        flags: RIGHT,
        link: 0,
        keys: vec![100],
        body: NodeBody::Leaf {
            values: vec![1000],
            fence: None,
        },
    };
    // R: the genuine root, routes <=50 to A and >50 to C.
    let node_r: LNode = Node {
        flags: DIR | LEFT | RIGHT,
        link: 0,
        keys: vec![50],
        body: NodeBody::Dir {
            children: vec![a.get(), c.get()],
        },
    };
    write_direct(&store, a, &node_a);
    write_direct(&store, c, &node_c);
    write_direct(&store, root, &node_r);
    let rrr = store.put(&(root.get() as i64), &LONG).unwrap();
    let map = BTreeMap::open(store, rrr.get(), LongFormat, LongFormat, 4).unwrap();

    // Inserting 45 (routed to A) overflows A → split. With the flags-based bug
    // this would replace the root and lose C; authoritative identity prevents it.
    map.put(45, 450).unwrap();

    // The real root R and sibling C must still be reachable (via dir routing):
    assert_eq!(
        map.get(&100).unwrap(),
        Some(1000),
        "sibling C must not be orphaned"
    );
    assert_eq!(map.get(&10).unwrap(), Some(100));
    assert_eq!(map.get(&40).unwrap(), Some(400));
    assert_eq!(map.get(&45).unwrap(), Some(450));
    // root pointer unchanged (no bogus root replacement)
    assert_eq!(map.root_recid_recid(), rrr.get());
}

#[test]
fn get_routed_to_null_child_errors_not_absent() {
    // A crafted dir whose off-spine child recid resolves to a null/preallocated
    // record must make get() raise DataCorruption via GetAction::on_null — NOT
    // silently return `None` (a present-by-write key invisible to reads). The
    // left-spine child is a valid leaf so open()/build_left_edges succeeds and
    // the null is reached only by descent by descent.
    let store = Arc::new(StoreOnHeap::new(true));
    let child0 = store.preallocate().unwrap();
    let null_child = store.preallocate().unwrap(); // never written → stays null
    let root = store.preallocate().unwrap();
    write(
        &store,
        child0,
        &Node {
            flags: LEFT | RIGHT,
            link: 0,
            keys: vec![10],
            body: NodeBody::Leaf {
                values: vec![100],
                fence: None,
            },
        },
    );
    write(
        &store,
        root,
        &Node {
            flags: DIR | LEFT | RIGHT,
            link: 0,
            keys: vec![50],
            body: NodeBody::Dir {
                children: vec![child0.get(), null_child.get()],
            },
        },
    );
    let rrr = store.put(&(root.get() as i64), &LONG).unwrap();
    let map = BTreeMap::open(store, rrr.get(), LongFormat, LongFormat, 16).unwrap();
    assert_eq!(map.get(&10).unwrap(), Some(100)); // spine child still works
    assert!(is_corrupt(map.get(&100))); // routes to null child → corrupt, not None
}
