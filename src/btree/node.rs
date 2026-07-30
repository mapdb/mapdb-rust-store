//! B-link tree node model + wire format (spec 03 §1), ported from Java
//! `BTreeMap.Node` / `NodeSerializer` byte-for-byte.
//!
//! Wire format (mapdb3 lineage): `packInt(keysLen<<4 | flags)`,
//! `[packLong(link)]` unless RIGHT, key group, then child recids (dir, packed
//! longs) or value group + optional 1-element fence group **last** (leaf). The
//! fence sits last so the read path never decodes it.

use crate::error::{DbError, Result};
use crate::io::{DataInput2, DataOutput2};
use crate::ser::long::LongFormat;
use crate::ser::{GroupFormat, Serializer};
use std::cmp::Ordering;
use std::marker::PhantomData;

/// Fixed-8-byte-BE group codec used for leaf value RECIDS in external-value mode
/// (Java `nodeValueFormat = LongFormat.INSTANCE`). Byte-compatible with Java's
/// `LongFormat.serialize(long[])`.
static NODE_RECID_FORMAT: LongFormat = LongFormat;

pub const DIR: i32 = 8;
pub const LEFT: i32 = 4;
pub const RIGHT: i32 = 2;

/// Per-element byte budget assumed for variable-width formats when sizing the
/// serialization buffer (Java `VAR_ELEM_EST`).
const VAR_ELEM_EST: usize = 32;

/// Immutable / copy-on-write node. Holds only the packed groups (never the
/// formats), so `Node<KF, VF>: Clone + Send + Sync + 'static` whenever the
/// group types are (they are, by the `GroupFormat` bounds).
pub struct Node<KF: GroupFormat, VF: GroupFormat> {
    pub flags: i32,
    /// Right-sibling recid; `0` ⇔ RIGHT flag (no right sibling).
    pub link: u64,
    pub keys: KF::Group,
    pub body: NodeBody<KF, VF>,
}

pub enum NodeBody<KF: GroupFormat, VF: GroupFormat> {
    Dir {
        children: Vec<u64>,
    },
    /// Inline leaf: `values` are the real values. `fence`: non-rightmost leaf
    /// only — a 1-element key group holding the leaf's inclusive high bound;
    /// `None` on rightmost leaves.
    Leaf {
        values: VF::Group,
        fence: Option<KF::Group>,
    },
    /// External-value leaf (`valueInline=false`): `recids` are
    /// the store recids of the value records (one per key), serialized as
    /// fixed-8-byte-BE longs (Java `LongFormat`). Same `fence` rule as `Leaf`.
    ExternalLeaf {
        recids: Vec<i64>,
        fence: Option<KF::Group>,
    },
}

// Manual Clone: `#[derive(Clone)]` would spuriously require `KF: Clone`.
impl<KF: GroupFormat, VF: GroupFormat> Clone for NodeBody<KF, VF> {
    fn clone(&self) -> Self {
        match self {
            NodeBody::Dir { children } => NodeBody::Dir {
                children: children.clone(),
            },
            NodeBody::Leaf { values, fence } => NodeBody::Leaf {
                values: values.clone(),
                fence: fence.clone(),
            },
            NodeBody::ExternalLeaf { recids, fence } => NodeBody::ExternalLeaf {
                recids: recids.clone(),
                fence: fence.clone(),
            },
        }
    }
}

impl<KF: GroupFormat, VF: GroupFormat> Clone for Node<KF, VF> {
    fn clone(&self) -> Self {
        Node {
            flags: self.flags,
            link: self.link,
            keys: self.keys.clone(),
            body: self.body.clone(),
        }
    }
}

impl<KF: GroupFormat, VF: GroupFormat> Node<KF, VF> {
    #[inline]
    pub fn is_dir(&self) -> bool {
        self.flags & DIR != 0
    }
    #[inline]
    pub fn is_right(&self) -> bool {
        self.flags & RIGHT != 0
    }

    /// Dir children (panics if called on a leaf — internal invariant).
    #[inline]
    pub fn children(&self) -> &[u64] {
        match &self.body {
            NodeBody::Dir { children } => children,
            _ => unreachable!("children() on a leaf node"),
        }
    }

    /// The leaf's inclusive high-bound fence (`None` for dirs or rightmost
    /// leaves), regardless of inline/external representation.
    #[inline]
    pub fn leaf_fence(&self) -> Option<&KF::Group> {
        match &self.body {
            NodeBody::Leaf { fence, .. } | NodeBody::ExternalLeaf { fence, .. } => fence.as_ref(),
            NodeBody::Dir { .. } => None,
        }
    }

    /// Debug-only structural invariants mirroring the Java `Node` ctor asserts.
    #[cfg(debug_assertions)]
    pub fn debug_check(&self) {
        debug_assert_eq!(
            self.is_right(),
            self.link == 0,
            "link/RIGHT mismatch (flags={}, link={})",
            self.flags,
            self.link
        );
        let has_fence = matches!(
            &self.body,
            NodeBody::Leaf { fence: Some(_), .. } | NodeBody::ExternalLeaf { fence: Some(_), .. }
        );
        debug_assert_eq!(
            has_fence,
            !self.is_dir() && !self.is_right(),
            "fence presence mismatch (flags={})",
            self.flags
        );
    }
}

/// `Serializer<Node>` bound to a pair of formats + a node-size hint. Cheap to
/// build per call (borrows the formats); used only by the btree, so `compare`
/// and `equals` are never reached.
pub struct NodeSerializer<'a, KF: GroupFormat, VF: GroupFormat> {
    pub kf: &'a KF,
    pub vf: &'a VF,
    /// `false` ⇔ external-value map: leaf value slots hold packed value recids
    /// (`LongFormat`) instead of `vf`-encoded values.
    pub value_inline: bool,
    pub size_hint: usize,
    _p: PhantomData<(KF, VF)>,
}

impl<'a, KF: GroupFormat, VF: GroupFormat> NodeSerializer<'a, KF, VF> {
    /// Inline-value serializer (the common case). Kept for the many call sites
    /// that never touch external values.
    pub fn new(kf: &'a KF, vf: &'a VF, max_node_size: usize) -> Self {
        Self::new_mode(kf, vf, max_node_size, true)
    }

    pub fn new_mode(kf: &'a KF, vf: &'a VF, max_node_size: usize, value_inline: bool) -> Self {
        // header (packInt + packLong <= ~11 B, round to 16) + a full node's
        // key/value bytes + 2 spare key slots (pre-split transient + fence).
        let ke = kf.element().fixed_size();
        // External leaves store 8-byte recids (Java LongFormat) regardless of `vf`.
        let ve = if value_inline {
            vf.element().fixed_size()
        } else {
            NODE_RECID_FORMAT.element().fixed_size()
        };
        let key_bytes = ke.unwrap_or(VAR_ELEM_EST);
        // leaf: one value per key; dir: one packed-long child per key (<=9 B).
        let val_bytes = match ve {
            Some(n) => n.max(9),
            None => VAR_ELEM_EST,
        };
        let size_hint = 16 + (max_node_size + 1) * (key_bytes + val_bytes) + 2 * key_bytes + 8;
        NodeSerializer {
            kf,
            vf,
            value_inline,
            size_hint,
            _p: PhantomData,
        }
    }

    /// Validate that a decoded key group is strictly increasing under the key
    /// format's ordering. Used on the full-decode path (writers, iteration,
    /// open); rejects crafted unsorted/duplicate-key nodes.
    fn check_sorted(&self, keys: &KF::Group) -> Result<()> {
        let n = self.kf.size(keys);
        for i in 1..n {
            let a = self.kf.get(keys, i - 1);
            let b = self.kf.get(keys, i);
            if self.kf.compare(&a, &b) != Ordering::Less {
                return Err(DbError::corrupt("node keys not strictly increasing"));
            }
        }
        Ok(())
    }
}

impl<'a, KF: GroupFormat, VF: GroupFormat> Serializer<Node<KF, VF>> for NodeSerializer<'a, KF, VF> {
    fn serialize(&self, out: &mut DataOutput2, n: &Node<KF, VF>) {
        let keys_len = self.kf.size(&n.keys);
        out.pack_int(((keys_len as i32) << 4) | n.flags);
        if !n.is_right() {
            out.pack_long(n.link);
        }
        self.kf.serialize(out, &n.keys);
        match &n.body {
            NodeBody::Dir { children } => {
                for &child in children {
                    out.pack_long(child);
                }
            }
            NodeBody::Leaf { values, fence } => {
                self.vf.serialize(out, values);
                if !n.is_right() {
                    // fence present on every non-rightmost leaf (invariant).
                    self.kf.serialize(
                        out,
                        fence.as_ref().expect("non-rightmost leaf without fence"),
                    );
                }
            }
            NodeBody::ExternalLeaf { recids, fence } => {
                // value recids as fixed-8-byte-BE longs (Java LongFormat).
                NODE_RECID_FORMAT.serialize(out, recids);
                if !n.is_right() {
                    self.kf.serialize(
                        out,
                        fence.as_ref().expect("non-rightmost leaf without fence"),
                    );
                }
            }
        }
    }

    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<Node<KF, VF>> {
        // Record start so we can require exact record consumption below: a valid
        // node serializes to exactly [header][link?][keys][children|values+fence]
        // with no slack (the store passes the exact content length, checksum
        // excluded). A crafted, checksum-valid header that under-reports keysLen
        // shifts the key/value boundary while leaving the record locally
        // decodable — the leftover trailing bytes are the tell (D5).
        let start = input.pos();
        let h = input.unpack_int()?;
        let flags = h & 0xF;
        let keys_len = ((h as u32) >> 4) as usize;
        // Every key occupies >= 1 serialized byte, so keysLen > size is corrupt.
        if let Some(sz) = size {
            if keys_len > sz {
                return Err(DbError::corrupt("node header keysLen exceeds record size"));
            }
        }
        // Structural validation of persisted recid invariants (D5): a crafted,
        // checksum-valid record must never yield a panic (via `nz(0)` or an
        // out-of-bounds child index) or a silent false-negative (a spurious 0
        // "no link" sentinel). A non-rightmost node MUST carry a right link.
        let link = if flags & RIGHT != 0 {
            0
        } else {
            let l = input.unpack_long()?;
            if l == 0 {
                return Err(DbError::corrupt("non-rightmost node with zero link"));
            }
            l
        };
        let keys = self.kf.deserialize(input, keys_len)?;
        // Search, routing, and fence math all assume strictly-increasing keys.
        // A crafted node with unsorted/duplicate keys would otherwise cause a
        // silent false-negative (binary search skips a present key) or a lost
        // update (a writer routes a replace into the wrong leaf) — validate here
        // so every writer/iterator/open path rejects it (D5).
        self.check_sorted(&keys)?;
        let body = if flags & DIR != 0 {
            let child_count = keys_len + if flags & RIGHT != 0 { 1 } else { 0 };
            // A non-rightmost dir has childCount == keysLen; a keyless one (0
            // children) can never route or be indexed — reject it.
            if child_count == 0 {
                return Err(DbError::corrupt("directory node with no children"));
            }
            let mut children = Vec::new();
            children.try_reserve(child_count)?;
            for _ in 0..child_count {
                let c = input.unpack_long()?;
                if c == 0 {
                    return Err(DbError::corrupt("directory child recid is zero"));
                }
                children.push(c);
            }
            NodeBody::Dir { children }
        } else {
            // Read the value group (inline values or external recids) then the
            // optional fence. Both representations share the fence bound check.
            let recids = if self.value_inline {
                None
            } else {
                Some(NODE_RECID_FORMAT.deserialize(input, keys_len)?)
            };
            let values = if self.value_inline {
                Some(self.vf.deserialize(input, keys_len)?)
            } else {
                None
            };
            // fence present iff (!dir && !right) — enforced by the read shape.
            let fence = if flags & RIGHT == 0 {
                let f = self.kf.deserialize(input, 1)?;
                // The fence is the leaf's inclusive high bound, so it must be
                // >= the leaf's greatest live key. A crafted fence below the
                // last key makes a writer treat an in-leaf key as "beyond the
                // leaf" and move right, losing an atomic replace (D5).
                if keys_len > 0 {
                    let last = self.kf.get(&keys, keys_len - 1);
                    let bound = self.kf.get(&f, 0);
                    if self.kf.compare(&last, &bound) == Ordering::Greater {
                        return Err(DbError::corrupt("leaf fence below greatest key"));
                    }
                }
                Some(f)
            } else {
                None
            };
            if let Some(recids) = recids {
                NodeBody::ExternalLeaf { recids, fence }
            } else {
                NodeBody::Leaf {
                    values: values.expect("inline leaf without values"),
                    fence,
                }
            }
        };
        // Require exact consumption: a well-formed node uses every content byte.
        if let Some(sz) = size {
            let consumed = input.pos() - start;
            if consumed != sz {
                return Err(DbError::corrupt(
                    "node record has unexpected trailing/short bytes (forged keysLen)",
                ));
            }
        }
        let node = Node {
            flags,
            link,
            keys,
            body,
        };
        #[cfg(debug_assertions)]
        node.debug_check();
        Ok(node)
    }

    fn size_hint(&self) -> usize {
        self.size_hint
    }

    fn compare(&self, _a: &Node<KF, VF>, _b: &Node<KF, VF>) -> Ordering {
        unreachable!("btree nodes are never compared")
    }

    fn equals(&self, _a: &Node<KF, VF>, _b: &Node<KF, VF>) -> bool {
        unreachable!("btree nodes are never compared for equality")
    }
}
