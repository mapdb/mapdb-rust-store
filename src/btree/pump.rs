//! `TreePump` — bottom-up bulk builder for the B-link trees (spec 03 §4),
//! ported from Java `TreePump`. Feed strictly ascending entries via [`put`],
//! then [`finish`] once; every node is written EXACTLY once via
//! preallocate-next-sibling-then-write (forward links, no back-patching), so on
//! a fresh store recids and data lay out sequentially in key order.

use crate::error::{DbError, Result};
use crate::store::Store;
use std::cmp::Ordering;

use super::node::{DIR, LEFT, RIGHT};

/// Sink that materializes + writes one finished node. Keeps the pump itself
/// serialization-free; [`super::map::BTreeMap`] (and later `BufferTreeMap`)
/// provide the concrete sink.
pub trait NodeSink {
    type Key: Clone;
    type Val: Clone;

    /// Key order (== `keyFormat.compare`); used for the strictly-ascending check.
    fn compare_keys(&self, a: &Self::Key, b: &Self::Key) -> Ordering;

    fn write_leaf(
        &self,
        recid: u64,
        flags: i32,
        link: u64,
        keys: Vec<Self::Key>,
        values: Vec<Self::Val>,
    ) -> Result<()>;

    fn write_dir(
        &self,
        recid: u64,
        flags: i32,
        link: u64,
        keys: Vec<Self::Key>,
        children: Vec<u64>,
    ) -> Result<()>;
}

/// One in-progress node per tree level; index 0 = leaf (values only there).
struct Level<Kk, Vv> {
    keys: Vec<Kk>,
    values: Vec<Vv>,    // leaf level only
    children: Vec<u64>, // dir levels only
    /// Preallocated recid of this level's NEXT node; 0 = none yet.
    pending: u64,
    /// No node flushed at this level yet (LEFT candidate).
    first: bool,
}

impl<Kk, Vv> Level<Kk, Vv> {
    fn new(_leaf: bool) -> Self {
        Level {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            pending: 0,
            first: true,
        }
    }
}

pub struct TreePump<'a, S: Store, Snk: NodeSink> {
    store: &'a S,
    sink: &'a Snk,
    node_fill: usize,
    levels: Vec<Level<Snk::Key, Snk::Val>>,
    prev_key: Option<Snk::Key>,
    finished: bool,
}

impl<'a, S: Store, Snk: NodeSink> TreePump<'a, S, Snk> {
    pub fn new(store: &'a S, sink: &'a Snk, max_node_size: usize, node_fill: usize) -> Self {
        assert!(
            node_fill >= 2 && node_fill <= max_node_size,
            "nodeFill must be in [2, maxNodeSize]: {node_fill}"
        );
        TreePump {
            store,
            sink,
            node_fill,
            levels: vec![Level::new(true)],
            prev_key: None,
            finished: false,
        }
    }

    /// Default pump fill: 3/4 of maxNodeSize (mapdb1/2/3 lineage). Uses saturating
    /// arithmetic so a hostile/huge `max_node_size` cannot overflow the multiply
    /// (defense in depth — the create path already bounds it to `<= 1<<20`, R3).
    pub fn default_fill(max_node_size: usize) -> usize {
        (max_node_size.saturating_mul(3) / 4).max(2)
    }

    pub fn put(&mut self, key: Snk::Key, value: Snk::Val) -> Result<()> {
        assert!(!self.finished, "pump already finished");
        if let Some(prev) = &self.prev_key {
            if self.sink.compare_keys(prev, &key) != Ordering::Less {
                return Err(DbError::NotSorted);
            }
        }
        // flush BEFORE adding: interior leaves hold exactly node_fill entries.
        if self.levels[0].keys.len() == self.node_fill {
            self.flush_leaf()?;
        }
        let leaf = &mut self.levels[0];
        leaf.keys.push(key.clone());
        leaf.values.push(value);
        self.prev_key = Some(key);
        Ok(())
    }

    /// Recid this level's next node lands in: reserved by the previous flush, or
    /// fresh for a level's first node.
    fn node_recid(&self, level: usize) -> Result<u64> {
        let pending = self.levels[level].pending;
        if pending != 0 {
            Ok(pending)
        } else {
            Ok(self.store.preallocate()?.get())
        }
    }

    fn flush_leaf(&mut self) -> Result<()> {
        let recid = self.node_recid(0)?;
        let link = self.store.preallocate()?.get();
        let (flags, keys, values, sep);
        {
            let leaf = &mut self.levels[0];
            leaf.pending = link;
            flags = if leaf.first { LEFT } else { 0 };
            leaf.first = false;
            keys = std::mem::take(&mut leaf.keys);
            values = std::mem::take(&mut leaf.values);
            sep = keys[keys.len() - 1].clone(); // leaf's inclusive high bound
        }
        self.sink.write_leaf(recid, flags, link, keys, values)?;
        self.push_up(1, sep, recid)
    }

    /// Register a flushed node with its parent level: sep = child's high bound.
    fn push_up(&mut self, level_idx: usize, sep: Snk::Key, child: u64) -> Result<()> {
        if self.levels.len() == level_idx {
            self.levels.push(Level::new(false));
        }
        if self.levels[level_idx].keys.len() == self.node_fill {
            self.flush_dir(level_idx)?;
        }
        let dir = &mut self.levels[level_idx];
        dir.keys.push(sep);
        dir.children.push(child);
        Ok(())
    }

    fn flush_dir(&mut self, level_idx: usize) -> Result<()> {
        let recid = self.node_recid(level_idx)?;
        let link = self.store.preallocate()?.get();
        let (flags, keys, children, sep);
        {
            let dir = &mut self.levels[level_idx];
            dir.pending = link;
            flags = DIR | if dir.first { LEFT } else { 0 };
            dir.first = false;
            keys = std::mem::take(&mut dir.keys);
            children = std::mem::take(&mut dir.children);
            sep = keys[keys.len() - 1].clone();
        }
        // non-rightmost dir shape: childCount == keysLen, last key = its bound.
        self.sink.write_dir(recid, flags, link, keys, children)?;
        self.push_up(level_idx + 1, sep, recid)
    }

    /// Flush the final (rightmost) node of every level, bottom-up; return the
    /// root NODE recid.
    pub fn finish(mut self) -> Result<u64> {
        assert!(!self.finished, "pump already finished");
        self.finished = true;
        let child0_recid = self.node_recid(0)?;
        let (leaf_flags, keys, values);
        {
            let leaf = &mut self.levels[0];
            leaf_flags = (if leaf.first { LEFT } else { 0 }) | RIGHT;
            keys = std::mem::take(&mut leaf.keys);
            values = std::mem::take(&mut leaf.values);
        }
        self.sink
            .write_leaf(child0_recid, leaf_flags, 0, keys, values)?;
        let mut child = child0_recid;
        for i in 1..self.levels.len() {
            let recid = self.node_recid(i)?;
            let (flags, keys, mut children);
            {
                let dir = &mut self.levels[i];
                flags = DIR | (if dir.first { LEFT } else { 0 }) | RIGHT;
                keys = std::mem::take(&mut dir.keys);
                children = std::mem::take(&mut dir.children);
            }
            children.push(child); // rightmost extra child, no key (RIGHT dir shape)
            self.sink.write_dir(recid, flags, 0, keys, children)?;
            child = recid;
        }
        Ok(child)
    }
}
