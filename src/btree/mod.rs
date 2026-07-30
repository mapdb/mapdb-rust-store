//! `btree` layer — ordered maps over a Store4 store (spec 03).
//!
//! This layer provides [`BTreeMap`] (B-link tree: lock-free push-down readers +
//! Lehman-Yao concurrent writers), [`TreePump`] bulk loading, and the shared
//! navigable [`RangeView`] layer.

pub mod map;
pub mod pump;
pub mod view;

// `node` mirrors Java's PRIVATE `BTreeMap.Node` / `NodeSerializer` inner classes:
// crate-internal so no external caller can construct a structurally-impossible
// node object and slip it past the byte-side validation into the object read
// path. All persisted (byte-store) nodes are still
// validated in `NodeSerializer::deserialize`.
pub(crate) mod node;

#[cfg(test)]
mod tests;

pub use map::{BTreeMap, EntryIter};
pub use pump::{NodeSink, TreePump};
pub use view::{OrderedMapAdapter, RangeView};
