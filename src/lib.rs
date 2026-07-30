//! Rust port of the mapdb5 (Store4) storage engine.
//!
//! Bottom-up module layering mirrors the Java packages:
//! `io -> ser -> format`, `ser -> store`, `store -> {btree, htree}`.
//! **The on-disk format is not stabilised** and carries no compatibility
//! guarantee, across implementations or across versions. It may change freely
//! and without notice; see `README.md`. The Java engine at
//! <https://github.com/mapdb/mapdb-java-store> is the reference implementation;
//! see `PORTING-GAPS.md` for what this port does not yet cover.

pub mod btree;
pub mod db;
pub mod error;
pub mod io;
pub mod listener;
pub mod queue;
pub mod ser;
pub mod store;

pub use error::{DbError, Result};
pub use listener::{
    FnListener, MapExtra, MapModificationListener, ModificationAwareMap,
    SynchronousMapModificationListener,
};
