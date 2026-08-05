//! Rust port of the mapdb5 (Store4) storage engine.
//!
//! Bottom-up module layering mirrors the Java packages:
//! `io -> ser -> format`, `ser -> store`, `store -> {btree, htree}`.
//! **The on-disk format is not stabilised** and carries no compatibility
//! guarantee, across implementations or across versions. It may change freely
//! and without notice; see `README.md`. The Java engine at
//! <https://github.com/mapdb/mapdb-java-store> is the reference implementation;
//! see `PORTING-GAPS.md` for what this port does not yet cover.

// `src/store/xfix.rs` is compiled BOTH here and — via `#[path]` — into
// `tests/xfixture_conformance.rs` and `tests/wal3_decode.rs` (decision C-D3:
// the schema-v2 `ro` cells need the crate-internal read-only opener, and the
// rest of the harness must stay shared). `crate::` means a different crate in
// each build, so that file names this one by its package name; this alias is
// what makes the name resolve inside the lib. Test-only, so nothing about the
// published crate changes.
#[cfg(test)]
extern crate self as mapdb_rust_store;

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
