//! `db` layer — the MapDB `DB`/`DBMaker` facade.
//!
//! - [`catalog`] — the MDBC-v1 name catalog codec at recid 1 (byte-compatible
//!   with Java).
//! - [`descriptor`] — stable, Java-compatible codec descriptor strings.
//! - [`store_kind`] — the [`ConfiguredStore`] forwarding enum backing the runtime
//!   [`DBMaker`].
//! - [`atomic`] — `Atomic.Long/Integer/Boolean/String/Var`.
//! - [`set`] — map-backed navigable set + the no-value format.
//! - [`db`] — the [`DB`] facade + typed makers + instance cache + close/Drop.
//! - [`maker`] — the [`DBMaker`] runtime builder.
//! - [`bind`] — secondary indexes / derived views over a primary `BTreeMap`.

pub mod atomic;
pub mod bind;
pub mod catalog;
pub mod descriptor;
pub mod maker;
pub mod set;
pub mod store_kind;

#[allow(clippy::module_inception)]
pub mod db;

#[cfg(test)]
mod tests;

pub use atomic::{AtomicBoolean, AtomicInteger, AtomicLong, AtomicString, AtomicVar};
pub use catalog::{NameCatalog, CATALOG_SER, RECID_CATALOG};
pub use db::{DbRollback, DB};
pub use descriptor::{GroupDescriptor, SerDescriptor};
pub use maker::DBMaker;
pub use set::{NavigableSet, NavigableSetView, NoValueFormat, NoValueSer};
pub use store_kind::ConfiguredStore;
