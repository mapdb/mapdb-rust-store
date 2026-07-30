//! `queue` layer — persistent queues over a Store4 store.
//!
//! - [`QueueLong`] — a persistent FIFO of `(timestamp, value)` long pairs, a
//!   direct store primitive constructed from three pointer recids (not a named
//!   DB catalog object).
//! - [`PersistentBlockingQueue`] — a generic (serializer-based) FIFO / LIFO /
//!   circular queue with blocking `take`/`put` (`Condvar` + `Mutex`).
//!
//! Persisted node/header records are byte-for-byte compatible with Java; see
//! each type's module docs and the golden-vector tests.

pub mod blocking;
pub mod long;

pub use blocking::{Mode, PersistentBlockingQueue};
pub use long::{Node, QueueLong, QueueLongTakeUntil};

#[cfg(test)]
mod tests;
