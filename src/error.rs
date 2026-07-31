//! Error type — the Java `DBException` hierarchy collapsed into one
//! `#[non_exhaustive]` enum (spec 02-store §2, decision D10).
//!
//! Every fallible store/format operation returns `Result<_, DbError>`.
//! TCK ports assert exact variants, so variant identity is part of the
//! contract.

use std::fmt;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, DbError>;

/// The single error enum. Mirrors the Java `DBException` subclasses:
/// - `GetVoid` ← `GetVoid` (read of a never-written / deleted recid)
/// - `RecordTooLarge` ← `RecordTooLarge`
/// - `DataCorruption` ← `DataCorruption` / `PointerChecksumBroken` / `WrongConfig`
/// - `StoreFull` ← `VolumeIOError`/allocator ceiling
/// - `StoreClosed` ← `StoreClosed`
/// - `VerifyFailed` ← `DataCorruption` raised by `verify()`
/// - `NotSorted` ← pump misorder
/// - `AlreadyOpen` ← duplicate-open lease (D12)
/// - `Io` ← `VolumeIOError` wrapping a real `std::io::Error`
#[non_exhaustive]
#[derive(Debug)]
pub enum DbError {
    /// Read of a Void or Deleted recid. Carries the offending recid.
    GetVoid(u64),
    /// Record content exceeds the maximum single-record capacity.
    RecordTooLarge,
    /// On-disk bytes failed a structural/parity/checksum invariant.
    /// The `&'static str` names the specific check for diagnostics.
    DataCorruption(Corruption),
    /// Allocator hit the 44-bit volume ceiling (or backing store is full).
    StoreFull,
    /// Operation attempted on a closed store.
    StoreClosed,
    /// `verify()` found the on-disk tiling inconsistent.
    VerifyFailed(String),
    /// Pump input was not strictly ascending (misorder or duplicate).
    NotSorted,
    /// A conflicting handle already holds the open lease for this header recid.
    AlreadyOpen { header_recid: u64 },
    /// A mutating operation was attempted on a logically read-only store
    /// (`StoreReadOnlyWrapper`). Mirrors Java's
    /// `UnsupportedOperationException("store is read-only")`.
    ReadOnly,
    /// A deliberately-unsupported API was invoked on a HEALTHY store (Java's
    /// `UnsupportedOperationException`) — e.g. a columnar scan or bulk build on an
    /// external-value map. Distinct from `DataCorruption` so callers do not treat
    /// the store as damaged. The `&'static str` names the unsupported operation.
    Unsupported(&'static str),
    /// A DB-facade configuration/usage error (Java `DBException.WrongConfiguration`):
    /// name already exists / does not exist, catalog type mismatch, a custom codec
    /// that was not re-supplied on reopen, an illegal collection name, or an
    /// illegal `DBMaker` option combination. Carries a human-readable reason.
    WrongConfiguration(String),
    /// A DB per-name instance-cache hit whose cached concrete handle type does not
    /// match the type the caller is opening, even though the stored catalog `#type`
    /// and descriptors agree. Distinct from
    /// [`WrongConfiguration`](DbError::WrongConfiguration) so a caller can tell a
    /// genuine mis-typed reopen from an already-open-with-a-different-Rust-type
    /// collision. Carries a human-readable reason.
    CachedTypeMismatch(String),
    /// Another handle — in this process or another one — holds the store lock
    /// on the same namespace. Java raises a plain `DBException` for both the
    /// cross-process case ("locked by another process") and the in-process one
    /// (`OverlappingFileLockException`); the port keeps them one variant
    /// because a caller can do nothing different about either. Carries the
    /// human-readable reason.
    Locked(String),
    /// Underlying I/O failure.
    Io(std::io::Error),
}

/// Detail payload for `DataCorruption`. Usually a `'static` reason string
/// naming the failed check; `Msg` carries a dynamic message when needed.
#[derive(Debug)]
pub enum Corruption {
    /// A named structural check failed (parity, magic, bounds, ...).
    Reason(&'static str),
    /// A dynamically-formatted corruption message.
    Msg(String),
}

impl DbError {
    /// Convenience constructor for a named corruption.
    pub fn corrupt(reason: &'static str) -> Self {
        DbError::DataCorruption(Corruption::Reason(reason))
    }
    /// Convenience constructor for a dynamic corruption message.
    pub fn corrupt_msg(msg: impl Into<String>) -> Self {
        DbError::DataCorruption(Corruption::Msg(msg.into()))
    }
    /// Convenience constructor for a wrong-configuration facade error.
    pub fn wrong_config(msg: impl Into<String>) -> Self {
        DbError::WrongConfiguration(msg.into())
    }
}

impl fmt::Display for Corruption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Corruption::Reason(r) => f.write_str(r),
            Corruption::Msg(m) => f.write_str(m),
        }
    }
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DbError::GetVoid(r) => write!(f, "record does not exist (recid={r})"),
            DbError::RecordTooLarge => f.write_str("record too large"),
            DbError::DataCorruption(c) => write!(f, "data corruption: {c}"),
            DbError::StoreFull => f.write_str("store full"),
            DbError::StoreClosed => f.write_str("store closed"),
            DbError::VerifyFailed(m) => write!(f, "verify failed: {m}"),
            DbError::NotSorted => f.write_str("input not sorted"),
            DbError::AlreadyOpen { header_recid } => {
                write!(f, "already open (header_recid={header_recid})")
            }
            DbError::ReadOnly => f.write_str("store is read-only"),
            DbError::Unsupported(op) => write!(f, "unsupported operation: {op}"),
            DbError::WrongConfiguration(m) => write!(f, "wrong configuration: {m}"),
            DbError::CachedTypeMismatch(m) => write!(f, "cached type mismatch: {m}"),
            DbError::Locked(m) => write!(f, "store is locked: {m}"),
            DbError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DbError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DbError {
    fn from(e: std::io::Error) -> Self {
        DbError::Io(e)
    }
}

/// `try_reserve` failures (bounded-allocation ceilings, D4) map to corruption,
/// never abort.
impl From<std::collections::TryReserveError> for DbError {
    fn from(_: std::collections::TryReserveError) -> Self {
        DbError::corrupt("allocation ceiling exceeded")
    }
}
