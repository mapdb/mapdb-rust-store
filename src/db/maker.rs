//! `DBMaker` — the Java-like runtime builder that returns a single
//! `DB<ConfiguredStore>` regardless of the backend chosen at run time (design
//! review §3). The typed constructors on [`DB`] remain the primary API; this
//! builder exists for source-shape compatibility with Java `DBMaker`.
//!
//! Direct-vs-WAL is `transaction_enable()` on a file builder (Java parity).
//! Read-only wraps the backend in `StoreReadOnlyWrapper`. Options that the port
//! does not model (mmap, executor, cleaner hack, checksum, file channel, JVM
//! shutdown hooks) are accepted as harmless no-ops; see PORTING-GAPS.

use crate::db::store_kind::ConfiguredStore;
use crate::db::DB;
use crate::error::{DbError, Result};
use crate::store::{StoreByteArray, StoreDirect, StoreOnHeap, StoreReadOnlyWrapper, StoreWAL};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
enum Backend {
    Heap,
    ByteArray,
    MemoryDirect,
    File(PathBuf),
}

/// A runtime DB builder (Java `DBMaker.Maker`).
pub struct DBMaker {
    backend: Backend,
    transaction_enable: bool,
    read_only: bool,
    delete_after_close: bool,
    delete_after_open: bool,
}

impl DBMaker {
    fn base(backend: Backend) -> Self {
        DBMaker {
            backend,
            transaction_enable: false,
            read_only: false,
            delete_after_close: false,
            delete_after_open: false,
        }
    }

    /// In-memory StoreDirect (Java `DBMaker.memoryDB()`).
    pub fn memory_db() -> Self {
        Self::base(Backend::MemoryDirect)
    }
    /// In-memory StoreDirect (Java `DBMaker.memoryDirectDB()`; same backend here).
    pub fn memory_direct_db() -> Self {
        Self::base(Backend::MemoryDirect)
    }
    /// Object heap store (Java `DBMaker.heapDB()`).
    pub fn heap_db() -> Self {
        Self::base(Backend::Heap)
    }
    /// In-memory byte-array store (Java `DBMaker.memoryByteArrayDB()`).
    pub fn memory_byte_array_db() -> Self {
        Self::base(Backend::ByteArray)
    }
    /// File-backed DB (Java `DBMaker.fileDB(file)`).
    pub fn file_db(path: impl AsRef<Path>) -> Self {
        Self::base(Backend::File(path.as_ref().to_path_buf()))
    }
    /// A fresh temporary file DB, deleted after close (Java `DBMaker.tempFileDB()`).
    pub fn temp_file_db() -> Self {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "mapdb5-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        path.push(unique);
        let mut m = Self::base(Backend::File(path));
        m.delete_after_close = true;
        m
    }

    /// Enable WAL transactions (file-only). Java `transactionEnable()`.
    pub fn transaction_enable(mut self) -> Self {
        self.transaction_enable = true;
        self
    }
    /// Open read-only (Java `readOnly()`).
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }
    /// Delete the backing file(s) after close (Java `fileDeleteAfterClose()`).
    pub fn file_delete_after_close(mut self) -> Self {
        self.delete_after_close = true;
        self
    }
    /// Delete the backing file immediately after open (Java `fileDeleteAfterOpen()`).
    pub fn file_delete_after_open(mut self) -> Self {
        self.delete_after_open = true;
        self
    }

    // ---- accepted no-op options (documented deviations) ----
    pub fn file_mmap_enable(self) -> Self {
        self
    }
    pub fn file_mmap_enable_if_supported(self) -> Self {
        self
    }
    pub fn executor_enable(self) -> Self {
        self
    }
    pub fn cleaner_hack_enable(self) -> Self {
        self
    }
    pub fn file_channel_enable(self) -> Self {
        self
    }
    pub fn checksum_store_enable(self) -> Self {
        self
    }
    pub fn close_on_jvm_shutdown(self) -> Self {
        self
    }
    pub fn close_on_jvm_shutdown_weak_reference(self) -> Self {
        self
    }

    /// Build the DB (Java `make()`).
    pub fn make(self) -> Result<DB<ConfiguredStore>> {
        // Reject illegal option combinations (Java `DBMaker.make` guards).
        if self.read_only && self.transaction_enable {
            return Err(DbError::wrong_config(
                "readOnly and transactionEnable are mutually exclusive",
            ));
        }
        let is_file = matches!(self.backend, Backend::File(_));
        if self.transaction_enable && !is_file {
            return Err(DbError::wrong_config(
                "transactionEnable requires a file DB",
            ));
        }
        if self.delete_after_open && !is_file {
            return Err(DbError::wrong_config(
                "fileDeleteAfterOpen requires a file DB",
            ));
        }
        if self.delete_after_close && !is_file {
            // Java rejects fileDeleteAfterClose on a non-file backend rather than
            // silently ignoring it (R10).
            return Err(DbError::wrong_config(
                "fileDeleteAfterClose requires a file DB",
            ));
        }
        if self.delete_after_open && self.transaction_enable {
            // WAL checkpointing recreates the path, so delete-after-open is unsafe.
            return Err(DbError::wrong_config(
                "fileDeleteAfterOpen is incompatible with transactionEnable (WAL)",
            ));
        }

        let mut cleanup: Vec<PathBuf> = Vec::new();
        // A file to unlink AFTER the DB is constructed + validated (delete-after-
        // open). Deferring the unlink means a failing make() over a pre-existing
        // non-MapDB file never destroys the user's data (MINOR review #10).
        let mut delete_after_open_path: Option<PathBuf> = None;
        let store: ConfiguredStore = match &self.backend {
            Backend::Heap => {
                if self.read_only {
                    ConfiguredStore::ReadOnlyHeap(StoreReadOnlyWrapper::new(StoreOnHeap::new(true)))
                } else {
                    ConfiguredStore::Heap(StoreOnHeap::new(true))
                }
            }
            Backend::ByteArray => {
                if self.read_only {
                    ConfiguredStore::ReadOnlyByteArray(StoreReadOnlyWrapper::new(
                        StoreByteArray::new(true),
                    ))
                } else {
                    ConfiguredStore::ByteArray(StoreByteArray::new(true))
                }
            }
            Backend::MemoryDirect => {
                if self.read_only {
                    ConfiguredStore::ReadOnlyDirect(StoreReadOnlyWrapper::new(
                        StoreDirect::new_heap()?,
                    ))
                } else {
                    ConfiguredStore::Direct(StoreDirect::new_heap()?)
                }
            }
            Backend::File(path) => {
                if self.delete_after_close {
                    cleanup.push(path.clone());
                }
                if self.transaction_enable {
                    ConfiguredStore::Wal(StoreWAL::open(path)?)
                } else {
                    let direct = StoreDirect::open_file(path)?;
                    if self.delete_after_open {
                        // Defer: unlink only once `DB::with_cleanup` has validated
                        // recid 1 as a real MapDB catalog. The open StoreDirect
                        // keeps working from its handle after the unlink (POSIX
                        // unlink-open).
                        delete_after_open_path = Some(path.clone());
                    }
                    if self.read_only {
                        ConfiguredStore::ReadOnlyDirect(StoreReadOnlyWrapper::new(direct))
                    } else {
                        ConfiguredStore::Direct(direct)
                    }
                }
            }
        };

        // Construct + validate the DB FIRST. If this fails (e.g. a pre-existing
        // non-MapDB file), we return the error WITHOUT unlinking anything: a path
        // this maker did not itself create is never destroyed (MINOR review #10).
        let db = DB::with_cleanup(Arc::new(store), cleanup)?;
        // Validated as a real MapDB store: now it is safe to unlink for
        // delete-after-open (the open store handle keeps working until close).
        // Java deletes both `<path>` and `<path>.ckpt` and PROPAGATES a real
        // deletion error; on failure we tear down the just-built DB (never leaking
        // an open handle) and surface the error, preserving any close error (R10).
        if let Some(path) = delete_after_open_path {
            if let Err(del_err) = remove_file_and_ckpt(&path) {
                return Err(match db.close() {
                    Ok(()) => del_err,
                    Err(close_err) => DbError::corrupt_msg(format!(
                        "fileDeleteAfterOpen failed [{del_err}]; and closing the DB also failed [{close_err}]"
                    )),
                });
            }
        }
        Ok(db)
    }
}

/// Delete `<path>` and its `<path>.ckpt` WAL sidecar. A missing file is not an
/// error (already gone); any other IO error on either file is returned (Java
/// deletes both and propagates non-NotFound failures — R10).
fn remove_file_and_ckpt(path: &Path) -> Result<()> {
    let mut ckpt = path.as_os_str().to_os_string();
    ckpt.push(".ckpt");
    for p in [path.to_path_buf(), PathBuf::from(ckpt)] {
        match std::fs::remove_file(&p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(DbError::Io(e)),
        }
    }
    Ok(())
}
