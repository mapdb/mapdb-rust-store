//! Crash-tier workload for mapdb5 (WAL backend). Writes forever until
//! SIGKILLed, under the write-ahead intent-journal protocol of `mapdb-rust-store-crash-harness`
//! — every batch's intent is durable in the journal (on a *different*
//! filesystem) before its mutations touch the store, and a durability ACK is
//! journaled only after the WAL commit barrier returned. `crash_check` replays
//! the journal against the recovered store.
//!
//! # WAL only (a deliberate deviation from the io-uring harness)
//!
//! The io-uring harness ran both a copy-on-write Direct backend and a WAL
//! backend, because *both* were power-cut recoverable. In mapdb5 only
//! [`StoreWAL`] is transactional. [`StoreDirect`] applies record mutations in
//! place and stamps a header checksum over the allocator words: an
//! allocator-changing uncommitted mutation makes a reopen fail the checksum
//! ("store was not closed cleanly"), while a same-capacity in-place update can
//! leave a valid header and expose *uncommitted* bytes on reopen. Either way it
//! does not recover to the last committed state at a random cut point, so a
//! recover-to-acked-state oracle cannot be satisfied on Direct. The crash tier
//! therefore exercises WAL, the store mapdb5 actually offers crash guarantees
//! for. (A separate "kill only at a committed boundary" durability test could
//! cover Direct; it is not this random-cut crash test and is out of scope.)
//!
//! # Protocol
//!
//! Group commit: a group's batches are all generated and their intents synced
//! first, then every batch's puts/removes are applied to the map, then the
//! progress marker [`COMMITTED_SEQ_KEY`] is set to the group's last sequence,
//! then one [`DB::commit`](mapdb_rust_store::db::DB::commit) makes the whole group (marker
//! included) durable and the ACK is journaled. There is no store-exposed
//! transaction id, so the harness's own 1-based sequence number *is* the txid: a
//! group's ACK/txid is its last batch's `seq`, always a multiple of `group`.
//! Committed states are therefore exactly the group boundaries, and the
//! committed marker records which one — the facts `crash_check` relies on.
//!
//! Maintenance: [`DB::compact`](mapdb_rust_store::db::DB::compact) on a WAL store performs
//! a log-compacting checkpoint (snapshot + fsync + atomic rename, then replay
//! from the truncated log), so it is journaled as a `compact` maintenance
//! interval and is the coverage the readiness policy requires.
//!
//! Readiness: only after the header, the run-id batch plus two later durable
//! groups, and one completed compaction does the workload journal an `R` record
//! and create the `<journal>.ready` sentinel the harness scripts wait for. A cut
//! before readiness fails the round rather than passing vacuously.

use mapdb_rust_store_crash_harness::{
    self as ch, Config, GenOp, MaintKind, Model, Record, COMMITTED_SEQ_KEY, RUN_ID_KEY,
};
use mapdb_rust_store::db::DB;
use mapdb_rust_store::ser::bytearray::ByteArrayFormat;

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Maintenance cadence in durable groups: compact (WAL checkpoint) at group 2,
/// then every 6th group — early enough that the readiness threshold (>=3 groups,
/// >=1 compaction) is met by group 3, and often enough to keep exercising the
/// log-truncation replay path over a long run.
const COMPACT_EVERY: u64 = 6;

struct Journal {
    file: std::fs::File,
}

impl Journal {
    fn create(path: &Path) -> Result<Journal, String> {
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("journal create {}: {e}", path.display()))?;
        Ok(Journal { file })
    }
    /// One complete line per `write_all` — the checker tolerates exactly one
    /// torn (newline-less) tail, so a mid-line crash is expected, never fatal.
    fn append(&mut self, rec: &Record) -> Result<(), String> {
        self.file
            .write_all(&ch::encode_line(rec))
            .map_err(|e| format!("journal write: {e}"))
    }
    fn sync(&mut self) -> Result<(), String> {
        self.file
            .sync_data()
            .map_err(|e| format!("journal sync: {e}"))
    }
}

fn run(cfg: Config, store_path: PathBuf, journal_path: PathBuf) -> Result<(), String> {
    let db = DB::make_wal(&store_path).map_err(|e| format!("open wal: {e:?}"))?;
    // Bound the WAL log so the inline auto-checkpoint keeps firing under load,
    // exercising replay-from-a-truncated-log recovery between explicit compacts.
    db.store()
        .set_auto_checkpoint_bytes(cfg.max_wal_bytes as i64)
        .map_err(|e| format!("auto-checkpoint config: {e:?}"))?;
    // Byte-array keys and values so the generated universe round-trips exactly;
    // values-outside-nodes so the large (>256 KiB) size classes become external
    // linked records rather than blowing the node size.
    let map = db
        .tree_map("crash", ByteArrayFormat, ByteArrayFormat)
        .values_outside_nodes_enable()
        .create()
        .map_err(|e| format!("create map: {e:?}"))?;

    let mut journal = Journal::create(&journal_path)?;
    journal.append(&Record::Header(cfg.clone()))?;
    journal.sync()?;

    let mut model = Model::new();
    let mut seq = 0u64;
    let mut groups = 0u64;
    let mut compactions = 0u64;
    // StoreWAL folds checkpoint into compact (compact == checkpoint), so the
    // harness never emits a separate `checkpoint` maintenance record.
    let checkpoints = 0u64;
    let mut ready = false;

    loop {
        // --- generate the group and journal its intents (write-ahead) ---
        let mut group_batches: Vec<(u64, Vec<GenOp>)> = Vec::with_capacity(cfg.group as usize);
        for _ in 0..cfg.group {
            seq += 1;
            let ops = ch::gen_batch(&mut model, &cfg, seq);
            group_batches.push((seq, ops));
        }
        for (s, ops) in &group_batches {
            journal.append(&Record::Intent {
                seq: *s,
                txid: *s,
                digest: ch::batch_digest(ops),
            })?;
        }
        journal.sync()?;

        // --- apply every batch's ops to the map, in order (uncommitted) ---
        for (s, ops) in &group_batches {
            if *s == 1 {
                // The run-id marker rides the first batch: a journal
                // can never be checked against another round's store image.
                map.put(RUN_ID_KEY.to_vec(), cfg.run_id.clone().into_bytes())
                    .map_err(|e| format!("put run-id: {e:?}"))?;
            }
            for op in ops {
                match *op {
                    GenOp::Insert(k, v) => {
                        map.put(ch::key_bytes(k), ch::value_bytes(cfg.seed, v))
                            .map_err(|e| format!("put seq {s}: {e:?}"))?;
                    }
                    GenOp::Remove(k) => {
                        map.remove(&ch::key_bytes(k))
                            .map_err(|e| format!("remove seq {s}: {e:?}"))?;
                    }
                }
            }
        }

        // --- progress marker, then one durability barrier per group, then ACK ---
        // The group's txid is its last batch's seq (a group boundary). The marker
        // is committed atomically with the group, so a recovered boundary always
        // carries its own seq — the checker reads it rather than guessing. After
        // commit() returns the whole group is durable, so the ACK proves it.
        let ack = group_batches.last().expect("non-empty group").0;
        map.put(COMMITTED_SEQ_KEY.to_vec(), ack.to_string().into_bytes())
            .map_err(|e| format!("put committed-seq: {e:?}"))?;
        db.commit().map_err(|e| format!("commit: {e:?}"))?;
        journal.append(&Record::Ack { txid: ack })?;
        journal.sync()?;
        groups += 1;

        // --- explicit maintenance (WAL log-compacting checkpoint) ---
        if groups % COMPACT_EVERY == 2 {
            compactions += 1;
            journal.append(&Record::Maint {
                begin: true,
                kind: MaintKind::Compact,
                ordinal: compactions,
            })?;
            journal.sync()?;
            db.compact().map_err(|e| format!("compact: {e:?}"))?;
            journal.append(&Record::Maint {
                begin: false,
                kind: MaintKind::Compact,
                ordinal: compactions,
            })?;
            journal.sync()?;
        }

        // --- readiness: durable coverage, then the sentinel ---
        if !ready && groups >= ch::READY_MIN_GROUPS && compactions >= ch::READY_MIN_COMPACTIONS {
            journal.append(&Record::Ready {
                ack_txid: ack,
                groups,
                checkpoints,
                compactions,
            })?;
            journal.sync()?;
            let sentinel = sentinel_path(&journal_path);
            std::fs::write(&sentinel, b"ready")
                .and_then(|_| std::fs::File::open(&sentinel)?.sync_all())
                .map_err(|e| format!("sentinel: {e}"))?;
            eprintln!(
                "crash_workload: ready after {groups} groups (ack txid {ack}, {compactions} compactions)"
            );
            ready = true;
        }
    }
}

fn sentinel_path(journal: &Path) -> PathBuf {
    let mut p = journal.as_os_str().to_owned();
    p.push(".ready");
    PathBuf::from(p)
}

fn usage() -> ! {
    eprintln!(
        "usage: crash_workload --backend wal --store <path> --journal <path> \
         --run-id <id> [--seed N] [--keys N] [--batch-ops N] [--group N]"
    );
    std::process::exit(2);
}

fn main() {
    let mut backend = None;
    let mut store = None;
    let mut journal = None;
    let mut run_id = None;
    let (mut seed, mut keys, mut batch_ops, mut group) = (1u64, 512u32, 6u32, 8u32);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--backend" => backend = Some(val()),
            "--store" => store = Some(PathBuf::from(val())),
            "--journal" => journal = Some(PathBuf::from(val())),
            "--run-id" => run_id = Some(val()),
            "--seed" => seed = val().parse().unwrap_or_else(|_| usage()),
            "--keys" => keys = val().parse().unwrap_or_else(|_| usage()),
            "--batch-ops" => batch_ops = val().parse().unwrap_or_else(|_| usage()),
            "--group" => group = val().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let (Some(backend), Some(store), Some(journal), Some(run_id)) =
        (backend, store, journal, run_id)
    else {
        usage()
    };
    if batch_ops < 3 || group == 0 || keys == 0 || run_id.contains(' ') {
        usage();
    }
    if backend != "wal" {
        eprintln!(
            "crash_workload: FATAL: backend '{backend}' is not crash-recoverable in mapdb5; \
             only 'wal' is supported by the crash tier (StoreDirect is non-transactional)"
        );
        std::process::exit(2);
    }
    let cfg = Config {
        run_id,
        backend,
        seed,
        keys,
        batch_ops,
        group,
        max_wal_bytes: 64 << 20,
    };
    // The loop only returns on error; a healthy workload dies by SIGKILL.
    if let Err(e) = run(cfg, store, journal) {
        eprintln!("crash_workload: FATAL: {e}");
        std::process::exit(1);
    }
}
