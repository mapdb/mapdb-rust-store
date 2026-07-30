//! Crash-tier checker for mapdb5 (WAL). Opens the crashed (or crash-imaged)
//! store through the production `DB::make_wal` recovery path and holds it to an
//! exhaustive oracle:
//!
//! - `make_wal` (WAL log replay) must succeed — any refusal or corruption
//!   verdict FAILs;
//! - the journal must satisfy the strict grammar (contiguous intents with
//!   `txid == seq`, digest agreement with the regenerated batches, one torn
//!   tail at most);
//! - the recovered contents must EXACTLY equal the deterministic replay through
//!   the *committed group boundary* `g` the store actually reached. mapdb5
//!   exposes no recovery watermark (no `visible_txid`), but the workload commits
//!   the [`COMMITTED_SEQ_KEY`] progress marker inside every group, so the
//!   recovered store carries `g` directly. The checker reads it — never guesses
//!   it from possibly-aliased contents — and REQUIRES it to be a multiple of
//!   `group` in `[max_ack, floor(max_intent / group) * group]`: `g >= max_ack`
//!   is durability (no acked group may be lost, even when later ops cancel it
//!   out), `g <= floor(max_intent/group)*group` is the prefix bound (no state
//!   without a completed write-ahead intent). Because commits are per-group that
//!   window is at most one in-flight group wide.
//! - the recovered contents are checked TWO independent ways against `replay(g)`:
//!   a whole-map `entries()` scan (strictly increasing, no dup, byte-exact
//!   equality with the expected map — rules out any extra/missing/wrong key,
//!   inside the universe or not) AND a point-lookup `get()` of every universe key
//!   plus both markers (exercises the routing path independently). `size_long()`
//!   must equal the replayed entry count, and `store.verify()` must pass.
//!
//! Verdict: one stable stdout line
//! `CRASH_CHECK verdict=PASS|FAIL [reason=<code>] backend=wal recovered_txid=…
//! ack_txid=… intents=… entries=… ready_groups=… ready_checkpoints=…
//! ready_compactions=… last_record=… maint_open_at_cut=0|1`; diagnostics on
//! stderr; exit 0 iff PASS. `maint_open_at_cut` reports an unmatched `M begin`
//! at the witness cutoff — NOT proof the cut landed inside compaction (the
//! process may die before the `M done` sync).

use mapdb_rust_store_crash_harness::{self as ch, Config, Model, Record, COMMITTED_SEQ_KEY, RUN_ID_KEY};
use mapdb_rust_store::db::DB;
use mapdb_rust_store::ser::bytearray::ByteArrayFormat;
use mapdb_rust_store::store::{Store, StoreWAL};

use std::collections::BTreeMap;
use std::path::PathBuf;

struct Fail {
    reason: &'static str,
    detail: String,
}

fn fail(reason: &'static str, detail: impl Into<String>) -> Fail {
    Fail {
        reason,
        detail: detail.into(),
    }
}

struct JournalView {
    cfg: Config,
    max_intent: u64,
    max_ack: u64,
    /// The prefix-validated `R` record: (ack_txid, groups, checkpoints,
    /// compactions). Coverage is REQUIRED from these fields, not assumed.
    ready: Option<(u64, u64, u64, u64)>,
    /// A trailing `M begin` without its `done` at the witness cutoff. Does NOT
    /// prove the cut landed inside compaction (the process may have died between
    /// the call returning and the `M done` sync) — hence the honest name.
    maint_open_at_cut: bool,
    /// The type token of the last complete journal record (result metadata).
    last_record: &'static str,
    intents: Vec<(u64, u32)>, // (seq == txid, digest)
}

/// The workload's group protocol, enforced over the ORDERED records — stronger
/// than the aggregate `max_ack`/`max_intent` checks in `check`, and independent
/// of the store. The shared grammar only proves ACKs
/// are monotone and below the intent frontier; a *workload* regression (a
/// partial-group ACK, a duplicate ACK, or advancing two groups deep without an
/// ACK) would otherwise reach the store oracle and could PASS. Here every ACK
/// must be the exact next boundary (`group`, `2*group`, …) with its whole group
/// of intents already present, maintenance may only run with no in-flight group,
/// and at the cut at most one group may be outstanding.
fn validate_group_protocol(records: &[Record], group: u64) -> Result<(), Fail> {
    let mut max_intent = 0u64;
    let mut max_ack = 0u64;
    let mut next_boundary = group; // the first ACK must be exactly `group`
    for rec in records {
        match rec {
            // The shared grammar already guarantees intents are contiguous from 1.
            Record::Intent { seq, .. } => max_intent = *seq,
            Record::Ack { txid } => {
                if *txid != next_boundary {
                    return Err(fail(
                        "group-ack-order",
                        format!("ACK {txid} is not the next group boundary {next_boundary}"),
                    ));
                }
                if max_intent < *txid {
                    return Err(fail(
                        "group-ack-order",
                        format!("ACK {txid} above the intent frontier {max_intent}"),
                    ));
                }
                max_ack = *txid;
                next_boundary += group;
            }
            // Maintenance runs between acknowledged groups — never with a group
            // in flight (the workload acks a group before generating the next).
            Record::Maint { .. } => {
                if max_intent != max_ack {
                    return Err(fail(
                        "group-maint-order",
                        format!("maintenance with an in-flight group (intent {max_intent} > ack {max_ack})"),
                    ));
                }
            }
            _ => {}
        }
    }
    // At the cut at most one group may be outstanding (unfloored — catches a
    // journal that advanced deep into a second unacknowledged group).
    if max_intent - max_ack > group {
        return Err(fail(
            "group-window",
            format!("intent frontier {max_intent} is more than one group ahead of ack {max_ack}"),
        ));
    }
    Ok(())
}

fn load_journal(path: &PathBuf) -> Result<JournalView, Fail> {
    let bytes = std::fs::read(path).map_err(|e| fail("journal-read", e.to_string()))?;
    let (records, _torn) =
        ch::parse_journal(&bytes).map_err(|e| fail(e.0, "journal grammar violation"))?;
    // Enforce the group protocol over the ordered records before collapsing them.
    let group = match records.first() {
        Some(Record::Header(c)) => c.group as u64,
        _ => return Err(fail("journal-no-header", "missing header")),
    };
    validate_group_protocol(&records, group)?;
    let mut cfg = None;
    let mut intents = Vec::new();
    let mut max_ack = 0;
    let mut ready = None;
    let mut open_maint = false;
    let mut last_record = "none";
    for rec in records {
        last_record = match &rec {
            Record::Header(_) => "H",
            Record::Intent { .. } => "I",
            Record::PostApply { .. } => "P",
            Record::Ack { .. } => "F",
            Record::Maint { .. } => "M",
            Record::Ready { .. } => "R",
        };
        match rec {
            Record::Header(c) => cfg = Some(c),
            Record::Intent { seq, digest, .. } => intents.push((seq, digest)),
            Record::PostApply { .. } => {}
            Record::Ack { txid } => max_ack = txid,
            Record::Maint { begin, .. } => open_maint = begin,
            Record::Ready {
                ack_txid,
                groups,
                checkpoints,
                compactions,
            } => ready = Some((ack_txid, groups, checkpoints, compactions)),
        }
    }
    let cfg = cfg.ok_or_else(|| fail("journal-no-header", "missing header"))?;
    Ok(JournalView {
        max_intent: intents.last().map(|(s, _)| *s).unwrap_or(0),
        max_ack,
        ready,
        maint_open_at_cut: open_maint,
        last_record,
        intents,
        cfg,
    })
}

/// Replays the generator through every intent (verifying each digest — the
/// workload and checker must agree batch-for-batch), capturing the model at each
/// requested group boundary in `wanted`.
fn replay_capture(view: &JournalView, wanted: &[u64]) -> Result<BTreeMap<u64, Model>, Fail> {
    let want: std::collections::BTreeSet<u64> = wanted.iter().copied().collect();
    let mut model = Model::new();
    let mut snaps = BTreeMap::new();
    for &(seq, digest) in &view.intents {
        let ops = ch::gen_batch(&mut model, &view.cfg, seq);
        if ch::batch_digest(&ops) != digest {
            return Err(fail(
                "intent-digest-mismatch",
                format!("seq {seq}: regenerated batch disagrees with journal"),
            ));
        }
        if want.contains(&seq) {
            snaps.insert(seq, model.clone());
        }
    }
    Ok(snaps)
}

/// The exact expected store contents at committed boundary `g`: every live
/// key's regenerated value bytes, plus the run-id and committed-seq markers.
fn expected_bytes(model: &Model, cfg: &Config, g: u64) -> BTreeMap<Vec<u8>, Vec<u8>> {
    let mut m = BTreeMap::new();
    for (&k, &vid) in model {
        m.insert(ch::key_bytes(k), ch::value_bytes(cfg.seed, vid));
    }
    m.insert(RUN_ID_KEY.to_vec(), cfg.run_id.clone().into_bytes());
    m.insert(COMMITTED_SEQ_KEY.to_vec(), g.to_string().into_bytes());
    m
}

fn check(
    backend: &str,
    store: &PathBuf,
    journal: &PathBuf,
    min_ack: u64,
) -> Result<(JournalView, u64, u64), Fail> {
    let view = load_journal(journal)?;
    if view.cfg.backend != backend {
        return Err(fail("config", "journal backend disagrees with --backend"));
    }
    if backend != "wal" {
        return Err(fail("config", "only the wal backend is crash-recoverable"));
    }
    // Coverage is enforced from the prefix-validated R record itself  — a journal without real compaction coverage cannot
    // pass, whatever its other records claim. (StoreWAL folds checkpoint into
    // compact, so the WAL checkpoint requirement collapses into compactions.)
    let Some((_r_ack, r_groups, _r_ckpts, r_compacts)) = view.ready else {
        return Err(fail("not-ready", "cut before the readiness record"));
    };
    if r_groups < ch::READY_MIN_GROUPS || r_compacts < ch::READY_MIN_COMPACTIONS {
        return Err(fail(
            "coverage",
            format!(
                "R records groups={r_groups} compactions={r_compacts} below the readiness policy"
            ),
        ));
    }
    if view.max_ack < min_ack {
        return Err(fail(
            "min-ack",
            format!("ack frontier {} below required {min_ack}", view.max_ack),
        ));
    }

    // The group invariants the boundary oracle depends on, enforced here rather
    // than in the shared journal grammar: the ACK
    // frontier must be a group boundary, and at most one further group may be in
    // flight, so the store can only have recovered to `max_ack` or the next
    // boundary. A journal that violates these is rejected, never trusted.
    let group = view.cfg.group as u64;
    if view.max_ack % group != 0 {
        return Err(fail(
            "ack-not-boundary",
            format!(
                "ack frontier {} is not a multiple of group {group}",
                view.max_ack
            ),
        ));
    }
    let g_hi = (view.max_intent / group) * group;
    if g_hi < view.max_ack || g_hi - view.max_ack > group {
        return Err(fail(
            "window-invalid",
            format!(
                "committed window [ack {}, floor(intent {}/group)={g_hi}] is not one in-flight group wide",
                view.max_ack, view.max_intent
            ),
        ));
    }

    let db = DB::<StoreWAL>::make_wal(store).map_err(|e| fail("open", format!("{e:?}")))?;
    let outcome = check_open(&db, &view, group, g_hi);
    // Always close (best-effort on the error path) so the WAL file is released.
    match outcome {
        Ok((recovered, entries)) => {
            db.close().map_err(|e| fail("close", format!("{e:?}")))?;
            Ok((view, recovered, entries))
        }
        Err(f) => {
            let _ = db.close();
            Err(f)
        }
    }
}

/// The open-store half of the oracle: read the recovered boundary from the
/// store's own committed marker, validate it, and hold the recovered contents to
/// the replay of exactly that boundary — two independent read paths.
fn check_open(
    db: &DB<StoreWAL>,
    view: &JournalView,
    group: u64,
    g_hi: u64,
) -> Result<(u64, u64), Fail> {
    let map = db
        .tree_map("crash", ByteArrayFormat, ByteArrayFormat)
        .open()
        .map_err(|e| fail("open-map", format!("{e:?}")))?;

    // Store-level invariant oracle (Java `Store.verify`): allocator/index/
    // free-list geometry of the inner StoreDirect.
    db.store()
        .verify()
        .map_err(|e| fail("verify", format!("{e:?}")))?;

    // Read the whole recovered map once. Iteration must be strictly increasing
    // (an out-of-order scan is corruption a normalized BTreeMap would hide);
    // a BTreeMap then catches an extra key anywhere.
    let entries = map.entries().map_err(|e| fail("read", format!("{e:?}")))?;
    let mut actual: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut prev: Option<Vec<u8>> = None;
    for (k, v) in entries {
        if let Some(p) = &prev {
            if &k <= p {
                return Err(fail(
                    "iteration-order",
                    "map iteration was not strictly increasing",
                ));
            }
        }
        prev = Some(k.clone());
        actual.insert(k, v);
    }
    let size = map
        .size_long()
        .map_err(|e| fail("size", format!("{e:?}")))?;
    if size as usize != actual.len() {
        return Err(fail(
            "size-mismatch",
            format!("size_long {size} != iterated entries {}", actual.len()),
        ));
    }

    // Run-id marker: this store image belongs to this journal.
    match actual.get(RUN_ID_KEY) {
        Some(v) if v.as_slice() == view.cfg.run_id.as_bytes() => {}
        other => {
            return Err(fail(
                "run-id",
                format!(
                    "marker {:?} != {:?}",
                    other.map(|v| v.len()),
                    view.cfg.run_id
                ),
            ));
        }
    }

    // The recovered boundary, read directly from the store's committed marker —
    // never guessed from contents. It must be a group
    // boundary in [max_ack, g_hi]: below max_ack loses an acked group, above g_hi
    // recovers a group with no completed intent.
    let g = match actual.get(COMMITTED_SEQ_KEY) {
        Some(v) => std::str::from_utf8(v)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(|| {
                fail(
                    "committed-marker",
                    "committed-seq marker is not a decimal u64",
                )
            })?,
        None => {
            return Err(fail(
                "committed-marker",
                "recovered store has no committed-seq marker",
            ))
        }
    };
    if g % group != 0 || g < view.max_ack || g > g_hi {
        return Err(fail(
            "recovered-boundary-range",
            format!(
                "committed boundary {g} outside group multiples of [ack {}, {g_hi}]",
                view.max_ack
            ),
        ));
    }

    // Replay to exactly that boundary (verifying every intent digest en route),
    // and hold the recovered contents to it two independent ways.
    let model = replay_capture(view, &[g])?
        .remove(&g)
        .ok_or_else(|| fail("replay", format!("no model snapshot at boundary {g}")))?;
    let expected = expected_bytes(&model, &view.cfg, g);

    // (1) whole-map scan equality.
    if actual != expected {
        return Err(fail(
            "state-mismatch",
            format!(
                "recovered scan != replay({g}) (entries {}, expected {})",
                actual.len(),
                expected.len()
            ),
        ));
    }
    // (2) independent point-lookup path over every universe key + both markers.
    for k in 0..view.cfg.keys {
        let key = ch::key_bytes(k);
        let got = map.get(&key).map_err(|e| fail("read", format!("{e:?}")))?;
        let want = model
            .get(&k)
            .map(|&vid| ch::value_bytes(view.cfg.seed, vid));
        if got != want {
            return Err(fail(
                "point-mismatch",
                format!(
                    "get(key {k}) = {:?}, replay({g}) = {:?}",
                    got.map(|v| v.len()),
                    want.map(|v| v.len())
                ),
            ));
        }
    }
    for (marker, want) in [
        (RUN_ID_KEY, view.cfg.run_id.clone().into_bytes()),
        (COMMITTED_SEQ_KEY, g.to_string().into_bytes()),
    ] {
        if map
            .get(&marker.to_vec())
            .map_err(|e| fail("read", format!("{e:?}")))?
            != Some(want)
        {
            return Err(fail("point-mismatch", "marker get() disagrees with replay"));
        }
    }
    Ok((g, actual.len() as u64))
}

fn usage() -> ! {
    eprintln!("usage: crash_check --backend wal --store <path> --journal <path> [--min-ack N]");
    std::process::exit(2);
}

fn main() {
    let mut backend = None;
    let mut store = None;
    let mut journal = None;
    let mut min_ack = 1u64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        match a.as_str() {
            "--backend" => backend = Some(val()),
            "--store" => store = Some(PathBuf::from(val())),
            "--journal" => journal = Some(PathBuf::from(val())),
            "--min-ack" => min_ack = val().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    let (Some(backend), Some(store), Some(journal)) = (backend, store, journal) else {
        usage()
    };
    match check(&backend, &store, &journal, min_ack) {
        Ok((view, recovered, entries)) => {
            let (_, r_groups, r_ckpts, r_compacts) = view.ready.unwrap_or_default();
            println!(
                "CRASH_CHECK verdict=PASS backend={backend} recovered_txid={recovered} \
                 ack_txid={} intents={} entries={entries} ready_groups={r_groups} \
                 ready_checkpoints={r_ckpts} ready_compactions={r_compacts} \
                 last_record={} maint_open_at_cut={}",
                view.max_ack,
                view.max_intent,
                view.last_record,
                u8::from(view.maint_open_at_cut)
            );
        }
        Err(f) => {
            eprintln!("crash_check: {}: {}", f.reason, f.detail);
            println!(
                "CRASH_CHECK verdict=FAIL reason={} backend={backend}",
                f.reason
            );
            std::process::exit(1);
        }
    }
}
