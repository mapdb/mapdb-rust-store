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
//! - the format v3 **segment namespace** must satisfy its own invariants, at
//!   the cut and again after recovery. The contents oracle above cannot see any
//!   of this: a recovery that reused a burnt segment name, resurrected a retired
//!   one, left create-crash residue behind, or failed to finish an unlink the
//!   crash interrupted produces exactly the right logical map and would have
//!   passed. The namespace is read INDEPENDENTLY (`ch::wal_namespace`, which
//!   re-derives names and headers from the format description rather than
//!   calling the store's own enumerator), once before the open — recovery
//!   mutates the file set, so the image is only observable then — and once
//!   after.
//! - the run must have EXERCISED that machinery: the journal's own namespace
//!   observations must show it rotating, retiring, and completing a whole-log
//!   clean that retired something. A round that never rotated cannot pass by
//!   satisfying namespace rules vacuously.
//! - the **recovery successor** must work: after the contents oracle, the
//!   checker commits one record on the recovered store, closes, and reopens. A
//!   bad `nextLsn`/`nextSeq` handoff is invisible in the store recovery just
//!   produced — it only bites whatever is written next, which in a crash round
//!   is nothing at all.
//!
//! Verdict: one stable stdout line
//! `CRASH_CHECK verdict=PASS|FAIL [reason=<code>] backend=wal recovered_txid=…
//! ack_txid=… intents=… entries=… ready_groups=… ready_checkpoints=…
//! ready_compactions=… last_record=… maint_open_at_cut=0|1 ns_segs_at_cut=…
//! ns_seq_lo=… ns_seq_hi=… ns_residue_at_cut=0|1 ns_gap_at_cut=…
//! ns_unlinked_by_recovery=… ns_created_by_recovery=0|1
//! ns_compactions_retiring=… ns_autoclean_events=…`; diagnostics on stderr;
//! exit 0 iff PASS. `maint_open_at_cut` reports an unmatched `M begin` at the
//! witness cutoff — NOT proof the cut landed inside compaction (the process may
//! die before the `M done` sync). `ns_residue_at_cut` and `ns_gap_at_cut` are
//! the rare-window counters: residue means a cut landed between a segment's
//! creation and its forced header, a gap means one landed inside an unlink run
//! or that a residue name was burnt. Neither can be required of a round, so
//! both are reported and aggregated across a campaign instead.

use mapdb_rust_store::db::DB;
use mapdb_rust_store::ser::bytearray::ByteArrayFormat;
use mapdb_rust_store::store::{Store, StoreWAL};
use mapdb_rust_store_crash_harness::wal_namespace::{self, Namespace};
use mapdb_rust_store_crash_harness::{
    self as ch, Config, Model, NsAt, Record, COMMITTED_SEQ_KEY, RUN_ID_KEY,
};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The key the successor phase writes AFTER recovery. Outside the generated
/// universe (universe keys start `k`) and distinct from both markers, so the
/// exact-equality oracle can account for it explicitly on the second open.
const SUCCESSOR_KEY: &[u8] = b"!crash-post-recovery";

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
    /// The last complete `N` observation: (lo, hi, count).
    ns_last: Option<(u64, u64, u64)>,
    /// What the run's own observations prove about the namespace machinery.
    ns: NsCoverage,
}

/// Namespace coverage, computed from the journal alone — deterministic, and so
/// a *requirement* rather than a cut-dependent counter. `ns_rotated` and
/// `ns_retired` are absolute facts about the run: a store's first segment is
/// seq 1, `hi` only moves when a segment is created, and `lo` only moves when
/// one is unlinked behind a forced `'K'`.
#[derive(Clone, Copy, Default, Debug)]
struct NsCoverage {
    /// A create beyond the store's first segment happened (rotate).
    rotated: bool,
    /// A segment was retired (forced `'K'` + `unlinkThrough`).
    retired: bool,
    /// Completed compactions whose bracketing pair shows `lo` advancing — the
    /// whole-log clean did retire the segments it rolled past.
    compactions_retiring: u64,
    /// Group boundaries where `lo` advanced with no compaction around them:
    /// the AUTOMATIC cleaning path, the one that runs inside `commit`.
    autoclean_events: u64,
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
            Record::Maint { .. } if max_intent != max_ack => {
                return Err(fail(
                    "group-maint-order",
                    format!(
                        "maintenance with an in-flight group (intent {max_intent} > ack {max_ack})"
                    ),
                ));
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

/// The namespace half of the ordered-record protocol. The shared grammar
/// already refuses an observation whose ends moved backwards; this reads the
/// same records for what they prove about the two operations that move them,
/// and holds each completed compaction to its bracketing pair.
fn namespace_coverage(records: &[Record]) -> Result<NsCoverage, Fail> {
    let mut cov = NsCoverage::default();
    let mut pre: Option<(u64, u64, u64)> = None; // the open compaction's `precompact`
    let mut prev_lo: Option<u64> = None; // the previous observation of ANY kind
    for rec in records {
        if let Record::Namespace { at, lo, hi, count } = rec {
            match at {
                NsAt::Group => {
                    // Only two things sit between the previous observation and
                    // this one: the group's applies, and its commit. Explicit
                    // compaction is accounted by its own bracketing pair, so a
                    // floor that advanced here advanced INSIDE `commit` — the
                    // automatic path.
                    if let Some(prev) = prev_lo {
                        if *lo > prev {
                            cov.autoclean_events += 1;
                        }
                    }
                    prev_lo = Some(*lo);
                }
                NsAt::PreCompact => {
                    pre = Some((*lo, *hi, *count));
                    prev_lo = Some(*lo);
                }
                NsAt::PostCompact => {
                    let Some((plo, phi, pcount)) = pre else {
                        return Err(fail(
                            "ns-compact-bracket",
                            "a postcompact observation with no precompact",
                        ));
                    };
                    // A whole-log clean rolls to a fresh segment and retires
                    // everything below it, so a multi-segment log MUST come out
                    // with a higher floor. (A single-segment log has nothing
                    // below the active one to retire — K4 keeps it.)
                    if pcount > 1 && *lo <= plo {
                        return Err(fail(
                            "ns-compact-no-retire",
                            format!(
                                "compaction over {pcount} segments [{plo}, {phi}] left the floor \
                                 at {lo}: a whole-log clean retires every segment below the one \
                                 it rolls to"
                            ),
                        ));
                    }
                    if *lo > plo {
                        cov.compactions_retiring += 1;
                    }
                    // What this compaction retired is credited to it, not to
                    // the automatic path at the next group boundary.
                    prev_lo = Some(*lo);
                    pre = None;
                }
            }
        }
    }
    if let Some((lo, hi, _)) = records.iter().rev().find_map(|r| match r {
        Record::Namespace { lo, hi, count, .. } => Some((*lo, *hi, *count)),
        _ => None,
    }) {
        // The store's first segment is seq 1 and neither end ever moves
        // backwards, so these two comparisons are the whole history.
        cov.rotated = hi > 1;
        cov.retired = lo > 1;
    }
    Ok(cov)
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
    let ns_cov = namespace_coverage(&records)?;
    let mut ns_last = None;
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
            Record::Namespace { .. } => "N",
            Record::Maint { .. } => "M",
            Record::Ready { .. } => "R",
        };
        match rec {
            Record::Header(c) => cfg = Some(c),
            Record::Intent { seq, digest, .. } => intents.push((seq, digest)),
            Record::PostApply { .. } => {}
            Record::Ack { txid } => max_ack = txid,
            Record::Namespace { lo, hi, count, .. } => ns_last = Some((lo, hi, count)),
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
        ns_last,
        ns: ns_cov,
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

/// Everything the verdict line reports about the namespace. The invariants are
/// asserted; these are the coverage numbers behind them.
#[derive(Clone, Copy, Default)]
struct NsVerdict {
    segs_at_cut: u64,
    lo_at_cut: u64,
    hi_at_cut: u64,
    /// Residue at the cut: a create that crashed before its header was forced
    /// (R2). Rare — the window is one `CREATE_NEW` plus one write wide.
    residue_at_cut: bool,
    /// A hole in the sequence numbers at the cut. Legitimate, and evidence
    /// that a cut landed inside an unlink run or that a residue name was burnt.
    gap_at_cut: u64,
    /// What RECOVERY did to the namespace: the K5/K8 replay of an unlink the
    /// crash interrupted, and the successor it rolled to (R7/N1).
    unlinked_by_recovery: u64,
    created_by_recovery: bool,
}

fn check(
    backend: &str,
    store: &Path,
    journal: &PathBuf,
    min_ack: u64,
) -> Result<(JournalView, u64, u64, NsVerdict), Fail> {
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

    // Coverage the run itself must have produced. Without these two the
    // namespace assertions below are vacuous — they would be holding a store
    // that never rotated and never retired to rules about rotating and
    // retiring — so they are a FAIL, not a counter.
    if !view.ns.rotated {
        return Err(fail(
            "ns-coverage",
            "the run never rotated: the highest segment sequence stayed 1, so create/rollover, \
             the forced 'K' and its unlink were never exercised",
        ));
    }
    if !view.ns.retired {
        return Err(fail(
            "ns-coverage",
            "the run never retired a segment: the lowest sequence stayed 1, so no forced 'K' ever \
             authorized an unlink",
        ));
    }
    if view.ns.compactions_retiring == 0 {
        return Err(fail(
            "ns-coverage",
            "no completed compaction retired anything: the whole-log clean's third phase \
             (forced 'K' then unlinkThrough) is unproven",
        ));
    }

    // The crash image, read BEFORE anything opens it — recovery mutates the
    // namespace (R2 deletes residue, R5 replays an unlink, R7 rotates), so this
    // is the only chance to see what the cut actually left behind.
    let pre = wal_namespace::scan(store).map_err(|e| fail("ns-scan", e))?;
    pre.check_image().map_err(|e| fail("ns-image", e))?;
    let mut nsv = NsVerdict {
        segs_at_cut: pre.count(),
        lo_at_cut: pre.lo() as u64,
        hi_at_cut: pre.hi() as u64,
        residue_at_cut: !pre.bad().is_empty(),
        gap_at_cut: pre.gaps(),
        ..NsVerdict::default()
    };
    // The image against the last thing the workload saw. Between that
    // observation and the cut the store may have created names above and
    // retired names below; it may never do the reverse, and a name it burnt is
    // gone for good.
    if let Some((lo, hi, _)) = view.ns_last {
        if nsv.lo_at_cut < lo || nsv.hi_at_cut < hi {
            return Err(fail(
                "ns-regressed-at-cut",
                format!(
                    "the crash image holds [{}, {}] but the workload last observed [{lo}, {hi}]: \
                     a retired name came back, or a burnt one was reused",
                    nsv.lo_at_cut, nsv.hi_at_cut
                ),
            ));
        }
    }

    let db = DB::<StoreWAL>::make_wal(store).map_err(|e| fail("open", format!("{e:?}")))?;
    let outcome = check_open(&db, &view, group, g_hi).and_then(|got| {
        // The namespace recovery left behind. Unlike the image half, nothing
        // here is excused by the cut point: recovery ran to completion, so
        // every partially applied namespace operation must now be finished.
        let post = wal_namespace::scan(store).map_err(|e| fail("ns-scan", e))?;
        post.check_recovered(&pre)
            .map_err(|e| fail("ns-recovered", e))?;
        nsv.unlinked_by_recovery = pre.count().saturating_sub(
            pre.segs
                .iter()
                .filter(|s| post.segs.iter().any(|p| p.seq == s.seq))
                .count() as u64,
        );
        nsv.created_by_recovery = post.count() + nsv.unlinked_by_recovery > pre.count();
        Ok((got, post))
    });
    // Always close (best-effort on the error path) so the segments are released.
    let ((recovered, entries), post) = match outcome {
        Ok(v) => {
            db.close().map_err(|e| fail("close", format!("{e:?}")))?;
            v
        }
        Err(f) => {
            let _ = db.close();
            return Err(f);
        }
    };

    // The recovery successor: a wrong `nextLsn` or `nextSeq` handoff is
    // invisible in the store recovery just produced — it only bites the next
    // thing written. So write one, and reopen.
    check_successor(store, &view, &post, recovered)?;
    Ok((view, recovered, entries, nsv))
}

/// Uses what recovery handed the writer: commits one record on the recovered
/// store, closes it, and reopens. A successor segment with a stale `firstLsn`,
/// a reused sequence number, or an `nextLsn` that collides with the recovered
/// log survives the first open and fails here — which is the point, since
/// nothing downstream of a crash round would ever have noticed.
fn check_successor(
    store: &Path,
    view: &JournalView,
    after_recovery: &Namespace,
    g: u64,
) -> Result<(), Fail> {
    let db =
        DB::<StoreWAL>::make_wal(store).map_err(|e| fail("successor-open", format!("{e:?}")))?;
    let outcome = (|| -> Result<(), Fail> {
        let map = db
            .tree_map("crash", ByteArrayFormat, ByteArrayFormat)
            .open()
            .map_err(|e| fail("successor-open-map", format!("{e:?}")))?;
        map.put(SUCCESSOR_KEY.to_vec(), g.to_string().into_bytes())
            .map_err(|e| fail("successor-put", format!("{e:?}")))?;
        db.commit()
            .map_err(|e| fail("successor-commit", format!("{e:?}")))?;
        Ok(())
    })();
    match outcome {
        Ok(()) => db
            .close()
            .map_err(|e| fail("successor-close", format!("{e:?}")))?,
        Err(f) => {
            let _ = db.close();
            return Err(f);
        }
    }

    // Reopen: the post-recovery commit must itself be durable, and the whole
    // recovered state must still be there underneath it.
    let ns = wal_namespace::scan(store).map_err(|e| fail("ns-scan", e))?;
    ns.check_recovered(after_recovery)
        .map_err(|e| fail("ns-successor", e))?;
    let db = DB::<StoreWAL>::make_wal(store).map_err(|e| fail("reopen", format!("{e:?}")))?;
    let outcome = (|| -> Result<(), Fail> {
        db.store()
            .verify()
            .map_err(|e| fail("reopen-verify", format!("{e:?}")))?;
        let map = db
            .tree_map("crash", ByteArrayFormat, ByteArrayFormat)
            .open()
            .map_err(|e| fail("reopen-map", format!("{e:?}")))?;
        let mut actual: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for (k, v) in map
            .entries()
            .map_err(|e| fail("reopen-read", format!("{e:?}")))?
        {
            actual.insert(k, v);
        }
        let model = replay_capture(view, &[g])?
            .remove(&g)
            .ok_or_else(|| fail("replay", format!("no model snapshot at boundary {g}")))?;
        let mut expected = expected_bytes(&model, &view.cfg, g);
        expected.insert(SUCCESSOR_KEY.to_vec(), g.to_string().into_bytes());
        if actual != expected {
            return Err(fail(
                "successor-state-mismatch",
                format!(
                    "after a post-recovery commit and reopen, contents != replay({g}) + the \
                     successor key (entries {}, expected {})",
                    actual.len(),
                    expected.len()
                ),
            ));
        }
        Ok(())
    })();
    match outcome {
        Ok(()) => db
            .close()
            .map_err(|e| fail("reopen-close", format!("{e:?}"))),
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
        Ok((view, recovered, entries, ns)) => {
            let (_, r_groups, r_ckpts, r_compacts) = view.ready.unwrap_or_default();
            println!(
                "CRASH_CHECK verdict=PASS backend={backend} recovered_txid={recovered} \
                 ack_txid={} intents={} entries={entries} ready_groups={r_groups} \
                 ready_checkpoints={r_ckpts} ready_compactions={r_compacts} \
                 last_record={} maint_open_at_cut={} ns_segs_at_cut={} ns_seq_lo={} \
                 ns_seq_hi={} ns_residue_at_cut={} ns_gap_at_cut={} \
                 ns_unlinked_by_recovery={} ns_created_by_recovery={} \
                 ns_compactions_retiring={} ns_autoclean_events={}",
                view.max_ack,
                view.max_intent,
                view.last_record,
                u8::from(view.maint_open_at_cut),
                ns.segs_at_cut,
                ns.lo_at_cut,
                ns.hi_at_cut,
                u8::from(ns.residue_at_cut),
                ns.gap_at_cut,
                ns.unlinked_by_recovery,
                u8::from(ns.created_by_recovery),
                view.ns.compactions_retiring,
                view.ns.autoclean_events,
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
