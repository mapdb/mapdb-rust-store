//! The schema-v2 **preflight corpus** against this engine — Stage C, slice
//! **C5r**.
//!
//! `tests/xfixtures-v2-corpus/` is a byte-identical copy of the `root`-marked
//! files of `todo/store-cross/preflight-v2/` — twelve files: `MANIFEST.tsv` and
//! one blob per `file` row, and nothing else (C5 plan §4c). It is the
//! `v2-oracle` profile: it carries `applies`, `action`, `bytes` and `reopen`
//! rows. The static `tests/xfixtures-v2/` sample stays `v2-core` and is
//! untouched by C5; [`xfixture_conformance`] still owns it, through the same
//! executor.
//!
//! # What this engine executes, and what it accounts for
//!
//! **The corpus addresses no `action`, `bytes` or `reopen` row to rust**, and
//! that is a property of the format rather than an oversight: `Cell.actions`,
//! `Cell.byte_assertions` and `Cell.reopen` exist on exactly one cell and are
//! keyed `("java", "rw")`. Revision 3 of the C5 plan claimed all three engines
//! execute them and both round-3 reviewers found the claim had an empty input
//! column for two engines. Manufacturing a port-addressed assertion in the
//! corpus to fix that would be fixture theatre.
//!
//! So plan §5.3 item 2 splits the flip: rust **accepts and accounts**, and its
//! execution paths get their inputs from SYNTHETIC manifests here — where a row
//! addressed to rust can be written by hand — run through the production path
//! over the corpus's real bytes. [`the_action_row_is_executed`],
//! [`the_bytes_row_is_graded`] and [`the_reopen_row_is_graded`] are those
//! inputs. An addressed row no handler consumed is a failure, not a no-op:
//! that accounting is the only thing that makes "executes" distinguishable
//! from "parses and drops".
//!
//! # rw here, ro in the crate
//!
//! `ro` cells need the crate-internal read-only opener and run in
//! `src/store/xfix_ro.rs` (decision C-D3). Each half grades the rows addressed
//! to its own mode; `MODES` has exactly two members and the parser refuses a
//! third, so between them the two halves cover every row addressed to rust.
//!
//! # The mutation campaign
//!
//! A set of NAMED cases in `scratchpad/mut_r.py` + `mutants_r.sh`. Each case
//! mutates one named site — deleting, replacing or moving it — or a named
//! combination, and the suite must then go red for the reason that case names.
//! The runner exits non-zero if any case survives, mis-kills or fails to apply,
//! so the count and the result are read from a run rather than asserted here.
//!
//! **It is a named campaign, not an exhaustive sweep.** What is true is
//! narrower: every check the campaign names has a red that names it. Most
//! checks are green when deleted unless something supplies an input, so they
//! are closed with DOCTORED manifests routed through the PRODUCTION path —
//! never by calling the check directly, which proves the method and leaves its
//! call unobserved. Where a check's red is unreachable from any conforming
//! corpus it gets a direct firing probe instead
//! ([`the_reopen_family_predicate_discriminates`],
//! [`the_read_only_write_probe_fires`] in the `ro` half).
//!
//! **The residue is the leaf problem**: a statement no other statement depends
//! on is invisible to deletion, and the last assertion in any chain is one. It
//! is pushed DOWN rather than eliminated, by collecting outcomes and comparing
//! them once per group — so one comparison per group is unobserved rather than
//! every statement in it.
//!
//! # What the campaign measured and could NOT kill
//!
//! Named, because a campaign that reports only its kills is a campaign whose
//! coverage claim nothing checks. Each of these was applied and the whole suite
//! stayed green:
//!
//! - `ran == want` in `run_v2_corpus_cells`. `applies == expect` fires first on
//!   every input that separates them, and `run_cells` runs one cell per
//!   `expect` row by construction, so the two can disagree only if the executor
//!   itself is broken. Deleting `applies == expect` IS killed — by this
//!   equality — so the pair guards each other in one direction only.
//! - the `ro_probed` comparison, the corpus-root file-set comparison, and the
//!   distribution-seal comparison. Each is the last statement in its group:
//!   they are what give the probe call, the root inventory and the copy its
//!   reds, and nothing observes THEM. This is the leaf pushed down, not
//!   removed.
//! - the `v2-core` profile assertion in `run_v2_cells`. The static sample
//!   carries no oracle row, so no input reaches it; it exists for the day one
//!   appears.
//! - `capture`'s "not a regular file" assertion. Not a leaf but SUBSUMED:
//!   deleting it leaves `read_named` failing on the same input with a worse
//!   message, so what the assertion buys is the diagnosis, not the refusal.
//!
//! Two more sites are killed only by the WEAKER signal "it failed for another
//! reason" — the doctored case's own check noticing that a different rule
//! fired. They are the "a wal3 cell with no post rows" guard, whose input also
//! trips the file-set rule one step later, and the two `run_action` refusals,
//! whose inputs also change the segment. The runner names the exact replacement
//! red so those cases stay falsifiable.

#[path = "../src/store/xfix.rs"]
mod xfix;

use mapdb_rust_store::store::StoreWAL;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xfix::red_of;

/// A scratch directory no other test in this binary can be handed.
///
/// `xfix::session_dir` keys on the tag and the PID and starts by removing what
/// it finds, so two tests that pass the same tag delete each other's cells
/// mid-run — cargo runs the cases in this file concurrently in ONE process.
/// The first version of this file shared one tag across nine cases and the
/// failures read as engine defects ("input x is gone and no post row says
/// so"), which is a good reminder that a harness bug wears the harness's own
/// vocabulary.
fn fresh_session(tag: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    xfix::session_dir(&format!("{tag}_{}", SEQ.fetch_add(1, Ordering::SeqCst)))
}

/// `freeze_v2.PREFLIGHT_DIST_SEALS["rust"]`, and the referent is
/// `todo/store-cross/preflight-v2/` — never this directory. A constant
/// regenerated from the tree it grades certifies that the tree equals itself.
///
/// Regenerate with `python3 todo/store-cross/freeze_v2.py --preflight
/// --dist-seals`.
pub const DIST_SEAL: &str = "6eb39e86807c4775f87096de296798ce24d2ae00e34e7187c3935a13582f8fec";

fn open_rw(base: &Path) -> mapdb_rust_store::error::Result<StoreWAL> {
    StoreWAL::open(base)
}

/// The corpus manifest with `edit` applied to its TEXT, re-parsed, over the
/// root's real blobs.
fn doctored(edit: impl Fn(&str) -> String) -> xfix::SampleV2 {
    let root = xfix::v2_corpus_root();
    let text = xfix::read_root_text(&root, "MANIFEST.tsv");
    let out = edit(&text);
    assert_ne!(
        out, text,
        "the doctoring changed nothing, so this case grades the same manifest twice"
    );
    xfix::load_sample_v2_text(&root, &out)
}

fn drop_rows(text: &str, prefix: &str) -> String {
    let kept: Vec<&str> = text
        .split('\n')
        .filter(|l| !l.starts_with(prefix))
        .collect();
    kept.join("\n")
}

/// Runs the whole rust `rw` suite over a doctored manifest and requires it to
/// refuse, by reason.
fn refuses_suite(what: &str, sample: &xfix::SampleV2, because: &str) {
    let session = fresh_session("xfcorpus_refuse");
    let msg = red_of(|| xfix::run_v2_corpus_cells(sample, "rw", &session, &open_rw));
    let _ = std::fs::remove_dir_all(&session);
    let msg = msg.unwrap_or_else(|| panic!("the suite accepted {what}"));
    assert!(
        msg.contains(because),
        "{what}: it failed for another reason: {msg}"
    );
}

/// Runs ONE doctored cell through the production path and requires a red whose
/// message names `because`.
fn refuses_cell(what: &str, sample: &xfix::SampleV2, fixture: &str, mode: &str, because: &str) {
    let msg = red_of(|| run_one(sample, fixture, mode));
    let msg = msg.unwrap_or_else(|| panic!("the cell accepted {what}"));
    assert!(
        msg.contains(because),
        "{what}: it failed for another reason: {msg}"
    );
}

fn run_one(sample: &xfix::SampleV2, fixture: &str, mode: &str) {
    let e = rust_cell(sample, fixture, mode);
    let session = fresh_session("xfcorpus_one");
    let cell = session.join("cell");
    std::fs::create_dir_all(&cell).unwrap();
    let mut cells = xfix::Cells::new(sample);
    cells.run_cell(&e, &cell, &open_rw, xfix::Dispatch::ByManifest);
    let _ = std::fs::remove_dir_all(&session);
}

fn rust_cell(sample: &xfix::SampleV2, fixture: &str, mode: &str) -> xfix::V2Expect {
    sample
        .manifest
        .expects
        .iter()
        .find(|e| e.fixture == fixture && e.engine == "rust" && e.mode == mode)
        .unwrap_or_else(|| panic!("no rust {mode} cell for {fixture}"))
        .clone()
}

// ---------------------------------------------------------------------------
// the cells
// ---------------------------------------------------------------------------

/// Every `applies` row addressed to rust in `rw`, run, and exactly those.
#[test]
fn corpus_rw_cells_conform() {
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let session = fresh_session("xfcorpus_rw");
    xfix::run_v2_corpus_cells(&sample, "rw", &session, &open_rw);
    let _ = std::fs::remove_dir_all(&session);
}

// ---------------------------------------------------------------------------
// the §3.11 mutant
// ---------------------------------------------------------------------------

/// Routing the `direct` cell through the WAL opener must turn this suite RED.
///
/// `reject-wal3-segment-at-direct` publishes a v3 segment as the bare file `x`
/// and expects `reject`/`direct`. Both openers refuse it, so the VERDICT
/// discriminates nothing here — which is why the plan required a mutant rather
/// than a deletion: restoring an `opener == "wal3"` refusal would prove parser
/// branching and nothing about this engine.
///
/// **What discriminates is the LOCK, and C5r measured both halves of §3.11's
/// mechanism rather than inheriting them** (java's flip found both false for
/// java). `StoreDirect::open_file` refuses the bare segment on its magic and
/// takes no `<base>.lock`; `StoreWAL::open` refuses it as D1 — a regular file
/// at the WAL base path — but takes the lock BEFORE that check. So a misrouted
/// cell leaves a stray `x.lock` that no `post` row names, and the two-sided
/// file-set rule is what goes red.
///
/// ONE comparison for all four facts this rests on: each opener's verdict and
/// each opener's file set. They are independent — a future rust whose WAL
/// opener accepted D1 would keep the file sets and lose the verdicts — but each
/// assertion written separately is a leaf the whole gate can lose without
/// noticing. Collapsing them leaves one leaf for the group instead of four.
#[test]
fn a_direct_cell_sent_to_the_wal_opener_goes_red() {
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let e = rust_cell(&sample, "reject-wal3-segment-at-direct", "rw");
    assert_eq!(e.opener, "direct", "the corpus's direct cell moved");
    let session = fresh_session("xfcorpus_mutant");

    let outcome = |dispatch: xfix::Dispatch, tag: &str| -> String {
        let cell = session.join(tag);
        std::fs::create_dir_all(&cell).unwrap();
        let mut cells = xfix::Cells::new(&sample);
        let e2 = e.clone();
        let cell2 = cell.clone();
        let red = red_of(move || cells.run_cell(&e2, &cell2, &open_rw, dispatch));
        let names: Vec<String> = dir_names(&cell).into_iter().collect();
        let verdict = match red {
            None => "PASSED".to_string(),
            Some(m) if m.contains("unexpected new file x.lock") => {
                "RED(unexpected new file x.lock)".to_string()
            }
            Some(m) => format!("RED({m})"),
        };
        format!("{verdict} {names:?}")
    };

    let control = outcome(xfix::Dispatch::ByManifest, "control");
    let misrouted = outcome(xfix::Dispatch::AlwaysWal3, "misrouted");
    assert_eq!(
        vec![control, misrouted],
        vec![
            "PASSED [\"x\"]".to_string(),
            "RED(unexpected new file x.lock) [\"x\", \"x.lock\"]".to_string(),
        ],
        "the direct cell through each opener: verdict, then what it left behind"
    );
    let _ = std::fs::remove_dir_all(&session);
}

fn dir_names(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// "accept and account": the three oracle rows, addressed to rust by hand
// ---------------------------------------------------------------------------

/// The synthetic Q8-shaped cell: `wal3-java-cleaned` rust `rw`, with its recid
/// rows stripped and an `action` row addressed to this engine.
///
/// The recid rows go because the commit REUSES a freed recid — `E` is the
/// corpus's `deleted` row and the allocator hands its slot straight back — so a
/// cell that both commits and asserts the pre-commit logical state is asserting
/// a contradiction. That is the same shape the real Q8 java cell has: an
/// `action` plus a `post` oracle and no recid rows, and it is why
/// `require_some_oracle` is a DISJUNCTION rather than "recid rows or nothing".
///
/// `extra` is appended after the action row so each case can add or withhold
/// exactly one more row.
fn action_cell(extra: &str) -> xfix::SampleV2 {
    let extra = extra.to_string();
    doctored(move |t| {
        format!(
            "{}{ACTION_ROW}{extra}",
            drop_rows(t, "recid\twal3-java-cleaned\t")
        )
    })
}

const ACTION_ROW: &str = "action\twal3-java-cleaned\trust\trw\tcommit_one_record\t\
                          op=put,payload_id=161,payload_len=64,recid_label=Q,serializer=raw\n";

fn action_post_row() -> String {
    format!(
        "post\twal3-java-cleaned\trust\trw\tx.wal.0000000000000004\tmodified:{ACTION_POST_LEN}:\
         {ACTION_POST_SHA}\n"
    )
}

/// An `action` row addressed to rust is EXECUTED, and its effect is graded.
///
/// The corpus addresses no action to this engine, so without a synthetic input
/// `run_action` would be code no test ever calls — lesson (i), a rule can be
/// correct, directly tested, and never called.
///
/// Three directions, because a post row that is right either way grades
/// nothing:
///
/// - with the action row AND the matching `post` row, the cell passes;
/// - with the `post` row and no action, nothing appends and the hash is wrong;
/// - with the action and no `post` row, the segment changed and the two-sided
///   unnamed-input rule says so.
///
/// The expected length and hash are PINNED, measured once from a run. That is
/// the standing the corpus's own `modified:279:…` row has; computing the
/// expectation from the run would compare the result to itself.
#[test]
fn the_action_row_is_executed() {
    run_one(&action_cell(&action_post_row()), "wal3-java-cleaned", "rw");

    // The post row WITHOUT the action. The recid rows stay here — with no
    // commit the corpus's logical state is still true, and stripping them
    // would leave an accept cell with no oracle, so `require_some_oracle`
    // would fire first and this case would report a red it did not produce
    // (lesson h).
    refuses_cell(
        "a post row for an append that never happened",
        &doctored(|t| format!("{t}{}", action_post_row())),
        "wal3-java-cleaned",
        "rw",
        "post[x.wal.0000000000000004 modified]: length",
    );

    // …and the CONTENT half of the same row, which the case above cannot reach
    // because the length disagrees first. Same length, one hex digit changed.
    refuses_cell(
        "a post row whose length is right and whose hash is not",
        &action_cell(&action_post_row().replace(ACTION_POST_SHA, ACTION_POST_SHA_WRONG)),
        "wal3-java-cleaned",
        "rw",
        "post[x.wal.0000000000000004 modified]: SHA-256",
    );

    refuses_cell(
        "an action whose effect no post row names",
        &action_cell(""),
        "wal3-java-cleaned",
        "rw",
        "input x.wal.0000000000000004 changed and no post row says so",
    );

    // The verb and its arguments are refused when this engine cannot honour
    // them — contract §2.3's "skipping it is forbidden". Both refusals come
    // from `run_action`, so a row it cannot execute stops the cell instead of
    // authoring a post state for behaviour that did not run.
    refuses_cell(
        "an action verb this engine does not implement",
        &doctored(|t| {
            format!(
                "{}action\twal3-java-cleaned\trust\trw\tcompact\t\
                 op=put,payload_id=161,payload_len=64,recid_label=Q,serializer=raw\n",
                drop_rows(t, "recid\twal3-java-cleaned\t")
            )
        }),
        "wal3-java-cleaned",
        "rw",
        "unknown action verb",
    );
    refuses_cell(
        "an action argument this engine does not implement",
        &doctored(|t| {
            format!(
                "{}action\twal3-java-cleaned\trust\trw\tcommit_one_record\t\
                 op=delete,payload_id=161,payload_len=64,recid_label=Q,serializer=raw\n",
                drop_rows(t, "recid\twal3-java-cleaned\t")
            )
        }),
        "wal3-java-cleaned",
        "rw",
        "unimplemented op",
    );
}

/// The pinned post state of `wal3-java-cleaned`'s active segment after
/// `commit_one_record(op=put, payload_id=161, payload_len=64, serializer=raw)`.
const ACTION_POST_LEN: usize = 279;
const ACTION_POST_SHA: &str = "b3286aee528406b137f8df56fd435e647617cb1ea9ff6e6ea403e16400a4ee4b";
/// The same digest with its last digit changed: the input the CONTENT half of
/// the `modified` verb needs, since a length disagreement fires first.
const ACTION_POST_SHA_WRONG: &str =
    "b3286aee528406b137f8df56fd435e647617cb1ea9ff6e6ea403e16400a4ee4c";

/// A `bytes` row addressed to rust is graded, VALUE and all.
///
/// The whole-file `post` hash covers the same bytes, so a handler that read the
/// range and compared it to nothing would pass every conforming input — codex
/// deleted exactly that equality in java and watched the gate stay green. The
/// doctored value has the same LENGTH and different content, so the post hash
/// cannot fire and only the equality can.
#[test]
fn the_bytes_row_is_graded() {
    let row = |offset: usize, hex: &str| {
        format!(
            "{}bytes\twal3-java-cleaned\trust\trw\tx.wal.0000000000000004\t{offset}\t{hex}\n",
            action_post_row()
        )
    };

    // TRUE: the bytes the action wrote, at the offset it wrote them.
    run_one(
        &action_cell(&row(ACTION_BYTES_OFFSET, ACTION_BYTES_HEX)),
        "wal3-java-cleaned",
        "rw",
    );
    // FALSE, at the same offset and the same length, so the whole-file post
    // hash cannot fire and only the equality can. Codex deleted exactly that
    // equality in java and watched the gate stay green.
    refuses_cell(
        "a bytes row whose value is wrong",
        &action_cell(&row(ACTION_BYTES_OFFSET, ACTION_BYTES_WRONG)),
        "wal3-java-cleaned",
        "rw",
        "the asserted bytes",
    );
    // Out of range: an assertion whose range cannot be reached is a failure,
    // never a skip.
    refuses_cell(
        "a bytes row past the end of the post state",
        &action_cell(&row(100_000, "aabb")),
        "wal3-java-cleaned",
        "rw",
        "the range ends at",
    );
}

/// The appended section header's `lsn` field — 8 bytes big-endian at header
/// offset 1 — in the post state the action above produces. The input segment is
/// 186 bytes, so this range exists only AFTER the commit: a `bytes` row is an
/// assertion against the captured post bytes and never a pre-open patch.
const ACTION_BYTES_OFFSET: usize = 187;
const ACTION_BYTES_HEX: &str = "000000000000000b";
const ACTION_BYTES_WRONG: &str = "000000000000000c";

/// A `reopen` row addressed to rust is run, and its FAMILY is graded.
///
/// `div-wal3-lsn-exhausted` is the cell whose image this engine refuses, and a
/// reopen of the same directory refuses the same way, so the row has a real
/// input. The family is `StoreFull` — contract §10.1 pins it, *"rust, zig, rw
/// and ro: reject, error family StoreFull exactly — not the corruption
/// family"* — which makes this the one place in this engine where a `reject`
/// cell's family is graded at all. The `expect` row has no column for it (see
/// `run_cell`'s reject arm).
///
/// Three inputs, because a family predicate with one member cannot be shown to
/// READ the row:
///
/// - `StoreFull` — the true family; the cell passes;
/// - `S2` — a family this engine DOES implement, whose predicate this refusal
///   fails. Only a family actually read from the row can tell this manifest
///   from the one above;
/// - `R4-floor` — a family this engine has no predicate for, which must be a
///   failure rather than "it threw something".
#[test]
fn the_reopen_row_is_graded() {
    let row = |family: &str| format!("reopen\tdiv-wal3-lsn-exhausted\trust\trw\t{family}\n");
    run_one(
        &doctored(|t| format!("{t}{}", row("StoreFull"))),
        "div-wal3-lsn-exhausted",
        "rw",
    );
    refuses_cell(
        "a reopen family this engine implements but this refusal is not",
        &doctored(|t| format!("{t}{}", row("S2"))),
        "div-wal3-lsn-exhausted",
        "rw",
        "not the S2 rule's refusal",
    );
    refuses_cell(
        "a reopen family this engine has no predicate for",
        &doctored(|t| format!("{t}{}", row("R4-floor"))),
        "div-wal3-lsn-exhausted",
        "rw",
        "has no predicate in this engine",
    );
}

/// The `reopen` family predicate must DISCRIMINATE, which no corpus can show.
///
/// A predicate that accepted any corruption at all would pass every cell —
/// lesson (g), a comparison can only see the variation its inputs contain. S9
/// is the immediate neighbour: the very next branch of the same scan, the same
/// error variant, a different rule. It must not match.
#[test]
fn the_reopen_family_predicate_discriminates() {
    use mapdb_rust_store::error::DbError;
    xfix::assert_family(
        "S2 control",
        "S2",
        &DbError::corrupt(
            "WAL segment x.wal.4: section LSN -9223372036854775808 at offset 187 does not \
             follow 9223372036854775807",
        ),
    );

    let refused = |what: &str, family: &str, e: DbError| {
        let e = std::panic::AssertUnwindSafe(e);
        let msg = red_of(move || xfix::assert_family("probe", family, &e));
        assert!(msg.is_some(), "the family predicate accepted {what}");
    };
    refused(
        "S9's refusal, which is the next branch of the same scan",
        "S2",
        DbError::corrupt(
            "WAL segment x.wal.4: section LSNs must be consecutive: 12 at offset 187 after 9",
        ),
    );
    refused(
        "a corruption verdict from another rule entirely",
        "S2",
        DbError::corrupt("WAL file x.wal is not a v3 segment"),
    );
    // The WHOLE S2 message on a non-corruption variant. In java this input
    // isolates the CLASS predicate; here it does not, and saying so is the
    // point: `Display for DbError` gives every non-corruption variant a prefix
    // of its own, so the rendered error is `verify failed: WAL segment …` and
    // the MESSAGE predicate refuses it. There is no rust input that reaches a
    // separate variant check, which is why `assert_family` states the S2 arm
    // as one claim rather than two (lesson h, from the inside).
    refused(
        "an operational failure wearing the right words",
        "S2",
        DbError::VerifyFailed(
            "WAL segment x.wal.4: section LSN 1 at offset 2 does not follow 0".to_string(),
        ),
    );
    refused(
        "a family this engine has no predicate for",
        "R4-floor",
        DbError::corrupt("anything at all"),
    );
    // The second implemented family, both ways round: it must accept its own
    // verdict and refuse the neighbouring one. Without the pair, "the family is
    // read from the row" and "every family means DataCorruption" would be
    // indistinguishable — the corpus varies in nothing here (lesson g).
    xfix::assert_family("StoreFull control", "StoreFull", &DbError::StoreFull);
    refused(
        "an S2 corruption verdict presented as StoreFull",
        "StoreFull",
        DbError::corrupt_msg(
            "WAL segment x.wal.4: section LSN 1 at offset 2 does not follow 0".to_string(),
        ),
    );
    refused(
        "a StoreFull verdict presented as S2",
        "S2",
        DbError::StoreFull,
    );
    // The pattern matches WHOLE. An unanchored substring test would accept the
    // S2 wording embedded in unrelated text.
    refused(
        "the S2 wording embedded in an unrelated message",
        "S2",
        DbError::corrupt(
            "prefix: WAL segment x: section LSN 1 at offset 2 does not follow 0; suffix",
        ),
    );
}

// ---------------------------------------------------------------------------
// the rules with no natural input
// ---------------------------------------------------------------------------

/// An oracle row addressed to a cell whose arm has no handler for it must FAIL
/// the cell.
///
/// The grammar permits an `action` row on a `reject` cell; the catalogue never
/// emits one, and the reject arm never opens a store to run it against. Without
/// the accountant that row is parsed, addressed, and silently dropped — which is
/// precisely "parses" wearing "executes"'s clothes.
#[test]
fn an_oracle_row_no_arm_can_run_fails_the_cell() {
    refuses_cell(
        "an action row on a reject cell",
        &doctored(|t| {
            format!(
                "{t}action\treject-wal3-d1-barebase\trust\trw\tcommit_one_record\t\
                 op=put,payload_id=1,payload_len=1,recid_label=Z,serializer=raw\n"
            )
        }),
        "reject-wal3-d1-barebase",
        "rw",
        "no handler consumed",
    );
}

/// An oracle row addressed to a cell this engine never runs must fail the
/// SUITE — the half per-cell accounting is structurally blind to.
///
/// The accountant is built from the rows addressed to the cell BEING RUN, so a
/// row addressed to a `(fixture, mode)` with no `expect` row is owed by nobody,
/// consumed by nobody and graded by nobody. All four addressed row types get
/// their own doctored input, because the check is one loop per type and C5j's
/// round 2 deleted two of them with the gate green when only the `bytes` shape
/// had a red.
///
/// Each case ADDS an orphan row rather than moving a real one: moving a row
/// away changes what the surviving cells assert and some earlier check fires
/// first, which would report KILLED while proving a different rule (lesson h).
#[test]
fn an_oracle_row_addressed_to_an_absent_cell_fails_the_suite() {
    // A real fixture with no rust `rw` cell at all: the direct fixture has only
    // an `rw` cell, so its `ro` mode is absent — and `reject-wal3-segment-at-
    // direct` is the one fixture whose rust cell set is partial in `rw`'s
    // favour. For `rw` orphans the absent cell is that fixture's `ro`… which
    // this half does not grade. So the orphan must be an `rw` row on a fixture
    // rust has no `rw` cell for, and the corpus has none — every fixture has a
    // rust `rw` cell. The doctoring therefore REMOVES a cell coherently and
    // leaves its oracle row behind, which is the same shape from the other end.
    let absent = "reject-wal3-segment-at-direct";
    let strip = |t: &str| {
        let mut out = t.to_string();
        for pfx in [
            &format!("applies\t{absent}\trust\trw"),
            &format!("expect\t{absent}\trust\trw\t"),
        ] {
            out = drop_rows(&out, pfx);
        }
        out
    };
    refuses_suite(
        "a bytes row addressed to a cell rust never runs",
        &doctored(|t| format!("{}bytes\t{absent}\trust\trw\tx\t0\tab\n", strip(t))),
        &format!("bytes {absent}/rw"),
    );
    refuses_suite(
        "a reopen row addressed to a cell rust never runs",
        &doctored(|t| format!("{}reopen\t{absent}\trust\trw\tS2\n", strip(t))),
        &format!("reopen {absent}/rw"),
    );
    refuses_suite(
        "an action row addressed to a cell rust never runs",
        &doctored(|t| {
            format!(
                "{}action\t{absent}\trust\trw\tcommit_one_record\t\
                 op=put,payload_id=1,payload_len=1,recid_label=Z,serializer=raw\n",
                strip(t)
            )
        }),
        &format!("action {absent}/rw"),
    );
    // …and `post`, the fourth addressed row type — named by contract §2.3 and
    // droppable in silence on both sides of the fence before it was.
    refuses_suite(
        "a post row addressed to a cell rust never runs",
        &doctored(|t| format!("{}post\t{absent}\trust\trw\tz.lock\tunchanged\n", strip(t))),
        &format!("post {absent}/rw"),
    );
}

/// A file the engine creates that no `post` row names must fail the cell.
///
/// The two-sided file-set rule is what makes the post oracle more than "the
/// named files are right"; a one-sided reading would pass a store that rewrote
/// or littered everything it was not asked about. Deleting the `x.lock` row is
/// how that side gets an input: the lock is still created, and now nothing
/// accounts for it.
///
/// It runs on the synthetic action cell because that is the only rust cell with
/// TWO post rows. Every corpus cell of this engine carries exactly one — the
/// universal `x.lock` — so dropping it would leave the cell with none and trip
/// the "a wal3 cell with no post rows is not a check" guard instead. An input
/// that trips several checks measures the first one only (lesson h), and this
/// case is about the second.
#[test]
fn a_file_no_post_row_names_fails_the_cell() {
    let with_lock_dropped = doctored(|t| {
        format!(
            "{}{ACTION_ROW}{}",
            drop_rows(
                &drop_rows(t, "recid\twal3-java-cleaned\t"),
                "post\twal3-java-cleaned\trust\trw\tx.lock\t"
            ),
            action_post_row()
        )
    });
    refuses_cell(
        "a cell that created x.lock with no row naming it",
        &with_lock_dropped,
        "wal3-java-cleaned",
        "rw",
        "unexpected new file x.lock",
    );
}

/// The reader contract is NON-VACUOUS on this corpus.
///
/// Deleting `assert_reader_contract` leaves the suite green, and it always
/// will: nothing observes the last assertion in a chain, so a deletion mutant
/// is the wrong instrument. What can be shown is that the assertion FIRES —
/// that `wal3-java-cleaned`'s six recid rows are compared against a real read
/// rather than carried past it. Record A holds `payload(116, 120)`; this says
/// 117 and the cell must refuse.
#[test]
fn the_reader_contract_is_not_vacuous() {
    refuses_cell(
        "a recid row whose payload id is wrong",
        &doctored(|t| {
            t.replace(
                "recid\twal3-java-cleaned\tA\t1\tlive\t116\t120",
                "recid\twal3-java-cleaned\tA\t1\tlive\t117\t120",
            )
        }),
        "wal3-java-cleaned",
        "rw",
        "recid 1",
    );
}

/// An accept cell that asserts nothing must be refused — the C3j guard, as the
/// disjunction plan §5.3 item 5 asked for.
///
/// C5j's first draft deleted this guard for the sealed root and offered the
/// distribution seal as the authority. Both reviewers refused, and this is the
/// input that proved them right: strip `wal3-java-cleaned`'s six recid rows and
/// the cell passes on nothing but the universal `x.lock` post row. **The seal
/// proves copy fidelity; assertion adequacy is a different property and
/// artifact identity cannot buy it.**
#[test]
fn an_accept_cell_that_asserts_nothing_is_refused() {
    refuses_cell(
        "a writable accept cell with no oracle at all",
        &doctored(|t| drop_rows(t, "recid\twal3-java-cleaned\t")),
        "wal3-java-cleaned",
        "rw",
        "asserts nothing about the store it opened",
    );
    // …and the disjunction is not vacuous in the other direction: the SAME
    // stripped fixture would pass in `ro`, where the read-only write refusal is
    // the claim. That half runs in `src/store/xfix_ro.rs`, which owns the
    // read-only opener; without the pair, "an accept cell must assert
    // something" and "ro is exempt" would be indistinguishable.
}

/// A wal3 cell with no `post` rows asserts nothing about the directory it
/// opened — and the guard keys on the MANIFEST's opener, not the dispatched
/// one.
///
/// Plan §5.3 item 5's second relaxation is the one this engine needs and java
/// did not: rust's `StoreDirect` takes no `<base>.lock`, so the direct cell
/// legitimately leaves the directory as it found it and carries no post row.
/// The relaxation is therefore conditioned on the opener rather than deleted,
/// and this case supplies the input that shows the condition is not a blanket
/// exemption: strip a WAL cell's only post row and it must still refuse.
#[test]
fn a_wal3_cell_with_no_post_rows_is_refused() {
    refuses_cell(
        "a wal3 cell with every post row removed",
        &doctored(|t| drop_rows(t, "post\treject-wal3-d1-barebase\trust\trw\t")),
        "reject-wal3-d1-barebase",
        "rw",
        "asserts nothing about the directory it just opened",
    );
}

/// An `applies` row that goes missing while its `expect` row stays must fail.
///
/// The two row types come from one catalogue and agree by construction, so only
/// a doctored manifest can separate them. Both set equalities in
/// `run_v2_corpus_cells` are plan §5.3 item 6, and neither has an input
/// otherwise.
#[test]
fn an_applies_row_missing_its_expect_fails() {
    let sample = doctored(|t| drop_rows(t, "applies\twal3-java-cleaned\trust\trw"));
    assert!(
        sample
            .manifest
            .expects
            .iter()
            .any(|e| e.fixture == "wal3-java-cleaned" && e.engine == "rust" && e.mode == "rw"),
        "the expect row must survive, or this proves only that a row was deleted"
    );
    refuses_suite(
        "an applies row deleted while its expect row stays",
        &sample,
        "are different sets",
    );
}

/// A `post` row addressed to a cell that RUNS is consumed and graded — the
/// `unchanged` verb's input.
///
/// A handler that skipped an `unchanged` row would be masked: the two-sided
/// unnamed-input rule independently re-verifies the same file and reports
/// green. That is why `post` needs the per-cell debt as well as the suite-wide
/// addressing check.
///
/// Both directions, because a verb that always holds is not a check: the same
/// `unchanged` row is true of a segment this cell leaves alone and false of one
/// the action grows.
#[test]
fn an_unchanged_post_row_is_graded() {
    let session = fresh_session("xfcorpus_unchanged");
    let sample = doctored(|t| {
        format!("{t}post\twal3-java-cleaned\trust\trw\tx.wal.0000000000000002\tunchanged\n")
    });
    xfix::run_v2_corpus_cells(&sample, "rw", &session, &open_rw);
    let _ = std::fs::remove_dir_all(&session);

    refuses_cell(
        "an `unchanged` row over the segment the action grew",
        &action_cell("post\twal3-java-cleaned\trust\trw\tx.wal.0000000000000004\tunchanged\n"),
        "wal3-java-cleaned",
        "rw",
        "bytes changed",
    );
}

// ---------------------------------------------------------------------------
// the accountant
// ---------------------------------------------------------------------------

/// The consumption accountant, unit-tested.
///
/// It is the only thing that makes "executes" distinguishable from "parses and
/// drops" for three of the four addressed row types — every one except
/// `action`, which has a failure of its own.
#[test]
fn an_unconsumed_oracle_row_is_a_failure() {
    let (a, b) = (1u8, 2u8);

    let mut ok = xfix::Consumption::new("ctx");
    ok.owe("action x", &a);
    ok.owe("reopen S2", &b);
    ok.consume("action x", &a);
    ok.consume("reopen S2", &b);
    ok.require_all_consumed();

    let mut dropped = xfix::Consumption::new("ctx");
    dropped.owe("action x", &a);
    dropped.owe("reopen S2", &b);
    dropped.consume("action x", &a);
    assert!(
        red_of(move || dropped.require_all_consumed()).is_some(),
        "accepted a row no handler consumed"
    );

    let mut twice = xfix::Consumption::new("ctx");
    twice.owe("action x", &a);
    twice.consume("action x", &a);
    assert!(
        red_of(move || twice.consume("action x", &a)).is_some(),
        "accepted the same row consumed twice"
    );

    let mut never = xfix::Consumption::new("ctx");
    assert!(
        red_of(move || never.consume("action x", &a)).is_some(),
        "accepted a row consumed that was never owed"
    );

    let mut other = xfix::Consumption::new("ctx");
    other.owe("action x", &a);
    assert!(
        red_of(move || other.consume("action x", &b)).is_some(),
        "accepted the key consumed with a different row"
    );
}

// ---------------------------------------------------------------------------
// the root itself
// ---------------------------------------------------------------------------

/// The corpus root holds `MANIFEST.tsv` plus one blob per `file` row and
/// nothing else (C5 plan §4c).
///
/// No golden tables and no post blobs: `GOLDEN-DECODE.tsv` and `GOLDEN-BODY.tsv`
/// belong to the static sample and stay there, and post-state blobs are not
/// distributed to any engine.
#[test]
fn the_corpus_root_has_nothing_unexplained() {
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let mut expected: BTreeSet<String> = BTreeSet::new();
    expected.insert("MANIFEST.tsv".to_string());
    for f in &sample.manifest.files {
        expected.insert(f.blob_name());
    }
    assert_eq!(
        expected,
        dir_names(&xfix::v2_corpus_root()),
        "the corpus root holds files no `file` row accounts for (or is missing one)"
    );
}

/// This root is byte-identical to todo's sealed tree.
///
/// Re-derives `freeze_v2.dist_seal(files, "rust")` — todo's own preimage
/// grammar, restricted to the `root`-marked files an engine actually receives —
/// from what is on disk, and compares it to the constant todo's gate re-derives
/// from `FROZEN.tsv` on every run. Neither side can move without the other
/// going red, and the comparison is in CI rather than in a review note.
///
/// The file SET comes from the directory listing rather than from
/// `MANIFEST.tsv`, so this and [`the_corpus_root_has_nothing_unexplained`] do
/// not consult the same source: a blob added to the tree moves the seal even if
/// no row mentions it.
///
/// What it does not certify, stated because the whole-artifact seal does
/// certify it: provenance. The four repo commits and `sync_v2.py`'s digest are
/// in `PREFLIGHT_SEAL`'s preimage and not in this one — they are not properties
/// of the distributed bytes and this repository has no way to check them.
#[test]
fn the_corpus_root_matches_todos_sealed_tree() {
    let dir = xfix::v2_corpus_root();
    let names = dir_names(&dir);
    assert!(!names.is_empty(), "the corpus root is empty");
    let mut pre = String::from("mapdb-xfixtures-dist\tv1\nengine\trust\n");
    for n in &names {
        let b = std::fs::read(dir.join(n)).unwrap();
        pre.push_str(&format!(
            "file\t{n}\t{}\t{}\troot\n",
            b.len(),
            xfix::sha256_hex(&b)
        ));
    }
    assert_eq!(
        DIST_SEAL,
        xfix::sha256_hex(pre.as_bytes()),
        "this root is not todo/store-cross/preflight-v2/'s `root` slice. Regenerate with \
         `freeze_v2.py --preflight --dist-seals`, and copy the TREE too — a constant updated \
         alone certifies whatever is here"
    );
}
