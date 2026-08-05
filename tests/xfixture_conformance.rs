//! Cross-port conformance harness — **both manifest schemas**, dispatched on
//! the version line (Stage C slice **C3r**).
//!
//! `tests/xfixtures/` is the live schema-v1 tree; `tests/xfixtures-v2/` is the
//! shared static schema-v2 sample (`todo/store-cross/sample-v2/`, byte for
//! byte). Keeping both roots in the suite at once is deliberate: C6 is a data
//! commit, and a reader that only ever saw the schema it was written for would
//! discover the other one on cutover day.
//!
//! What runs here, and what does not:
//!
//! - **v1 cells** — accept/reject, `direct` and `wal` openers, as before.
//! - **v2 `rw` cells** — through the public [`StoreWAL::open`].
//! - **v2 `ro` cells** — NOT here. They need the crate-internal read-only
//!   opener and run in `src/store/xfix_ro.rs` (decision C-D3). Deleting a
//!   `rw` expect row fails the set check below; deleting an `ro` one fails the
//!   set check there.
//! - **the two §11.2 comparisons** — framing against `GOLDEN-DECODE.tsv`,
//!   decoded bodies against `GOLDEN-BODY.tsv`, which the frozen Java reader
//!   authored.

#[path = "../src/store/xfix.rs"]
mod xfix;

use mapdb_rust_store::error::DbError;
use mapdb_rust_store::store::{Store, StoreDirect, StoreWAL};
use std::collections::BTreeSet;
use std::path::Path;

// ---------------------------------------------------------------------------
// schema v1 — the live tree
// ---------------------------------------------------------------------------

/// The `wal-v1-*` accept cells RETIRE at this engine's WAL v3 cutover: the port
/// refuses format v1 outright (there is no migration, by design) and its opener
/// no longer takes a WAL FILE path at all, so the cell cannot even be
/// expressed. D6 retires these IDs family-wide at Stage C; until the java and
/// zig generators stop emitting v1 rows they stay in the shared manifest for
/// the engines that still speak v1, and this engine skips them. The list is
/// EXACT and asserted below: a new accept row addressed to rust must not be
/// silently dropped by a prefix match.
const RETIRED_V1_ACCEPTS: [&str; 4] = [
    "wal-v1-rust-tail",
    "wal-v1-rust-ckpt",
    "wal-v1-zig-tail",
    "wal-v1-zig-ckpt",
];

#[test]
fn v1_cells_pass() {
    let root = xfix::v1_root();
    let loaded = xfix::parse(&xfix::read_root_text(&root, "MANIFEST.tsv"));
    assert_eq!(loaded.version(), 1, "the live tree must still be schema v1");
    let m = loaded.v1();

    let session = xfix::session_dir("xfix_v1");

    // gunzip every fixture file ONCE, verifying gz sha, raw length and raw sha
    // before any cell runs.
    let mut baselines = std::collections::BTreeMap::new();
    for f in &m.files {
        let gz = xfix::read_root_file(&root, &format!("{}.gz", f.rel));
        assert_eq!(xfix::sha256_hex(&gz), f.gz_sha, "gzSha256 for {}", f.rel);
        let raw = xfix::gunzip(&gz, &f.rel);
        assert_eq!(raw.len() as u64, f.raw_len, "rawLen for {}", f.rel);
        assert_eq!(xfix::sha256_hex(&raw), f.raw_sha, "rawSha256 for {}", f.rel);
        baselines.insert((f.fixture.clone(), f.rel.clone()), raw);
    }

    let mut retired = Vec::new();
    let mut ran = 0usize;
    for (i, e) in m.expects.iter().enumerate() {
        if e.engine != xfix::ENGINE {
            continue;
        }
        if e.verdict == "accept" && e.opener == "wal" {
            assert!(
                RETIRED_V1_ACCEPTS.contains(&e.fixture.as_str()),
                "cell {i}: accept-wal fixture {} is not one of the four v1 cells retired at the \
                 v3 cutover — a new WAL accept row needs a v3 (base-path) cell, not a skip",
                e.fixture
            );
            retired.push(e.fixture.clone());
            continue;
        }
        ran += 1;
        let ctx = format!(
            "v1 cell {i}: fixture={} verdict={} opener={} placeAs={} openArg={}",
            e.fixture, e.verdict, e.opener, e.place_as, e.open_arg
        );
        let files = m.files_of(&e.fixture);
        assert_eq!(files.len(), 1, "[{ctx}] a v1 fixture has exactly one file");
        let baseline = &baselines[&(e.fixture.clone(), files[0].rel.clone())];

        let cell = session.join(format!("v1-{i}"));
        std::fs::create_dir_all(&cell).unwrap();
        let working = cell.join(&e.place_as);
        std::fs::write(&working, baseline).unwrap();
        let before = dir_entries(&cell);
        let target = cell.join(&e.open_arg);

        let recids = m.recids_of(&e.fixture);
        match (e.verdict.as_str(), e.opener.as_str()) {
            ("accept", "direct") => {
                let s = StoreDirect::open_file(&target)
                    .unwrap_or_else(|err| panic!("[{ctx}] accept cell failed to open: {err}"));
                xfix::assert_reader_contract(&s, &recids, &ctx);
                s.close().unwrap();
            }
            ("reject", "direct") => match StoreDirect::open_file(&target) {
                Err(DbError::DataCorruption(_)) => {}
                Err(other) => panic!("[{ctx}] expected DataCorruption, got: {other}"),
                Ok(s) => {
                    let _ = s.close();
                    panic!("[{ctx}] reject cell opened successfully");
                }
            },
            ("reject", "wal") => {
                // The v3 opener takes a BASE path. Every v1 reject row's
                // openArg names a regular file the cell placed there, so each
                // now refuses through D1's bare-base row rather than through a
                // v1 header check — the same verdict for the same image, which
                // is what the cell asserts.
                match StoreWAL::open(&target) {
                    Err(DbError::DataCorruption(_)) => {}
                    Err(other) => panic!("[{ctx}] expected DataCorruption, got: {other}"),
                    Ok(s) => {
                        let _ = s.close();
                        panic!("[{ctx}] reject cell opened successfully");
                    }
                }
            }
            (v, o) => panic!("[{ctx}] unsupported cell verdict={v} opener={o}"),
        }

        assert_eq!(
            std::fs::read(&working).unwrap(),
            *baseline,
            "[{ctx}] working copy bytes changed"
        );
        let new_files: Vec<String> = dir_entries(&cell).difference(&before).cloned().collect();
        for f in &new_files {
            assert!(
                f.ends_with(".lock"),
                "[{ctx}] unexpected new file in cell dir: {f} (all new: {new_files:?})"
            );
        }
        std::fs::remove_dir_all(&cell).unwrap();
    }
    assert!(ran > 0, "the v1 manifest has no expect rows for rust");
    retired.sort();
    let mut expected: Vec<String> = RETIRED_V1_ACCEPTS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    expected.sort();
    assert_eq!(
        retired, expected,
        "every retired v1 accept cell must still be present in the shared manifest: if one is \
         gone, Stage C has begun and this skip list must go with it"
    );
    let _ = std::fs::remove_dir_all(&session);
}

fn dir_entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("read cell dir")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// schema v2 — the shared static sample
// ---------------------------------------------------------------------------

#[test]
fn sample_v2_rw_cells_pass() {
    let sample = xfix::load_sample_v2(&xfix::v2_root());
    let session = xfix::session_dir("xfix_v2_rw");
    xfix::run_v2_cells(&sample, "rw", &session, &StoreWAL::open);
    let _ = std::fs::remove_dir_all(&session);
}

/// FRAMING, against the python-authored pin.
///
/// This is the comparison `GOLDEN.tsv` cannot make: a raw sha attests which
/// bytes were read and says nothing about the parse. It is also the only check
/// in the slice that reaches the section COUNT — both section CRCs bind a
/// section's own bytes to its own offset, so a reader that stopped one section
/// early would still validate every section it did read, and would still open
/// through the engine.
#[test]
fn sample_v2_framing_matches_golden_decode() {
    let root = xfix::v2_root();
    let sample = xfix::load_sample_v2(&root);
    let want_text = xfix::read_root_text(&root, "GOLDEN-DECODE.tsv");
    let want = xfix::golden_rows(&want_text);
    let got = xfix::render_framing(&sample);
    xfix::assert_rows_equal("GOLDEN-DECODE.tsv", &want, &got);
    assert!(
        want.iter().any(|r| r.starts_with("hdr\t")) && want.iter().any(|r| r.starts_with("sec\t")),
        "GOLDEN-DECODE.tsv pins no headers or no sections"
    );
}

/// DECODED BODIES, against the file the FROZEN JAVA READER authored.
///
/// Contract §11.2 settles body semantics engine-against-engine rather than in
/// a python pin, because `walfmt.py` is a structural codec and store record
/// semantics written there would be a fifth implementation nobody reviews.
/// Java is authoritative by construction: it wrote this file.
#[test]
fn sample_v2_body_matches_golden_body() {
    let root = xfix::v2_root();
    let sample = xfix::load_sample_v2(&root);
    let want_text = xfix::read_root_text(&root, "GOLDEN-BODY.tsv");
    let want = xfix::golden_rows(&want_text);
    let got = xfix::render_body(&sample);
    xfix::assert_rows_equal("GOLDEN-BODY.tsv", &want, &got);

    // The distinction the whole file exists for must actually be IN it, or the
    // comparison above is a comparison of two files that never disagree about
    // the interesting case. `lenPlus == 0` is NULL content, `lenPlus == 1` is
    // zero-length content, and they differ in BOTH the lenPlus and the sha
    // column so no single-column bug can hide one as the other.
    assert!(
        want.iter().any(|r| r.contains("\tRECORD\t12\t0\t0\t-")),
        "GOLDEN-BODY.tsv pins no NULL-content record"
    );
    assert!(
        want.iter()
            .any(|r| r.ends_with(&format!("\t1\t{}", xfix::EMPTY_SHA))),
        "GOLDEN-BODY.tsv pins no zero-length-content record"
    );
    assert!(
        want.iter().any(|r| r.starts_with("mark\t")),
        "GOLDEN-BODY.tsv pins no mark"
    );
}

/// Nothing in the distributed v2 root may be unexplained.
///
/// The three tables plus one blob per `file` row, and nothing else. A stray
/// `.gz` that no manifest row names is either a fixture the suite silently
/// never runs or a leftover from a half-finished sync; both are the kind of
/// thing that is only ever noticed by a check that enumerates.
#[test]
fn the_v2_resource_tree_has_nothing_unexplained() {
    let root = xfix::v2_root();
    let sample = xfix::load_sample_v2(&root);
    let mut want: BTreeSet<String> = ["MANIFEST.tsv", "GOLDEN-DECODE.tsv", "GOLDEN-BODY.tsv"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    for f in &sample.manifest.files {
        assert!(
            want.insert(f.blob_name()),
            "two file rows claim the blob {}",
            f.blob_name()
        );
    }
    assert_eq!(
        dir_entries(&root),
        want,
        "the distributed v2 root is not exactly the manifest's blobs plus the three tables"
    );
}

// ---------------------------------------------------------------------------
// the dispatch itself
// ---------------------------------------------------------------------------

/// The version line is a HARD dispatch, and the reason is the arity collision.
///
/// `expect <fid> <engine> <verdict> <opener> <placeAs> <openArg>` (v1) and
/// `expect <fid> <engine> <mode> <verdict> <opener> <openArg>` (v2) are both
/// seven fields. Guessing the schema from a row's shape would read `accept` as
/// a mode and `wal3` as a verdict without a single arity check firing, so the
/// version line decides and an unknown version is refused rather than assumed
/// to be the newest.
#[test]
fn the_two_schemas_are_told_apart_by_their_version_line_only() {
    let v1 = "version\t1\nfixture\tf\tk\tjava\tc\nfile\tf\tx.db\t1\taa\tbb\n\
              expect\tf\trust\taccept\tdirect\tx.db\tx.db\n";
    let v2 = "version\t2\nfixture\tf\tk\tjava\tc\nfile\tf\tx\t1\taa\tbb\n\
              expect\tf\trust\tro\taccept\twal3\tx\n";
    assert_eq!(xfix::parse(v1).version(), 1);
    assert_eq!(xfix::parse(v2).version(), 2);

    // The v1 row, read as v2, is not merely different — its third column is a
    // verdict where v2 wants a mode, so the vocabulary check catches it. That
    // is the collision made visible rather than assumed away.
    let v1_rows_labelled_v2 = v1.replacen("version\t1", "version\t2", 1);
    xfix::assert_manifest_refused(
        "v1 expect rows under a v2 version line",
        &v1_rows_labelled_v2,
    );

    xfix::assert_manifest_refused("an unknown schema version", "version\t3\n");
    xfix::assert_manifest_refused("a manifest with no version line", "fixture\tf\tk\tj\tc\n");
    xfix::assert_manifest_refused("a version line with a trailing field", "version\t2\tx\n");
}

/// Rows the reader must refuse rather than skip.
///
/// A reader that ignores what it does not recognise turns every future schema
/// addition into a silent no-op: the row is in the manifest, the suite is
/// green, and nothing ran.
#[test]
fn unrecognised_and_malformed_rows_are_refused() {
    let head = "version\t2\nfixture\tf\twal3-namespace\tjava\tc\n";
    let file = "file\tf\tx.wal.0000000000000001\t36\taa\tbb\n";

    xfix::assert_manifest_refused(
        "an unknown row type",
        &format!("{head}{file}sparkle\tf\tx\n"),
    );
    xfix::assert_manifest_refused("a short file row", &format!("{head}file\tf\tx\t36\taa\n"));
    xfix::assert_manifest_refused(
        "a file row with an empty field",
        &format!("{head}file\tf\tx\t36\t\tbb\n"),
    );
    xfix::assert_manifest_refused(
        "a non-canonical integer",
        &format!("{head}file\tf\tx\t036\taa\tbb\n"),
    );
    xfix::assert_manifest_refused(
        "a relName that escapes the cell directory",
        &format!("{head}file\tf\t../x\t36\taa\tbb\n"),
    );
    xfix::assert_manifest_refused(
        "an unknown engine",
        &format!("{head}{file}expect\tf\tgo\tro\taccept\twal3\tx\n"),
    );
    xfix::assert_manifest_refused(
        "an unknown mode",
        &format!("{head}{file}expect\tf\trust\trwx\taccept\twal3\tx\n"),
    );
    xfix::assert_manifest_refused(
        "a duplicate expect row",
        &format!(
            "{head}{file}expect\tf\trust\tro\taccept\twal3\tx\n\
             expect\tf\trust\tro\taccept\twal3\tx\n"
        ),
    );
    xfix::assert_manifest_refused(
        "a duplicate post row",
        &format!(
            "{head}{file}post\tf\trust\tro\tx.lock\tunchanged\n\
             post\tf\trust\tro\tx.lock\tdeleted\n"
        ),
    );
    xfix::assert_manifest_refused(
        "an unknown post disposition",
        &format!("{head}{file}post\tf\trust\tro\tx.lock\tvanished\n"),
    );
    xfix::assert_manifest_refused(
        "a sized post disposition missing its sha",
        &format!("{head}{file}post\tf\trust\tro\tx.lock\tcreated:0\n"),
    );
    xfix::assert_manifest_refused(
        "a duplicate recid within one fixture",
        &format!("{head}{file}recid\tf\ta\t1\tlive\t1\t1\nrecid\tf\tb\t1\tlive\t1\t1\n"),
    );
    xfix::assert_manifest_refused(
        "an unbounded recidrange",
        &format!("{head}{file}recidrange\tf\tr\t1\t99999999\tlive\t1\t1\n"),
    );
    xfix::assert_manifest_refused("a v2 manifest with no file rows", head);
}

/// A `bytes` row is refused BY NAME, not skipped.
///
/// `bytes` describes a derived fixture built by `walfmt.py` from another
/// fixture's image, and C4 is the slice that introduces both the deriver and
/// the fixtures. Until then the honest thing for a reader to do is stop: a
/// `bytes` row silently ignored is a fixture that the manifest says exists and
/// that nothing ever opens.
#[test]
fn a_bytes_row_is_refused_until_c4_can_execute_it() {
    xfix::assert_manifest_refused(
        "a v2 `bytes` row",
        "version\t2\nfixture\tf\twal3-namespace\tjava\tc\n\
         file\tf\tx.wal.0000000000000001\t36\taa\tbb\n\
         bytes\tf\tsrc\tx\t0\t36\taa\n",
    );
}

// ---------------------------------------------------------------------------
// the D6 post-state rule, exercised directly
// ---------------------------------------------------------------------------

/// Both sides of the post-state rule, on inputs the sample cannot supply.
///
/// The sample's `rw` and `ro` cells leave every segment untouched and create
/// one `x.lock`, so the corpus is CONSTANT in everything the rule's second side
/// checks: removing "an unnamed input must still be there byte for byte" and
/// "a file that is neither an input nor named must not exist" leaves the whole
/// suite green — measured, both mutants survived the first campaign. That is
/// lesson (g) again, and the answer is the same: an input built to vary. These
/// directories are built by hand.
#[test]
fn the_post_state_rule_fails_in_both_directions() {
    let session = xfix::session_dir("xfix_post");
    let mut n = 0usize;

    /// `(what, inputs placed, files present afterwards, post rows, accepted?)`
    type Case = (
        &'static str,
        Vec<(&'static str, &'static [u8])>,
        Vec<(&'static str, &'static [u8])>,
        Vec<&'static str>,
        bool,
    );
    let cases: Vec<Case> = vec![
        (
            "an untouched input named by nothing",
            vec![("seg", b"abc")],
            vec![("seg", b"abc")],
            vec![],
            true,
        ),
        (
            "an unnamed input rewritten behind the rule's back",
            vec![("seg", b"abc")],
            vec![("seg", b"abd")],
            vec![],
            false,
        ),
        (
            "an unnamed input deleted behind the rule's back",
            vec![("seg", b"abc")],
            vec![],
            vec![],
            false,
        ),
        (
            "a file that is neither an input nor named",
            vec![("seg", b"abc")],
            vec![("seg", b"abc"), ("surprise", b"x")],
            vec![],
            false,
        ),
        (
            "a lock file the post rows do declare",
            vec![("seg", b"abc")],
            vec![("seg", b"abc"), ("x.lock", b"")],
            vec!["x.lock\tcreated:0:E"],
            true,
        ),
        (
            "a `created` file whose sha does not match",
            vec![("seg", b"abc")],
            vec![("seg", b"abc"), ("x.lock", b"z")],
            vec!["x.lock\tcreated:0:E"],
            false,
        ),
        (
            "a `deleted` file that is still there",
            vec![("seg", b"abc")],
            vec![("seg", b"abc")],
            vec!["seg\tdeleted"],
            false,
        ),
        (
            "a `deleted` file that really is gone",
            vec![("seg", b"abc")],
            vec![],
            vec!["seg\tdeleted"],
            true,
        ),
        (
            "an `unchanged` file that changed",
            vec![("seg", b"abc")],
            vec![("seg", b"abd")],
            vec!["seg\tunchanged"],
            false,
        ),
        (
            "`modified` naming a file that was never an input",
            vec![("seg", b"abc")],
            vec![("seg", b"abc"), ("new", b"q")],
            vec!["new\tmodified:1:S"],
            false,
        ),
    ];

    for (what, inputs, after, posts, want_ok) in cases {
        n += 1;
        let cell = session.join(format!("post-{n}"));
        std::fs::create_dir_all(&cell).unwrap();
        let mut before = std::collections::BTreeMap::new();
        for (name, bytes) in &inputs {
            before.insert((*name).to_string(), bytes.to_vec());
        }
        for (name, bytes) in &after {
            std::fs::write(cell.join(name), bytes).unwrap();
        }

        // The post rows are written as manifest text and parsed by the real
        // reader, so the disposition grammar under test is the shipped one.
        let mut text = "version\t2\nfixture\tf\twal3-namespace\tjava\tc\n\
                        file\tf\tseg\t3\taa\tbb\n"
            .to_string();
        for row in &posts {
            let row = row
                .replace('E', xfix::EMPTY_SHA)
                .replace('S', &xfix::sha256_hex(b"q"));
            text.push_str(&format!("post\tf\trust\tro\t{row}\n"));
        }
        let loaded = xfix::parse(&text);
        let m = loaded.v2();
        let rows = m.posts_of("f", "rust", "ro");

        let cell2 = cell.clone();
        let run = move || xfix::assert_post_state(&cell2, &before, &rows, what);
        if want_ok {
            run();
        } else {
            xfix::assert_refused(what, run);
        }
        std::fs::remove_dir_all(&cell).unwrap();
    }
    assert_eq!(n, 10, "the post-state battery lost a case");
    let _ = std::fs::remove_dir_all(&session);
}
