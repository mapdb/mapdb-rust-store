//! Cross-port conformance harness — schema **v2 only** (Stage C slice **C7r**).
//!
//! `tests/xfixtures-v2/` is the shared static schema-v2 sample
//! (`todo/store-cross/sample-v2/`, byte for byte). Schema v1 and its skip list
//! retired at C7r after the corpus cutover (C6) was green. The frozen corpus
//! runs in `xfixture_corpus.rs`.
//!
//! What runs here, and what does not:
//!
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

use mapdb_rust_store::store::StoreWAL;
use std::collections::BTreeSet;
use std::path::Path;

fn dir_entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("read dir")
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

    // The file's own provenance block, which the row comparison drops. Java
    // compares this file's whole text; rust compares rows, so without this the
    // header could be rewritten to claim a different author while every test
    // stayed green — and the authority claim is the reason this port is graded
    // against this file at all.
    let comments: Vec<&str> = want_text
        .lines()
        .take_while(|l| l.starts_with('#'))
        .collect();
    assert_eq!(
        comments,
        xfix::GOLDEN_BODY_HEADER,
        "GOLDEN-BODY.tsv's provenance header is not the one this port was written against"
    );

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
    assert!(
        want.iter().any(|r| r.contains("\tAPPEND\t")),
        "GOLDEN-BODY.tsv pins no APPEND entry (C9a / O1)"
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
// the version gate (C7r)
// ---------------------------------------------------------------------------

/// Schema version 1 is retired; schema version 2 is the only accepted form.
///
/// The historical reason the version line is a hard gate: v1 and v2 `expect`
/// rows were both seven fields with different columns. A reader that keyed on
/// arity would put `mode` where `verdict` belongs. These cases pin that v2
/// columns land correctly and that v1 (and any other version) is refused.
#[test]
fn the_reader_accepts_only_schema_v2() {
    let v2 = "version\t2\nfixture\tf\twal3-namespace\tjava\tc\nfile\tf\tx\t1\taa\tbb\n\
              expect\tf\trust\tro\taccept\twal3\tx\n";
    assert_eq!(xfix::parse(v2).version(), 2);
    assert_eq!(xfix::parse(v2).v2().expects[0].mode, "ro");
    assert_eq!(xfix::parse(v2).v2().expects[0].verdict, "accept");

    xfix::assert_manifest_refused(
        "retired schema version 1",
        "version\t1\nfixture\tf\tdirect\tjava\tc\nfile\tf\tx.db\t1\taa\tbb\n\
         expect\tf\trust\taccept\tdirect\tx.db\tx.db\n",
    );
    // A v1-shaped expect under a v2 version line is still refused (vocabulary).
    xfix::assert_manifest_refused(
        "v1 expect rows under a v2 version line",
        "version\t2\nfixture\tf\tdirect\tjava\tc\nfile\tf\tx.db\t1\taa\tbb\n\
         expect\tf\trust\taccept\tdirect\tx.db\tx.db\n",
    );
    xfix::assert_manifest_refused("an unknown schema version", "version\t3\n");
    xfix::assert_manifest_refused(
        "a manifest with no version line",
        "fixture\tf\tdirect\tjava\tc\n",
    );
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
        "an unknown verdict on a java row",
        &format!("{head}{file}expect\tf\tjava\tro\tmaybe\twal3\tx\n"),
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

    // Vocabularies contract §2 makes load-bearing. The C3r review found kind,
    // generatorEngine and opener stored unchecked — and `opener` in particular
    // was only ever validated by `run_v2_cells`, AFTER it had filtered to this
    // engine's rows, so a bad opener on a java or zig row reached nothing. The
    // cases below are therefore addressed to OTHER engines on purpose: executor
    // filtering must not be able to masquerade as parser validation.
    xfix::assert_manifest_refused(
        "an unknown opener on a java row",
        &format!("{head}{file}expect\tf\tjava\tro\taccept\twal9\tx\n"),
    );
    xfix::assert_manifest_refused(
        "an unknown opener on a zig row",
        &format!("{head}{file}expect\tf\tzig\trw\taccept\tdirekt\tx\n"),
    );
    xfix::assert_manifest_refused(
        "an unknown fixture kind",
        "version\t2\nfixture\tf\twal3-namespaces\tjava\tc\n\
         file\tf\tx.wal.0000000000000001\t36\taa\tbb\n",
    );
    xfix::assert_manifest_refused(
        "an unknown generatorEngine",
        "version\t2\nfixture\tf\twal3-namespace\tgo\tc\n\
         file\tf\tx.wal.0000000000000001\t36\taa\tbb\n",
    );
    // `port-wal` and `java-wal-namespace` are RETAINED tokens: no v2 fixture
    // uses them, and §2 says retiring a family is not a reason for a
    // version-dispatch parser to reject the token. So they must still parse.
    for kind in [
        "direct",
        "reject",
        "wal3-namespace",
        "port-wal",
        "java-wal-namespace",
    ] {
        xfix::parse(&format!(
            "version\t2\nfixture\tf\t{kind}\tjava\tc\n\
             file\tf\tx.wal.0000000000000001\t36\taa\tbb\n"
        ));
    }

    // §2 amendment 3: `generatorEngine = derived` and a `derived` row imply
    // each other, exactly once.
    xfix::assert_manifest_refused(
        "a derived fixture with no derived row",
        "version\t2\nfixture\tf\treject\tderived\tc\n\
         file\tf\tx.wal.0000000000000001\t36\taa\tbb\n",
    );
    xfix::assert_manifest_refused(
        "a derived row on a fixture an engine wrote",
        &format!("{head}{file}derived\tf\tsrc\t1\trecipe\n"),
    );
}

/// A fixture id a row REFERS to must be DECLARED, and vice versa.
///
/// Without this the exact-cell-set rule has a coordinated escape: delete a
/// `fixture` row together with this engine's `expect` rows for it, and both
/// halves of the executor see a consistently smaller world — the expected set
/// shrinks by exactly the cell that stopped running. The `file` and `recid` rows
/// stay behind, the golden comparisons still decode them, and the resource
/// inventory is unchanged. Found by the C3r review; the fix is to make the
/// declaration load-bearing for the rows that were NOT deleted.
#[test]
fn every_referenced_fixture_must_be_declared_and_every_declared_one_used() {
    let decl = "version\t2\nfixture\tf\twal3-namespace\tjava\tc\n";
    let file = "file\tf\tx.wal.0000000000000001\t36\taa\tbb\n";

    xfix::parse(&format!("{decl}{file}"));

    xfix::assert_manifest_refused(
        "a file row naming a fixture with no fixture row",
        &format!("{decl}{file}file\tg\tx.wal.0000000000000002\t36\tcc\tdd\n"),
    );
    xfix::assert_manifest_refused(
        "an expect row naming a fixture with no fixture row",
        &format!("{decl}{file}expect\tg\trust\tro\taccept\twal3\tx\n"),
    );
    xfix::assert_manifest_refused(
        "a post row naming a fixture with no fixture row",
        &format!("{decl}{file}post\tg\trust\tro\tx.lock\tunchanged\n"),
    );
    xfix::assert_manifest_refused(
        "a recid row naming a fixture with no fixture row",
        &format!("{decl}{file}recid\tg\tr1\t1\tlive\t1\t8\n"),
    );
    xfix::assert_manifest_refused(
        "a declared fixture no row refers to",
        &format!("{decl}{file}fixture\tg\twal3-namespace\tjava\tc\n"),
    );
}

/// A `bytes` row is refused BY NAME, not skipped.
///
/// `bytes` describes a derived fixture built by `walfmt.py` from another
/// fixture's image, and C4 is the slice that introduces both the deriver and
/// the fixtures. Until then the honest thing for a reader to do is stop: a
/// `bytes` row silently ignored is a fixture that the manifest says exists and
/// that nothing ever opens.
/// The four `v2-oracle` row types C5 added — parsed, and their own rules
/// enforced.
///
/// Until C5r this reader refused a `bytes` row outright. That refusal was the
/// honest state of the world (C4 introduced the derived fixtures the row
/// describes and no engine could execute it), and it is now a defect: the
/// preflight corpus carries all four types, and a reader that refuses one
/// refuses the corpus.
///
/// The rows below are GRAMMATICALLY VALID except in the one place each case
/// names. That distinction is not pedantry — the first version of the old
/// `bytes` case supplied an unknown engine, an unknown mode AND a relName of
/// `0`, and the unconditional panic masked all three, so the test proved only
/// that a malformed row is refused.
#[test]
fn the_c5_oracle_rows_parse_and_are_checked() {
    let head = "version\t2\nfixture\tf\twal3-namespace\tjava\tc\n\
                file\tf\tx.wal.0000000000000001\t36\taa\tbb\n";
    let good = format!(
        "{head}applies\tf\trust\tro\n\
         expect\tf\trust\tro\taccept\twal3\tx\n\
         action\tf\trust\tro\tcommit_one_record\top=put,payload_id=1,payload_len=2,\
         recid_label=Z,serializer=raw\n\
         bytes\tf\trust\tro\tx.wal.0000000000000001\t0\taabb\n\
         reopen\tf\trust\tro\tS2\n\
         family\tf\trust\tro\tS2\n"
    );
    let m = xfix::parse(&good);
    let m = m.v2();
    assert_eq!(m.applies.len(), 1, "the applies row parsed");
    assert_eq!(m.actions_of("f", "rust", "ro").len(), 1);
    assert_eq!(m.bytes_of("f", "rust", "ro").len(), 1);
    assert_eq!(m.reopens_of("f", "rust", "ro").len(), 1);
    assert_eq!(m.families_of("f", "rust", "ro").len(), 1);
    assert_eq!(m.actions[0].verb, "commit_one_record");
    assert_eq!(m.bytes[0].offset, 0);
    assert_eq!(m.bytes[0].hex, "aabb");
    assert_eq!(m.reopens[0].family, "S2");
    assert_eq!(m.families[0].family, "S2");

    let cases: [(&str, String); 14] = [
        (
            "a duplicate applies row",
            format!("{head}applies\tf\trust\tro\napplies\tf\trust\tro\n"),
        ),
        (
            "an out-of-vocabulary engine on an applies row",
            format!("{head}applies\tf\tgo\tro\n"),
        ),
        (
            "an out-of-vocabulary mode on a reopen row",
            format!("{head}reopen\tf\trust\trwx\tS2\n"),
        ),
        (
            "a second reopen row for one cell",
            format!("{head}reopen\tf\trust\tro\tS2\nreopen\tf\trust\tro\tS9\n"),
        ),
        (
            "an out-of-vocabulary mode on a family row",
            format!("{head}family\tf\trust\trwx\tS2\n"),
        ),
        (
            "a second family row for one cell",
            format!("{head}family\tf\trust\tro\tS2\nfamily\tf\trust\tro\tS9\n"),
        ),
        (
            "a second action row for one cell and verb",
            format!("{head}action\tf\trust\tro\tv\ta=1\naction\tf\trust\tro\tv\ta=2\n"),
        ),
        (
            "action argument keys out of sorted order",
            format!("{head}action\tf\trust\tro\tv\tb=1,a=2\n"),
        ),
        (
            "a repeated action argument key",
            format!("{head}action\tf\trust\tro\tv\ta=1,a=2\n"),
        ),
        (
            "an action argument key outside [a-z][a-z0-9_]*",
            format!("{head}action\tf\trust\tro\tv\tA=1\n"),
        ),
        (
            "an action argument value outside the pinned character class",
            format!("{head}action\tf\trust\tro\tv\ta=one two\n"),
        ),
        (
            "an action argument that is not a k=v pair",
            format!("{head}action\tf\trust\tro\tv\tab\n"),
        ),
        (
            "a bytes row whose value is odd-length hex",
            format!("{head}bytes\tf\trust\tro\tx.wal.0000000000000001\t0\taab\n"),
        ),
        (
            "a bytes row whose value is uppercase hex",
            format!("{head}bytes\tf\trust\tro\tx.wal.0000000000000001\t0\tAA\n"),
        ),
    ];
    for (what, text) in &cases {
        xfix::assert_manifest_refused(what, text);
    }
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
        // §2.1's split has TWO sides: a post row is an explicit override of an
        // input, or an explicit NEW file. The three cases below are the ones the
        // C3r review found missing, and each was green under the old rule.
        (
            "`created` naming a file that WAS an input",
            vec![("seg", b"q")],
            vec![("seg", b"q")],
            vec!["seg\tcreated:1:S"],
            false,
        ),
        (
            "`deleted` naming a file that was never an input",
            vec![("seg", b"abc")],
            vec![("seg", b"abc")],
            vec!["ghost\tdeleted"],
            false,
        ),
        // Refused, and C5r moved WHERE. The rule used to decide presence with
        // `symlink_metadata` itself; now `capture` does, one step earlier, and
        // refuses a name that is present but is not a regular file. Both
        // refuse this input for the same reason — `read(..).ok()` would call a
        // directory "absent" and let a `deleted` row pass on something very
        // much still there — and the campaign measured that deleting
        // `capture`'s assertion leaves `read_named` refusing it too. So this
        // case still measures the hazard its name claims; what it no longer
        // isolates is which of the two statements catches it.
        // The two quadrants round 5 found had no input. Both are false rows the
        // checker exists to reject, and a defective executor accepting either
        // passed the complete gate: `None == None`, so equality alone cannot
        // establish that an `unchanged` row names a file that was there, and
        // every `created` case had a post file, so nothing required one.
        (
            "an `unchanged` row naming a file absent before and after",
            vec![("seg", b"abc")],
            vec![("seg", b"abc")],
            vec!["ghost\tunchanged"],
            false,
        ),
        (
            "a `created` file that is missing after the cell",
            vec![("seg", b"abc")],
            vec![("seg", b"abc")],
            vec!["x.lock\tcreated:0:E"],
            false,
        ),
        // The two verbs whose RELATION to the input was never asserted, both
        // directions each. Round 3 of review found the shared arm made both
        // decorative — a file that grew satisfied `truncated`, an unchanged
        // file satisfied `modified` — and constructed these inputs to prove it.
        // The disposition I had written said this needed C5t's torn-tail
        // images; it needed six lines here.
        (
            "a `truncated` file that really shrank",
            vec![("seg", b"qx")],
            vec![("seg", b"q")],
            vec!["seg\ttruncated:1:S"],
            true,
        ),
        (
            "a `truncated` file that grew",
            vec![("seg", b"")],
            vec![("seg", b"q")],
            vec!["seg\ttruncated:1:S"],
            false,
        ),
        // The PROPER half of "proper prefix", which round 4 found had no input:
        // with only a growth and a non-prefix to fail on, `<` could regress to
        // `<=` and the whole gate stayed green. A one-statement conjunction
        // removes the masking between its halves and does NOT prove them —
        // deleting it shows that some part matters, never which. The inputs
        // have to cover each half.
        (
            "a `truncated` file whose bytes are exactly the input",
            vec![("seg", b"q")],
            vec![("seg", b"q")],
            vec!["seg\ttruncated:1:S"],
            false,
        ),
        (
            "a `truncated` file that shrank but is not a PREFIX of the input",
            vec![("seg", b"zq")],
            vec![("seg", b"q")],
            vec!["seg\ttruncated:1:S"],
            false,
        ),
        (
            "a `modified` file that really changed",
            vec![("seg", b"z")],
            vec![("seg", b"q")],
            vec!["seg\tmodified:1:S"],
            true,
        ),
        (
            "a `modified` file whose bytes are unchanged",
            vec![("seg", b"q")],
            vec![("seg", b"q")],
            vec!["seg\tmodified:1:S"],
            false,
        ),
        (
            "a `modified` row describing what is really a truncation",
            vec![("seg", b"qx")],
            vec![("seg", b"q")],
            vec!["seg\tmodified:1:S"],
            false,
        ),
        (
            "a `deleted` file replaced by a directory of the same name",
            vec![("seg", b"abc")],
            vec![],
            vec!["seg\tdeleted"],
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
        if what.contains("replaced by a directory") {
            std::fs::create_dir(cell.join("seg")).unwrap();
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

        // The capture is taken the way a real cell takes it, so this battery
        // grades the shipped pairing rather than a second reading of the
        // directory: `capture` is where a name that is not a regular file is
        // refused, and the "replaced by a directory" case reaches it there.
        let cell2 = cell.clone();
        let run = move || {
            let mut owed = xfix::Consumption::new(what);
            for p in &rows {
                owed.owe(&format!("post {}", p.rel), *p);
            }
            let after = xfix::capture(&cell2, what);
            xfix::assert_post_state(&rows, &before, &after, what, &mut owed);
            // Every row this battery hands the rule must come back consumed:
            // otherwise a `want_ok` case could pass by the rule skipping it.
            owed.require_all_consumed();
        };
        if want_ok {
            run();
        } else {
            xfix::assert_refused(what, run);
        }
        std::fs::remove_dir_all(&cell).unwrap();
    }
    assert_eq!(n, 22, "the post-state battery lost a case");
    let _ = std::fs::remove_dir_all(&session);
}
