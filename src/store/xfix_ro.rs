//! The schema-v2 **`ro` cell executor** — decision **C-D3**.
//!
//! It lives in the crate rather than in `tests/` for one reason: a read-only
//! open goes through [`StoreWAL::open_cfg`] with [`WalOptions::read_only`], and
//! both are `pub(crate)`. The Stage C plan's first revision proposed exporting
//! a `#[doc(hidden)] pub fn open_read_only` instead, and that was refused —
//! `#[doc(hidden)]` hides an item from rustdoc and from nothing else, so the
//! function would still be callable by any downstream crate and still
//! semver-visible, which contradicts D7's "no public read-only DB surface in
//! this workstream". Java already having a public `openReadOnly` authorizes
//! nothing here.
//!
//! So the cells that need the crate-internal opener run here, the cells that do
//! not run in `tests/xfixture_conformance.rs`, and everything else —
//! manifest dispatch, decoder, assertions — is shared through [`super::xfix`],
//! which both builds compile. **Rust gains no public API.**

use super::wal::{StoreWAL, WalOptions};
use super::xfix;
use super::Store;
use std::path::{Path, PathBuf};

/// Opens read-only through the crate-internal seam. `read_only: true` is the
/// whole point of the module; everything else is the default configuration, so
/// an `ro` cell differs from its `rw` twin in exactly one field.
fn open_ro(base: &Path) -> crate::error::Result<StoreWAL> {
    StoreWAL::open_cfg(
        base,
        WalOptions {
            read_only: true,
            ..Default::default()
        },
    )
}

#[test]
fn sample_v2_ro_cells_pass() {
    let sample = xfix::load_sample_v2(&xfix::v2_root());
    let session = xfix::session_dir("xfix_v2_ro");
    xfix::run_v2_cells(&sample, "ro", &session, &open_ro);
    let _ = std::fs::remove_dir_all(&session);
}

// ---------------------------------------------------------------------------
// the preflight corpus, `ro` half — slice C5r
// ---------------------------------------------------------------------------

/// Every `applies` row addressed to rust in `ro`, run, and exactly those.
///
/// The `rw` half is `tests/xfixture_corpus.rs`, which also carries the doctored
/// cases: they exercise rules that are not mode-specific, and running each of
/// them twice would double the suite to grade the same statement. What must run
/// HERE is everything the read-only opener is the only way to reach — this
/// cell set, the write probe, and the probe's own firing.
#[test]
fn corpus_ro_cells_conform() {
    // C8f f3: raw sealed MANIFEST (family rows frozen in corpus-v2).
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let session = xfix::session_dir("xfix_corpus_ro");
    xfix::run_v2_corpus_cells(&sample, "ro", &session, &open_ro);
    let _ = std::fs::remove_dir_all(&session);
}

/// C9m `roset`: skip probe bookkeeping so `ro_probed` disagrees with `ro_cells`.
/// Lives here because only this module reaches the crate-internal read-only open.
#[test]
fn roset_detects_missing_probe_records() {
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let session = xfix::session_dir("xfix_roset_skip");
    let msg = xfix::red_of(|| {
        xfix::with_skip_ro_probe_record(|| {
            xfix::run_v2_corpus_cells(&sample, "ro", &session, &open_ro);
        });
    })
    .unwrap_or_else(|| panic!("roset accepted a ro suite that recorded no probes"));
    assert!(
        msg.contains("the ro cells whose read-only handle was probed with a write"),
        "roset: got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&session);
}

/// The read-only refusal is not vacuous: the same write, on the same fixture,
/// through the two handles, with opposite outcomes.
///
/// C3z's review found the general shape this closes — `mode` was parsed,
/// vocabulary-checked and used to pick an opener, and then nothing observed the
/// difference, so every `ro` cell in java and rust was a writable open wearing
/// a label. Asserting only the refusal would leave the same hole one step
/// along: a `put` that failed for an unrelated reason in BOTH modes would
/// satisfy it. So the `rw` half runs here too, and it is the same call.
///
/// This runs outside the cell executor deliberately — the `rw` write mutates
/// the directory, and doing it inside a cell would author a post state no
/// manifest row describes.
#[test]
fn the_read_only_write_refusal_discriminates() {
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let session = xfix::session_dir("xfix_corpus_ro_disc");
    let fid = "wal3-java-cleaned";

    let rw_dir = stage(&sample, fid, &session, "rw");
    let recid = {
        let s = StoreWAL::open(&rw_dir.join("x")).expect("the rw handle must open");
        let r = s.put(&vec![1u8, 2, 3], &xfix::R);
        s.close().unwrap();
        r
    };
    assert!(
        recid.is_ok(),
        "the writable handle refused the write ({recid:?}), so the read-only half below proves \
         nothing about the mode"
    );

    let ro_dir = stage(&sample, fid, &session, "ro");
    let refusal = {
        let s = open_ro(&ro_dir.join("x")).expect("the ro handle must open");
        let r = s.put(&vec![1u8, 2, 3], &xfix::R);
        s.close().unwrap();
        r
    };
    let err = refusal.expect_err("the read-only handle ACCEPTED the write");
    assert!(
        err.to_string().contains("read-only"),
        "the refusal does not name the mode: {err}"
    );
    let _ = std::fs::remove_dir_all(&session);
}

/// The read-only probe's assertion FIRES — which no corpus input can show.
///
/// A conforming engine refuses the write, so the red side is unreachable from
/// the corpus and the assertion could be deleted with the whole gate green
/// while `Cells::ro_probed` still attested the probe "ran". So the method is
/// handed the two inputs the corpus cannot produce: a WRITABLE handle, and a
/// handle that refuses for the wrong reason.
///
/// **The reds are COLLECTED and compared as an ordered list, not asserted one
/// at a time.** A statement that no other statement depends on is invisible to
/// deletion, however many assertions it contains; comparing the collected
/// outcomes makes each INPUT observable too — drop either call and the list is
/// short. The order is what binds `ACCEPTED` to the writable handle and
/// `WRONG-REASON` to the closed one, which a set would lose.
#[test]
fn the_read_only_write_probe_fires() {
    let sample = xfix::load_sample_v2(&xfix::v2_corpus_root());
    let session = xfix::session_dir("xfix_corpus_ro_fire");
    let fid = "wal3-java-cleaned";
    let e = sample
        .manifest
        .expects
        .iter()
        .find(|e| e.fixture == fid && e.engine == xfix::ENGINE && e.mode == "ro")
        .expect("the corpus has no rust ro cell for wal3-java-cleaned")
        .clone();

    let mut cells = xfix::Cells::new(&sample);
    let mut reds = Vec::new();

    let writable = StoreWAL::open(&stage(&sample, fid, &session, "w").join("x")).unwrap();
    reds.push(classify(&mut cells, &e, &writable));
    writable.close().unwrap();

    // A handle that refuses for a DIFFERENT reason: closed, and not read-only,
    // so the refusal is `StoreClosed` and its message cannot name the mode.
    let closed = StoreWAL::open(&stage(&sample, fid, &session, "c").join("x")).unwrap();
    closed.close().unwrap();
    reds.push(classify(&mut cells, &e, &closed));

    assert_eq!(
        reds,
        vec!["ACCEPTED".to_string(), "WRONG-REASON".to_string()],
        "the probe's two inputs and the red each must produce"
    );
    assert!(
        cells.ro_probed.is_empty(),
        "the probe recorded a cell it had just refused — the recording must stay downstream of \
         the assertion"
    );
    let _ = std::fs::remove_dir_all(&session);
}

/// Classifies the red one probe input produced, so the two can be compared as
/// an ordered list.
///
/// Classification is by substring, so an unrelated panic carrying either phrase
/// would be labelled as the wanted red. Sound for the two inputs here — the
/// closure calls only `assert_write_refused`, and the store's own failures are
/// `DbError`s it catches — and stated rather than left implied, because it is a
/// claim about the message and not about where it came from.
fn classify(cells: &mut xfix::Cells<'_>, e: &xfix::V2Expect, s: &StoreWAL) -> String {
    let mut probe = std::panic::AssertUnwindSafe((cells, e, s));
    match xfix::red_of(move || {
        let (cells, e, s) = &mut *probe;
        cells.assert_write_refused("probe", e, s);
    }) {
        None => "NO-RED".to_string(),
        Some(msg) if msg.contains("the write was ACCEPTED") => "ACCEPTED".to_string(),
        Some(msg) if msg.contains("refused with:") => "WRONG-REASON".to_string(),
        Some(msg) => format!("OTHER: {msg}"),
    }
}

/// The `ro` half of `require_some_oracle`'s disjunction.
///
/// `tests/xfixture_corpus.rs` proves the `rw` direction: strip
/// `wal3-java-cleaned`'s recid rows and the writable accept cell is refused,
/// because it now asserts nothing. The SAME stripped fixture must PASS in `ro`,
/// where the read-only write refusal is the claim. Without the pair, "an accept
/// cell must assert something" and "ro is exempt from everything" would be
/// indistinguishable.
#[test]
fn a_bare_accept_cell_passes_in_ro_where_the_write_probe_is_the_claim() {
    let root = xfix::v2_corpus_root();
    let text = xfix::read_root_text(&root, "MANIFEST.tsv");
    let stripped: Vec<&str> = text
        .split('\n')
        .filter(|l| !l.starts_with("recid\twal3-java-cleaned\t"))
        .collect();
    let stripped = stripped.join("\n");
    assert_ne!(
        stripped, text,
        "the corpus has no wal3-java-cleaned recid rows"
    );
    let sample = xfix::load_sample_v2_text(&root, &stripped);

    let e = sample
        .manifest
        .expects
        .iter()
        .find(|e| e.fixture == "wal3-java-cleaned" && e.engine == xfix::ENGINE && e.mode == "ro")
        .expect("no rust ro cell")
        .clone();
    let session = xfix::session_dir("xfix_corpus_ro_bare");
    let cell = session.join("cell");
    std::fs::create_dir_all(&cell).unwrap();
    xfix::Cells::new(&sample).run_cell(&e, &cell, &open_ro, xfix::Dispatch::ByManifest);
    let _ = std::fs::remove_dir_all(&session);
}

/// Copies one fixture's inputs into a fresh directory under `session`.
fn stage(sample: &xfix::SampleV2, fid: &str, session: &Path, tag: &str) -> PathBuf {
    let dir = session.join(tag);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in sample.manifest.files_of(fid) {
        std::fs::write(dir.join(&f.rel), sample.bytes_of(f)).unwrap();
    }
    dir
}

/// The transcription check that can only live in-crate.
///
/// [`super::xfix`] is compiled into the integration tests as well, where
/// `crate::` is a test binary, so it cannot import the engine's codec
/// constants and transcribes them instead. A transcription that drifts would
/// make the shared decoder describe a format this engine no longer writes,
/// and every comparison built on it would keep agreeing with itself. This is
/// the one place both sets of names are in scope at once.
#[test]
fn the_transcribed_constants_match_the_engine() {
    use super::index_val as iv;
    use super::wal_recover as rec;
    use super::wal_segments as seg;

    assert_eq!(xfix::SEG_HDR as u64, seg::SEG_HDR, "SEG_HDR");
    assert_eq!(
        xfix::SEG_HDR_CRC_LEN,
        seg::SEG_HDR_CRC_LEN,
        "SEG_HDR_CRC_LEN"
    );
    assert_eq!(&xfix::MAGIC[..], &seg::MAGIC[..], "MAGIC");
    assert_eq!(
        xfix::FORMAT_VERSION as i32,
        seg::FORMAT_VERSION,
        "FORMAT_VERSION"
    );

    assert_eq!(xfix::SEC_HDR, rec::SEC_HDR, "SEC_HDR");
    assert_eq!(
        xfix::SEC_HDR_CRC_LEN,
        rec::SEC_HDR_CRC_LEN,
        "SEC_HDR_CRC_LEN"
    );
    assert_eq!(xfix::MARK_BODY_LEN, rec::MARK_BODY_LEN, "MARK_BODY_LEN");
    assert_eq!(xfix::TAG_SECTION, rec::TAG_SECTION, "TAG_SECTION");
    assert_eq!(xfix::TAG_IMAGE, rec::TAG_IMAGE, "TAG_IMAGE");
    assert_eq!(xfix::TAG_MARK, rec::TAG_MARK, "TAG_MARK");
    assert_eq!(xfix::T_PREALLOC, rec::T_PREALLOC, "T_PREALLOC");
    assert_eq!(xfix::T_RECORD, rec::T_RECORD, "T_RECORD");
    assert_eq!(xfix::T_APPEND, rec::T_APPEND, "T_APPEND");
    assert_eq!(xfix::T_DELETE, rec::T_DELETE, "T_DELETE");

    // Half of `cap_valid`'s rule, and the one the C3r review found missing from
    // both the transcription and the witness built on it.
    assert_eq!(xfix::MAX_CAPACITY, iv::MAX_CAPACITY as i64, "MAX_CAPACITY");
}
