//! StoreDirect cross-port harness (Stage C, **C7r** residual).
//!
//! Contract §9 retires the WAL schema-v1 tree; it does not retire the StoreDirect
//! accept images (`direct-v1-*`) or the shared malformed-StoreDirect reject images.
//! Those lived under the same schema-v1 root, so C7 keeps them as a dedicated
//! schema-v2 root (`tests/xfixtures-direct/`) without reintroducing dual dispatch.

#[path = "../src/store/xfix.rs"]
mod xfix;

use mapdb_rust_store::error::DbError;
use mapdb_rust_store::store::{Store, StoreDirect};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn direct_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/xfixtures-direct")
}

#[test]
fn store_direct_cross_port_cells_conform() {
    let root = direct_root();
    let sample = xfix::load_sample_v2(&root);
    let m = &sample.manifest;
    let session = xfix::session_dir("xfix_direct");
    let mut accepts = 0usize;
    let mut rejects = 0usize;

    for e in &m.expects {
        if e.engine != xfix::ENGINE {
            continue;
        }
        assert_eq!(
            e.opener, "direct",
            "this harness only runs the direct opener"
        );
        assert_eq!(
            e.mode, "rw",
            "StoreDirect has no read-only cell in this harness"
        );

        let cell = session.join(format!("cell-{}", accepts + rejects));
        std::fs::create_dir_all(&cell).unwrap();
        for f in m.files.iter().filter(|f| f.fixture == e.fixture) {
            let bytes = sample
                .raw
                .get(&(f.fixture.clone(), f.rel.clone()))
                .unwrap_or_else(|| panic!("missing raw for {}/{}", f.fixture, f.rel));
            std::fs::write(cell.join(&f.rel), bytes).unwrap();
        }
        let target = cell.join(&e.open_arg);
        let before = std::fs::read(&target).unwrap();

        match e.verdict.as_str() {
            "accept" => {
                let s = StoreDirect::open_file(&target)
                    .unwrap_or_else(|err| panic!("{}: accept failed to open: {err}", e.fixture));
                let recids = m.recids_of(&e.fixture);
                xfix::assert_reader_contract(&s, &recids, &format!("direct {}", e.fixture));
                s.close().unwrap();
                accepts += 1;
            }
            "reject" => match StoreDirect::open_file(&target) {
                Err(DbError::DataCorruption(_)) => rejects += 1,
                Err(other) => panic!("{}: expected DataCorruption, got: {other}", e.fixture),
                Ok(s) => {
                    let _ = s.close();
                    panic!("{}: reject cell opened successfully", e.fixture);
                }
            },
            other => panic!("unknown verdict {other}"),
        }

        assert_eq!(
            std::fs::read(&target).unwrap(),
            before,
            "{}: working copy bytes changed",
            e.fixture
        );
        let _ = std::fs::remove_dir_all(&cell);
    }
    assert_eq!(
        accepts, 3,
        "missing a StoreDirect accept cell (3 writers × this reader)"
    );
    assert_eq!(
        rejects, 4,
        "missing a StoreDirect reject cell (4 shared malformed images)"
    );
    let _ = std::fs::remove_dir_all(&session);
}

// ---------------------------------------------------------------------------
// the root itself
// ---------------------------------------------------------------------------

fn dir_entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// Production inventory check — parameterized on `root` so the red-check below
/// can doctor a COPY while this stays the one assertion both run.
fn assert_direct_rootset(root: &Path, sample: &xfix::SampleV2) {
    let mut want: BTreeSet<String> = BTreeSet::new();
    want.insert("MANIFEST.tsv".to_string());
    for f in &sample.manifest.files {
        assert!(
            want.insert(f.blob_name()),
            "two file rows claim the blob {}",
            f.blob_name()
        );
    }
    assert_eq!(
        want,
        dir_entries(root),
        "the direct root holds files no `file` row accounts for (or is missing one)"
    );
}

/// `MANIFEST.tsv` plus one blob per `file` row, and nothing else.
///
/// The corpus root and the distributed v2 sample each have this check; this
/// root — carved out by C7r and never given one — did not, so an unaccounted
/// extra file here was invisible. That is the failure the enumerating form
/// exists to catch and the counting form cannot: `store_direct_cross_port_
/// cells_conform` pins 3 accepts and 4 rejects, which a stray eighth blob no
/// `expect` row names does not disturb, and a stray blob is either a fixture
/// the suite silently never runs or a leftover from a half-finished sync.
///
/// No golden tables here, unlike the v2 sample: `GOLDEN-DECODE.tsv` and
/// `GOLDEN-BODY.tsv` describe WAL segments, and this root holds StoreDirect
/// files.
#[test]
fn the_direct_root_has_nothing_unexplained() {
    let root = direct_root();
    let sample = xfix::load_sample_v2(&root);
    assert_direct_rootset(&root, &sample);
}

/// The inventory check's own red: an extra file in the root must be refused,
/// and by THAT diagnostic.
///
/// Runs against a copy, because the check under test is an assertion about the
/// real root and a test that doctored the real root would be editing the
/// fixture tree the rest of the suite reads.
#[test]
fn direct_rootset_refuses_an_extra_file() {
    let src = direct_root();
    let sample = xfix::load_sample_v2(&src);
    let session = xfix::session_dir("xfix_direct_rootset");
    let root = session.join("root");
    std::fs::create_dir_all(&root).unwrap();
    for name in dir_entries(&src) {
        std::fs::copy(src.join(&name), root.join(&name)).unwrap();
    }
    // Green on the faithful copy first, or "refused the extra file" could be
    // "refused the copy" — the same mistake the corpus root's red-check makes
    // impossible by copying the tree it then doctors.
    assert_direct_rootset(&root, &sample);

    std::fs::write(root.join("EXTRA_NOT_IN_MANIFEST"), b"x").unwrap();
    let msg = xfix::red_of(|| assert_direct_rootset(&root, &sample))
        .unwrap_or_else(|| panic!("the direct rootset accepted a root with an extra file"));
    assert!(
        msg.contains("no `file` row accounts for"),
        "direct rootset: got: {msg}"
    );
}
