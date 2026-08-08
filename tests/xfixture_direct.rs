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
use std::path::PathBuf;

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
