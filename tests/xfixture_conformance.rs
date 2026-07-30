//! Cross-port conformance harness (Stages 1+2) — consumes the checked-in
//! `tests/xfixtures/` bundle produced by the sync script.
//!
//! These fixtures pin the CURRENT state of an UNSTABLE on-disk format for
//! divergence detection between the engines. Cross-engine openability is an
//! implementation fact, not a supported feature; any format change regenerates
//! the fixtures as part of that change.
//!
//! Flow: load `MANIFEST.tsv` (HARD-FAIL if missing or version != 1), gunzip
//! every fixture file once into a session temp dir verifying length + SHA-256,
//! then run every `expect` row with engine == rust in a fresh per-cell temp
//! dir: accept cells (direct AND wal openers) run `verify()` + the per-recid
//! contract + the `get_all_recids` set check; reject cells demand
//! `DbError::DataCorruption`. Every cell asserts the working copy is
//! byte-unchanged afterwards and that no files beyond the allowed `.lock`
//! sidecars appeared (in particular, a `.ckpt` companion must NOT appear
//! after a wal cell's clean close).

use flate2::read::GzDecoder;
use mapdb_rust_store::error::{DbError, Result};
use mapdb_rust_store::io::{DataInput2, DataOutput2};
use mapdb_rust_store::ser::Serializer;
use mapdb_rust_store::store::{Recid, Store, StoreDirect, StoreWAL};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::io::Read as _;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

/// Raw-bytes serializer (same shape as the TCK's `RawSer`): record content ==
/// logical value, so gets compare directly against the contract payloads.
struct RawSer;
impl Serializer<Vec<u8>> for RawSer {
    fn serialize(&self, out: &mut DataOutput2, v: &Vec<u8>) {
        out.write_all(v);
    }
    fn deserialize(&self, input: &mut dyn DataInput2, size: Option<usize>) -> Result<Vec<u8>> {
        let n = size.expect("raw serializer needs a framed size");
        let mut b = vec![0u8; n];
        input.read_fully(&mut b)?;
        Ok(b)
    }
    fn compare(&self, a: &Vec<u8>, b: &Vec<u8>) -> Ordering {
        a.cmp(b)
    }
    fn equals(&self, a: &Vec<u8>, b: &Vec<u8>) -> bool {
        a == b
    }
}
const R: RawSer = RawSer;

/// Contract payload function: `payload(payloadId, len)[i] = (i*131 + payloadId) & 0xff`.
/// Recomputed per cell — the >1 MiB payload is never cached globally.
fn payload(payload_id: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u64).wrapping_mul(131).wrapping_add(payload_id) & 0xff) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// MANIFEST.tsv model
// ---------------------------------------------------------------------------

struct FileRow {
    fixture: String,
    rel: String,
    raw_len: u64,
    raw_sha: String,
    gz_sha: String,
}

struct ExpectRow {
    fixture: String,
    engine: String,
    verdict: String,
    opener: String,
    place_as: String,
    open_arg: String,
}

#[derive(Clone, Copy, PartialEq)]
enum RecidState {
    Live,
    Null,
    Prealloc,
    Deleted,
}

struct RecidRow {
    fixture: String,
    label: String,
    recid: u64,
    state: RecidState,
    payload_id: u64,
    len: usize,
}

struct Manifest {
    files: Vec<FileRow>,
    expects: Vec<ExpectRow>,
    recids: Vec<RecidRow>,
}

fn parse_state(s: &str, line: &str) -> RecidState {
    match s {
        "live" => RecidState::Live,
        "null" => RecidState::Null,
        "prealloc" => RecidState::Prealloc,
        "deleted" => RecidState::Deleted,
        other => panic!("unknown recid state {other:?} in manifest line: {line}"),
    }
}

fn parse_manifest(text: &str) -> Manifest {
    let mut m = Manifest {
        files: Vec::new(),
        expects: Vec::new(),
        recids: Vec::new(),
    };
    let mut version_seen = false;
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if !version_seen {
            // first data line MUST be `version<TAB>1`; HARD-FAIL otherwise.
            assert_eq!(
                f,
                vec!["version", "1"],
                "unsupported MANIFEST.tsv version line: {line:?}"
            );
            version_seen = true;
            continue;
        }
        match f[0] {
            "fixture" => {
                assert_eq!(f.len(), 5, "bad fixture row: {line}");
                // id/kind/generatorEngine/generatorCommit — informational here.
            }
            "file" => {
                assert_eq!(f.len(), 6, "bad file row: {line}");
                m.files.push(FileRow {
                    fixture: f[1].to_string(),
                    rel: f[2].to_string(),
                    raw_len: f[3].parse().expect("rawLen"),
                    raw_sha: f[4].to_string(),
                    gz_sha: f[5].to_string(),
                });
            }
            "expect" => {
                assert_eq!(f.len(), 7, "bad expect row: {line}");
                m.expects.push(ExpectRow {
                    fixture: f[1].to_string(),
                    engine: f[2].to_string(),
                    verdict: f[3].to_string(),
                    opener: f[4].to_string(),
                    place_as: f[5].to_string(),
                    open_arg: f[6].to_string(),
                });
            }
            "recid" => {
                assert_eq!(f.len(), 7, "bad recid row: {line}");
                m.recids.push(RecidRow {
                    fixture: f[1].to_string(),
                    label: f[2].to_string(),
                    recid: f[3].parse().expect("recid"),
                    state: parse_state(f[4], line),
                    payload_id: f[5].parse().expect("payloadId"),
                    len: f[6].parse().expect("len"),
                });
            }
            "recidrange" => {
                assert_eq!(f.len(), 8, "bad recidrange row: {line}");
                let from: u64 = f[3].parse().expect("fromRecid");
                let to: u64 = f[4].parse().expect("toRecid");
                let state = parse_state(f[5], line);
                let base: u64 = f[6].parse().expect("payloadIdBase");
                let len: usize = f[7].parse().expect("len");
                assert!(from <= to, "empty recidrange: {line}");
                for r in from..=to {
                    m.recids.push(RecidRow {
                        fixture: f[1].to_string(),
                        label: format!("{}[{}]", f[2], r - from),
                        recid: r,
                        state,
                        // recidrange payloadId for recid r = base + (r - from)
                        payload_id: base + (r - from),
                        len,
                    });
                }
            }
            "edit" => {
                assert_eq!(f.len(), 6, "bad edit row: {line}");
                // informational (records how reject files were derived).
            }
            other => panic!("unknown manifest row type {other:?}: {line}"),
        }
    }
    assert!(version_seen, "MANIFEST.tsv contains no version row");
    m
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn nz(recid: u64) -> Recid {
    NonZeroU64::new(recid).expect("manifest recid must be nonzero")
}

fn dir_entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("read cell dir")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// Accept cell: verify() + per-recid contract + recid-set check + close.
///
/// Shared by the direct and wal arms — the SAME reader assertion block runs in
/// both. `StoreWAL` DOES expose `verify()` (the `Store` trait requires it, see
/// `impl Store for StoreWAL`), so wal cells run verify() too; nothing is
/// skipped.
fn run_accept<S: Store>(s: &S, recids: &[&RecidRow], ctx: &str) {
    s.verify()
        .unwrap_or_else(|e| panic!("[{ctx}] verify() failed: {e}"));
    let mut want_all: BTreeSet<Recid> = BTreeSet::new();
    for row in recids {
        let recid = nz(row.recid);
        let label = &row.label;
        match row.state {
            RecidState::Live => {
                let got = s
                    .get(recid, &R)
                    .unwrap_or_else(|e| panic!("[{ctx}] get({label}) failed: {e}"));
                assert_eq!(
                    got,
                    Some(payload(row.payload_id, row.len)),
                    "[{ctx}] {label} (recid {recid}) content mismatch"
                );
                want_all.insert(recid);
            }
            RecidState::Null => {
                assert_eq!(
                    s.get(recid, &R).unwrap(),
                    None,
                    "[{ctx}] {label} (recid {recid}) must read as null"
                );
                want_all.insert(recid);
            }
            RecidState::Prealloc => {
                assert_eq!(
                    s.get(recid, &R).unwrap(),
                    None,
                    "[{ctx}] {label} (recid {recid}) prealloc must read as null"
                );
                // excluded from get_all_recids — enforced by the set equality below.
            }
            RecidState::Deleted => {
                assert!(
                    matches!(s.get(recid, &R), Err(DbError::GetVoid(x)) if x == recid.get()),
                    "[{ctx}] {label} (recid {recid}) must be deleted (GetVoid)"
                );
            }
        }
    }
    let all: BTreeSet<Recid> = s.get_all_recids().unwrap().into_iter().collect();
    assert_eq!(
        all, want_all,
        "[{ctx}] get_all_recids must equal the manifest's live+null set"
    );
    s.close().unwrap();
}

// ---------------------------------------------------------------------------
// the suite: one test method driving all cells (per-cell context in messages)
// ---------------------------------------------------------------------------

#[test]
fn xfixture_conformance() {
    let res_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/xfixtures");
    let manifest_path = res_dir.join("MANIFEST.tsv");
    // MUST fail (not skip) when absent: a missing manifest means the sync step
    // was never run for this checkout.
    let text = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "{} is missing or unreadable ({e}); run the xfixtures sync step",
            manifest_path.display()
        )
    });
    let m = parse_manifest(&text);

    let session = std::env::temp_dir().join(format!("mapdb5_xfix_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&session);
    std::fs::create_dir_all(&session).expect("create session dir");

    // gunzip every fixture file once; verify gz + raw SHA-256 and rawLen
    // BEFORE running any cell.
    let mut baselines: std::collections::BTreeMap<(String, String), PathBuf> =
        std::collections::BTreeMap::new();
    for fr in &m.files {
        let gz_path = res_dir.join(format!("{}.gz", fr.rel));
        let gz = std::fs::read(&gz_path)
            .unwrap_or_else(|e| panic!("fixture file {} missing: {e}", gz_path.display()));
        assert_eq!(
            sha256_hex(&gz),
            fr.gz_sha,
            "gzSha256 mismatch for {} ({})",
            fr.rel,
            fr.fixture
        );
        let mut raw = Vec::new();
        GzDecoder::new(gz.as_slice())
            .read_to_end(&mut raw)
            .unwrap_or_else(|e| panic!("gunzip {} failed: {e}", fr.rel));
        assert_eq!(
            raw.len() as u64,
            fr.raw_len,
            "rawLen mismatch for {}",
            fr.rel
        );
        assert_eq!(
            sha256_hex(&raw),
            fr.raw_sha,
            "rawSha256 mismatch for {} ({})",
            fr.rel,
            fr.fixture
        );
        let base_dir = session.join("baseline").join(&fr.fixture);
        std::fs::create_dir_all(&base_dir).unwrap();
        let base = base_dir.join(&fr.rel);
        std::fs::write(&base, &raw).unwrap();
        baselines.insert((fr.fixture.clone(), fr.rel.clone()), base);
    }

    // run every expect row addressed to this engine.
    let mut ran = 0usize;
    for (i, ex) in m.expects.iter().enumerate() {
        if ex.engine != "rust" {
            continue;
        }
        ran += 1;
        let ctx = format!(
            "cell {i}: fixture={} verdict={} opener={} placeAs={} openArg={}",
            ex.fixture, ex.verdict, ex.opener, ex.place_as, ex.open_arg
        );
        let files: Vec<&FileRow> = m.files.iter().filter(|f| f.fixture == ex.fixture).collect();
        // Stages 1+2: every fixture has exactly ONE file row.
        assert_eq!(files.len(), 1, "[{ctx}] fixture must have exactly one file");
        let baseline = &baselines[&(ex.fixture.clone(), files[0].rel.clone())];

        let cell = session.join(format!("cell-{i}"));
        std::fs::create_dir_all(&cell).unwrap();
        let working = cell.join(&ex.place_as);
        std::fs::copy(baseline, &working).unwrap();
        let snapshot = std::fs::read(&working).unwrap();
        let before = dir_entries(&cell);

        let recids: Vec<&RecidRow> = m
            .recids
            .iter()
            .filter(|r| r.fixture == ex.fixture)
            .collect();
        match (ex.verdict.as_str(), ex.opener.as_str()) {
            ("accept", "direct") => {
                let s = StoreDirect::open_file(&cell.join(&ex.open_arg))
                    .unwrap_or_else(|e| panic!("[{ctx}] accept cell failed to open: {e}"));
                run_accept(&s, &recids, &ctx);
            }
            ("accept", "wal") => {
                // openArg is the literal WAL file path within the cell dir
                // (`StoreWAL::open` takes the WAL FILE path itself).
                let s = StoreWAL::open(&cell.join(&ex.open_arg))
                    .unwrap_or_else(|e| panic!("[{ctx}] accept wal cell failed to open: {e}"));
                run_accept(&s, &recids, &ctx);
                // A `.ckpt` companion must NOT exist after a clean close; its
                // appearance is a failure (also enforced by the generic
                // new-files check below, which only allows `.lock`).
                assert!(
                    !dir_entries(&cell).iter().any(|f| f.ends_with(".ckpt")),
                    "[{ctx}] .ckpt companion must not exist after a clean close"
                );
            }
            ("reject", "direct") => match StoreDirect::open_file(&cell.join(&ex.open_arg)) {
                Err(DbError::DataCorruption(_)) => {}
                Err(other) => panic!("[{ctx}] expected DataCorruption, got: {other}"),
                Ok(s) => {
                    let _ = s.close();
                    panic!("[{ctx}] reject cell opened successfully");
                }
            },
            ("reject", "wal") => {
                // openArg is the literal WAL file path within the cell dir
                // (`StoreWAL::open` takes the WAL FILE path itself; the
                // manifest carries e.g. `x.wal`, matching placeAs).
                let wal_file = cell.join(&ex.open_arg);
                match StoreWAL::open(&wal_file) {
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

        // working copy must be byte-identical...
        let after = std::fs::read(&working).unwrap();
        assert_eq!(after, snapshot, "[{ctx}] working copy bytes changed");
        // ...and no new files may appear beyond the allowed sidecars, which are
        // enumerated (and excluded from the byte comparison above).
        let new_files: Vec<String> = dir_entries(&cell).difference(&before).cloned().collect();
        for f in &new_files {
            assert!(
                f.ends_with(".lock"),
                "[{ctx}] unexpected new file in cell dir: {f} (all new: {new_files:?})"
            );
        }
        std::fs::remove_dir_all(&cell).unwrap();
    }
    assert!(ran > 0, "manifest contains no expect rows for engine=rust");
    let _ = std::fs::remove_dir_all(&session);
}
