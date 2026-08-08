//! Cross-process WAL lock probe for Stage C C8x (`wal3-c8-plan.md` §3).
//!
//! Crate-internal so RO reaches [`super::wal::StoreWAL::open_cfg`] /
//! [`WalOptions::read_only`] without a public API. Invoked by the orchestrator
//! as the libtest executable with `--exact --ignored --nocapture` and the
//! `MAPDB_LOCK_PROBE_*` environment protocol.

use super::wal::{StoreWAL, WalOptions};
use crate::error::DbError;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

fn env_req(k: &str) -> String {
    env::var(k).unwrap_or_else(|_| panic!("missing env {k}"))
}

fn open_store(base: &Path, mode: &str) -> Result<StoreWAL, DbError> {
    match mode {
        "rw" => StoreWAL::open(base),
        "ro" => StoreWAL::open_cfg(
            base,
            WalOptions {
                read_only: true,
                ..Default::default()
            },
        ),
        other => panic!("mode must be rw|ro, got {other}"),
    }
}

fn hold(base: &Path, mode: &str, ready: &Path, release: &Path) {
    assert!(
        !ready.exists() && !release.exists(),
        "ready/release must be initially absent"
    );
    let store = open_store(base, mode).unwrap_or_else(|e| panic!("hold open failed: {e}"));
    fs::write(ready, b"ready\n").expect("write ready");
    println!("HOLD_READY");
    let deadline = Instant::now() + Duration::from_secs(30 * 60);
    while !release.exists() {
        if Instant::now() > deadline {
            panic!("release file never appeared: {}", release.display());
        }
        thread::sleep(Duration::from_millis(20));
    }
    drop(store);
}

fn open_cmd(base: &Path, mode: &str) {
    match open_store(base, mode) {
        Ok(s) => {
            drop(s);
            println!("OK");
        }
        Err(DbError::Locked(_)) => println!("REFUSED"),
        Err(e) => println!("OTHER:{e:?}:{e}"),
    }
}

/// Orchestrator entry: only runs when `MAPDB_LOCK_PROBE_CMD` is set.
///
/// ```text
/// MAPDB_LOCK_PROBE_CMD=open MAPDB_LOCK_PROBE_BASE=… MAPDB_LOCK_PROBE_MODE=rw \
///   target/debug/deps/mapdb_store-… --exact --ignored --nocapture wal3_lock_probe
/// ```
#[test]
#[ignore = "C8x lock probe CLI; invoked by lock_matrix.py"]
fn wal3_lock_probe() {
    let cmd = match env::var("MAPDB_LOCK_PROBE_CMD") {
        Ok(c) => c,
        Err(_) => return, // not a probe invocation
    };
    let base = PathBuf::from(env_req("MAPDB_LOCK_PROBE_BASE"));
    let mode = env_req("MAPDB_LOCK_PROBE_MODE");
    match cmd.as_str() {
        "hold" => {
            let ready = PathBuf::from(env_req("MAPDB_LOCK_PROBE_READY"));
            let release = PathBuf::from(env_req("MAPDB_LOCK_PROBE_RELEASE"));
            hold(&base, &mode, &ready, &release);
        }
        "open" => open_cmd(&base, &mode),
        other => panic!("cmd must be hold|open, got {other}"),
    }
}
