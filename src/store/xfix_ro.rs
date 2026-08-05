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
use std::path::Path;

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
