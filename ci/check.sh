#!/usr/bin/env bash
# The baseline gate, runnable with no GitHub in the loop. Run before every
# push/merge; .github/workflows/ci.yml runs the same jobs (gate, crash-smoke,
# package, audit) on every push once the commit reaches the remote. Any failure
# fails the gate (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."

# cargo resolves `fmt`/`clippy` through $CARGO_HOME/bin before $PATH; on hosts
# where that directory holds only toolchain-less rustup shims the resolution
# errors out, so fall back to the system cargo-fmt/cargo-clippy binaries.
cargo_sub() {
  local sub=$1
  shift
  if cargo "$sub" --version >/dev/null 2>&1; then
    cargo "$sub" "$@"
  else
    "cargo-$sub" "$@"
  fi
}

echo "== fmt =="
cargo_sub fmt --check

echo "== clippy (default) =="
cargo_sub clippy --locked --all-targets -- -D warnings

echo "== clippy (crash harness workspace member) =="
# The harness is a separate, unpublished workspace member; a root --all-targets
# pass does not reach it.
cargo_sub clippy --locked --all-targets -p mapdb-rust-store-crash-harness -- -D warnings

echo "== test =="
cargo test --locked

echo "== process-crash-smoke (unprivileged crash tier) =="
# SIGKILL + reopen through the real syscall path. Not a power-cut claim — that
# is the scheduled crash campaign.
cargo build --locked -p mapdb-rust-store-crash-harness --bins
ci/crash/crash-smoke.sh

echo "== package (what a consumer actually receives) =="
pkg_list="$(mktemp)"
trap 'rm -f "$pkg_list"' EXIT
cargo package --locked --list > "$pkg_list"
for f in README.md NOTICE.md LICENSE-EPL-1.0.txt LICENSE-EDL-1.0.txt; do
  grep -qx "$f" "$pkg_list" || { echo "missing from crate: $f"; exit 1; }
done
! grep -q '^tools/' "$pkg_list" || { echo "crash harness leaked into the crate"; exit 1; }
cargo package --locked

echo "== audit (RustSec advisories vs committed lock) =="
if command -v cargo-audit >/dev/null 2>&1; then
  cargo audit
else
  echo "cargo-audit not installed (cargo install cargo-audit --version =0.22.2 --locked)" >&2
  exit 1
fi

echo "== gate PASSED =="
