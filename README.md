# mapdb-rust-store

A Rust port of the **MapDB5** storage engine: a record store with write-ahead
logging and crash recovery, a B+tree map and a hash-tree map over it, the MapDB
serializer family, and the `DB`/`DBMaker` facade.

> **The on-disk format is not stabilised.** There is no compatibility guarantee —
> not between the Java, Rust and Zig implementations, and not between versions of
> any one of them. A file written by one engine may not open under another, or
> under a later build of the same engine. Implementers may change the format
> freely and without notice; there is no migration path and none is planned.
> Do not put data you care about in it.

The reference implementation is the Java engine at
<https://github.com/mapdb/mapdb-java-store>; there is a third, independent port
in Zig at <https://github.com/mapdb/mapdb-zig-store>.

Not to be confused with <https://github.com/mapdb/mapdb-rust-iouring>, which is a
different, asynchronous, Linux-first engine with its own on-disk format.

## Support status

**This is not a supported release.** It has never been published to crates.io,
the API has no stability guarantee, and it has not been run in production. It is
a reviewed, tested port of a defined subset of the Java engine — which is a
different claim from "ready".

[`PORTING-GAPS.md`](PORTING-GAPS.md) is the honest list: every deliberate
deviation from the Java reference, per area, with the reason. Read it before
depending on any behaviour that matters to you. The headline ones:

- Multi-record operations are **not failure-atomic** on non-transactional stores
  (`StoreDirect`, heap): an I/O failure mid-operation can orphan a value record.
  WAL stores are atomic on commit. Java shares this limitation.
- Re-entrant synchronous listeners **deadlock** here where Java's
  `ReentrantReadWriteLock` tolerates them. Both codebases forbid them.
- Reads take locks; there is no optimistic or seqlock read path. The Java
  engine's `StampedLock` optimistic reads race in-place record writes, which is
  undefined behaviour in Rust, so a locked read is the only sound transcription.
- The `STRING_INTERN`, `CLASS` and `JAVA` serializers are not ported — they have
  no cross-language meaning.

## Requirements

- A stable Rust toolchain (2021 edition). No nightly features.
- Linux or macOS. The crash-tier scripts under `ci/crash/` are Linux-only and
  the privileged tier needs root, device-mapper and `xfsprogs`.

## Build and test

```sh
cargo build --locked
cargo test  --locked            # the unit and integration suites
```

`Cargo.lock` is committed and CI builds `--locked`; do that locally too if you
want the graph the tests were run against.

### Crash tier

Durability is tested out of process, not only by unit tests:

```sh
ci/crash/crash-smoke.sh                                   # unprivileged: SIGKILL + reopen
sudo ci/crash/crash-tier.sh --fs ext4 --rounds 3 \
     --workload $PWD/target/debug/crash_workload \
     --checker  $PWD/target/debug/crash_check \
     --work /tmp/crash-tier                               # privileged: real power cut
```

The privileged tier cuts power by suspending a device-mapper target
(`dmsetup suspend --noflush --nolockfs`) and copying the backing file while
suspended, so everything still in the dirty page cache is genuinely lost. The
oracle is a write-ahead intent journal on a *different* filesystem: intents are
`fdatasync`'d before the write is enqueued and an ACK is journalled only after
the backend's durability barrier returns, so a recovered state that lost an
acknowledged write fails the round.

**Only `StoreWAL` is crash-recoverable.** `StoreDirect` is non-transactional: an
uncommitted in-place mutation makes a reopen fail the header checksum by design,
so at a random cut point it cannot recover to the last acknowledged state. The
crash tier therefore exercises WAL only, and that is not an oversight.

The tooling lives in `tools/crash-harness/`, a separate workspace member that is
never published and is not part of the `mapdb-rust-store` library.

## What this does not claim

The crash model covers dirty-page-cache loss, write and sync ordering, ext4/XFS
journal recovery, and the WAL replay and checkpoint protocols. It does **not**
model a lying volatile device cache, FUA or flush dishonesty, sector tearing, or
a filesystem that violates powersafe overwrite.

## Layout

```
src/
  io.rs        DataInput2/DataOutput2, packed varints
  error.rs     DbError taxonomy
  ser/         serializers and group formats
  store/       Store trait, StoreOnHeap, StoreByteArray, StoreDirect, StoreWAL
  btree/       BTreeMap, node, view, pump
  db/          the DB/DBMaker facade, name catalog, atomics
  queue/       QueueLong and the blocking queue
  listener.rs  modification listeners
tests/         integration tests
tools/crash-harness/
               destructive crash-tier tooling (separate workspace member)
ci/crash/      crash-tier scripts
```

## License

Dual EPL-1.0 / EDL-1.0 (`SPDX-License-Identifier: EPL-1.0 OR BSD-3-Clause`).
See [`LICENSE-EPL-1.0.txt`](LICENSE-EPL-1.0.txt),
[`LICENSE-EDL-1.0.txt`](LICENSE-EDL-1.0.txt) and [`NOTICE.md`](NOTICE.md).
