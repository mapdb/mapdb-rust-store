# PORTING-GAPS — mapdb5 Java to Rust

Deliberate deviations from the Java `org.mapdb` reference implementation
(<https://github.com/mapdb/mapdb-java-store>). These are intentional v1 choices,
not defects: each preserves correctness. Note the on-disk format is not
stabilised and no cross-implementation compatibility is guaranteed (see
`README.md`). Read this before depending on a behaviour that matters to
you — it is the honest limits list for this port.

## Both ports (serializer scope)
- **Skipped serializers:** `STRING_INTERN` (JVM string intern pool), `CLASS`, and
  `JAVA` (JVM object serialization) have no cross-language meaning and are not
  ported. Their catalog descriptor ids are never emitted.

## BTreeMap
- **Listener errors are `Result::Err`, not thrown exceptions.** A listener panic is
  caught (`catch_unwind`) and converted to a listener error, faithful to Java's
  catch-Throwable. When a secondary listener error is masked by a primary
  structural-propagation error, the secondary error is dropped — `DbError` has no
  suppressed-error/cause chain (Java attaches it as `suppressed`). Information loss
  only; the primary error is always surfaced.
- **Multi-record ops are not failure-atomic on non-transactional stores**
  (`StoreDirect`/heap): an I/O failure between a value record and the leaf-node
  write can orphan a value record. Java shares this for non-tx stores; WAL stores
  are atomic on commit. The counter now poisons its in-memory handle consistently
  on failure.
- **Re-entrant sync listeners deadlock** (parking_lot `RwLock` is not
  write→read reentrant) where Java's `ReentrantReadWriteLock` tolerates it. Both
  codebases forbid re-entrant listeners; documented.

## Queues (QueueLong, blocking queue)
- **Non-negative i64 domain on u64 fields.** `QueueLong` timestamp/value and
  blocking-queue capacity are `u64` restricted to `0..=i64::MAX`; a value
  `> i64::MAX` is rejected to preserve Java read-back. (the DB facade made the blocking-queue
  `open` validation enforce `capacity ∈ 1..=i64::MAX` and `size <= capacity`, with
  checked `remaining_capacity`/`dequeue` — the earlier wrap/panic gap is fixed and
  reachable via the DB facade.)
- **No thread interruption.** Blocking `take`/`put` use timeout methods plus a
  close/shutdown flag returning `StoreClosed`; there is no `InterruptedException`.
  Blocking coordinates only threads sharing the live handle (not cross-process).
- **Re-entrant callback into the same queue handle errors** (thread-scoped guard)
  instead of deadlocking (parking_lot / std locks are non-reentrant vs Java
  reentrant `synchronized`).
- **`QueueLong.printContent`** (debug-only) not ported. Error taxonomy: pure
  argument errors use `DbError::corrupt` (no `InvalidArg` variant);
  `AbstractQueue.add`-when-full maps to `Unsupported`.

## DB facade
- **`maxNodeSize` upper bound.** The port caps `maxNodeSize` at `1<<20` at BOTH
  create and reopen, so a create can never persist a value the catalog validator
  would later reject (which would brick the whole DB). Java (`BTreeMap`) has NO
  upper bound, so a hypothetical Java-written file with `maxNodeSize > 1<<20` is
  not openable by this port (validated as corruption). Lower bound `>= 4` matches
  Java.
- **`delete` reclaims little of a collection's storage.** For a map/set opened in
  this session, teardown runs the cached handle's `clear()`, which removes entries
  (freeing external-value records) but leaves the empty B-tree **node** records and
  the structural `rootRecidRecid`/`counterRecid` allocated; a map/set never opened
  in this session has no cached handle, so its entries are not even cleared. Store
  `compact` copies live records and is NOT reachability GC, so none of these
  orphaned records are reclaimed until a full store rebuild. This matches Java
  (`obj.clear()` leaves structural records; the typed Rust API additionally cannot
  reconstruct an unopened collection's concrete type from catalog text). The
  critical invariant — `rootRecidRecid`/`counterRecid` are never *freed*, so a live
  handle clone can never write a reused recid (delete-corruption safety) — holds
  regardless.
- **A deleted atomic's stale handle is a use-after-delete hazard (Java parity).**
  Java `DB.delete` frees an atomic's single record (`store.delete(atomicRecid)`),
  and this port does the same; the cached handle is dropped, but an external
  `Atomic*` clone the caller still holds keeps that recid. Because the store
  immediately reuses freed recids, using the stale clone after delete can read or
  overwrite whatever object next claims the recid. This is the documented
  raw-handle-after-delete contract (identical in Java): do not use a handle to a
  named object after deleting it. Maps/sets avoid the analogous corruption only
  because their structural recids are deliberately leaked (above).
- **Persistent `Bind` targets are partial.** `secondary_value`/`secondary_key`/
  `secondary_keys`/`secondary_values`/`map_inverse`/`map_put_after_delete` accept
  any `SecMap`/`SecSet` target including a persistent `BTreeMap`. Still
  in-memory-secondary-only: `histogram` (its single-lock atomic count-update would
  need an atomic-compute method on the trait for a race-free persistent version)
  and the one-to-many *set* indexes over a persistent `NavigableSet` (would need a
  tuple `GroupFormat` for the `(derived, key)` element). Java accepts any
  persistent `Map`/`Set` here; add if a consumer needs it.
- **Bare `CUSTOM` descriptor matches any custom codec.** A configurable custom
  codec persists an opaque `CUSTOM` marker with no configuration fingerprint, so
  two differently-configured instances of one codec are indistinguishable on
  reopen (a cache/descriptor check accepts either). Java's `CUSTOM:<fqcn>` is only
  marginally stronger (class identity, not config). A caller-supplied stable
  descriptor capturing the wire-relevant configuration would close this (API
  change); v1 documents the limitation and never claims cross-language reopen
  compatibility for `CUSTOM` codecs.
- **Untyped `db.get(name)`** is not ported (monomorphization forbids recovering a
  caller-chosen type from catalog text); typed opens replace it. **`ArraySerializer`
  (`ARRAY:`)** is treated as a custom codec (re-supply required). **Out of scope:**
  HTreeMap/hashMap/hashSet, expiry, sharded variants, SortedTableMap makers,
  IndexTreeList, JVM shutdown hooks (accepted as no-ops), TxMaker. `QueueLong`
  stays a direct store primitive with no catalog type.
- **`NavigableSet` has no `comparator()` accessor.** Java `NavigableSet.comparator()`
  exposes the (possibly custom) key comparator, returning `null` for natural order.
  In this port key order is fixed by the key `GroupFormat` (D1/D2 monomorphization);
  there is no runtime comparator to return. The full navigation surface
  (`lower`/`floor`/`ceiling`/`higher`, `poll_first`/`poll_last`, `descending_set`,
  and live `sub_set`/`head_set`/`tail_set` views over the backing map's `RangeView`)
  IS ported; only the comparator handle is omitted.
- **Descending iteration is a streaming retained-path walk, not Java's
  per-step re-descent.** `descending_entry_iter` (spec 03 §7 second cut)
  buffers one leaf's in-range entries per step and re-descends only from the
  deepest still-covering retained dir frame — O(leaf) memory, O(1) amortized
  node loads per leaf, and it yields `Result` items (Java's iterator throws).
  `poll_last`/`floor`/`lower` ride it, replacing the earlier O(range)
  ascending scan-keep-last. Same weak consistency as ascending iteration; a
  corrupt tree surfaces as `DataCorruption("descending scan bound did not
  decrease")` instead of looping.
- **`memory_db()` and `memory_direct_db()`** both map to an in-memory
  `StoreDirect` (no separate on-/off-heap direct distinction).
- **rollback of external handles** (map/set only): a D12 lease is held only by
  maps/sets, so an old map/set handle stays memory-safe but is not reusable — D12
  returns `AlreadyOpen` for a fresh independent open until the old clone drops
  (external Rust clones cannot be force-dropped). Queues are closed by their cache
  hook and can be reopened; atomics have neither a lease nor invalidation, so an
  old atomic clone can coexist with a fresh handle after rollback (same stale-recid
  hazard as delete, above).

## External crash tier (`ci/crash/`, `tools/crash-harness/`)

Ported from the io-uring engine's crash harness. The harness is a separate
workspace member, not part of the `mapdb-rust-store` library. Deliberate adaptations to
mapdb5's durability model:

- **WAL only.** The source harness crash-tested both a copy-on-write `Direct`
  backend and a WAL backend, because both were power-cut recoverable. mapdb5's
  `StoreDirect` is non-transactional: `commit()` fsyncs data then stamps a header
  checksum over the allocator words, and `open_file` *rejects* a store whose
  recomputed checksum disagrees ("store was not closed cleanly"). Any uncommitted
  in-place mutation bumps those words, so after a crash at a random cut point a
  reopen refuses rather than recovering to the last committed state — it can never
  satisfy a recover-to-acked-state oracle. Only `StoreWAL` (log replay, torn-tail
  tolerant, `.ckpt` crash-during-checkpoint recovery) is exercised. A separate
  graceful-restart consistency test could cover Direct; it is not a crash test.
- **No transaction id → committed progress marker.** The source read
  `metrics().visible_txid` to learn the exact recovered txid and compared contents
  to `replay(visible_txid)`. mapdb5 exposes no recovery watermark, so the harness's
  own 1-based sequence number is the txid and the workload commits once per group;
  committed states are therefore exactly the group boundaries. The workload also
  commits a `!crash-committed-seq` marker (the group's last seq) atomically inside
  every group, so the recovered store *carries* the boundary it reached — the
  checker reads it directly rather than guessing from possibly-aliased contents
  (necessary because cancelling ops, e.g. a 1-key universe, make different
  boundaries share a logical map; guessing could hide a lost acked group). The
  boundary must be a multiple of `group` in `[max_ack, floor(max_intent/group)*group]`
  (`g >= max_ack` is durability, the upper bound is the completed-intent prefix),
  and the recovered contents must byte-exactly equal `replay(g)` checked two
  independent ways (whole-map scan + per-key point lookup). The checker also runs
  an ordered ACK/group state-machine validator over the journal (each ACK the
  exact next boundary, maintenance only between acked groups, ≤1 in-flight group)
  so a *workload* protocol regression is caught independently of the store.
  **Residual (documented, not a defect):** mapdb5 has no B-tree structural
  verifier equivalent to the source's `map.verify()`, so structural
  (separator/routing/reachability) coverage rests on the dual logical read path +
  `size_long()` + the allocator/index/free-list `Store::verify()`; an
  allocator-consistent unreachable record is not independently rejected. The
  privileged tier's device teardown and the always-on CI gate are likewise open
  follow-ups (the campaign workflow is weekly/on-demand only).
- **checkpoint == compact.** `StoreWAL::compact` *is* a log clean,
  so the workload emits only `compact` maintenance records and the readiness policy
  requires `groups>=3 && compactions>=1` (the source's WAL-only `checkpoints>=1`
  collapses into it).

Build/run: `cargo build -p mapdb-rust-store-crash-harness --bins`; the
unprivileged SIGKILL smoke is `ci/crash/crash-smoke.sh`; the privileged
dm-suspend power-cut tier is `ci/crash/crash-tier.sh --fs ext4|xfs …` (root); the
weekly depth campaign is `.github/workflows/crash-campaign.yml`.

## What "byte-for-byte with Java" means in this source

Comments throughout name a value encoding as byte-for-byte or byte-compatible
with Java's — the packed varints, the name catalog, the codec descriptor
strings, the queue node records, the serializer families. Those statements are
narrow and they are tested: each is pinned by golden vectors taken from the
encoding it was ported from.

They are **not** a statement that a store file interoperates. A store file is
those codecs plus a header, an allocator layout, a WAL framing and a recovery
protocol. The WAL half of that no longer diverges by DESIGN — this engine
adopted the Java engine's segmented WAL format v3 — but note what does and does
not yet back that up: the format is pinned byte-for-byte against a
Java-written segment in `wal_recover.rs`'s test kit, while the cross-engine
FIXTURES do not pin v3 yet. Stage C is what supplies the v3 namespace bundles;
until it lands, the four v1 WAL fixture cells are skipped and no fixture
exercises a v3 namespace. The store file's own header and allocator layout are
a separate question
that those fixtures answer per case. A per-codec fidelity claim says the bytes
of one value match; it says nothing about whether another engine can open the
file those bytes live in. It cannot.
