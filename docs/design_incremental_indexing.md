# Design: Incremental Indexing and Compaction for AgentMemDBDisk

**Status:** First slice implemented (ExactIndex checkpoint)  
**Related:** [design_roadmap.md](design_roadmap.md), [design_on_disk_storage.md](design_on_disk_storage.md)

## Goal

Reduce AgentMemDBDisk open time for large logs by avoiding full replay. Support efficient compaction without full index rebuild.

## Current State

- **Open:** Replay entire `episodes.jsonl` line-by-line; rebuild index from scratch. O(n) in episode count.
- **Store:** Append to log, insert into index. O(1) append, O(log n) HNSW insert.
- **Prune:** Rebuild index + rewrite log (compaction). Already implemented.
- **Load time:** Dominated by replay for large logs (e.g. 100k+ episodes).

## Use Cases

- **Fast restart:** Agent process restarts; DB has 50k episodes. Current: ~5–10s replay. Target: <1s via checkpoint.
- **Compaction without full rebuild:** Prune 10% of episodes; avoid rebuilding index for the 90% retained.

## Approach

### Option A: Index Checkpoint

Persist the index (HNSW or Exact) to disk periodically or on close.

- **Checkpoint file:** `index.bin` (or `index.json` for Exact) — serialized index state.
- **Log watermark:** `meta.json` records `replayed_up_to_byte` or `replayed_up_to_line`.
- **Open sequence:**
  1. If `index.bin` exists and log hasn't grown: load index, skip replay.
  2. If log grew: load index, replay only new lines since checkpoint.
  3. If no checkpoint or incompatible: full replay (current behavior).

**Pros:** Fast load when checkpoint is recent.  
**Cons:** HNSW serialization — hnswx may not support; Exact is trivial. Checkpoint invalidation on prune.

### Option B: Lazy / Streaming Replay

Replay in chunks, yield to caller between chunks. Doesn't reduce total work but improves perceived latency.

**Pros:** Simple.  
**Cons:** Doesn't reduce open time.

### Option C: Compaction-Only Optimization

When pruning, we already rewrite the log. The index rebuild is unavoidable (we're changing the episode set). No change needed.

**Conclusion:** Compaction is done. The win is checkpoint for fast restart.

### Recommendation

**First slice (implemented):** Optional index checkpoint for **ExactIndex** only.

- `DiskOptions::exact_with_checkpoint(dim)` — exact index with checkpoint enabled.
- `AgentMemDBDisk::open_with_options(path, opts)` — when `use_checkpoint` and exact backend, reads `exact_checkpoint.json` if valid.
- `checkpoint()` — writes `exact_checkpoint.json`, updates `meta.json` with `checkpoint_line_count`. Call after stores.
- On open: if `exact_checkpoint.json` exists and `meta.checkpoint_line_count == log_line_count`, load from checkpoint, skip replay. Else full replay.
- On prune: checkpoint file is removed (index invalidated).

**Future:** HNSW checkpoint if hnswx adds serialization, or we implement a minimal persistence layer.

## Backward Compatibility

- Checkpoint is optional. Default: no checkpoint (current behavior).
- Old DBs without checkpoint files: full replay. No migration needed.

## Out of Scope

- Concurrent writers
- Distributed / sharded storage
- Binary log format (stay with JSONL for now)
