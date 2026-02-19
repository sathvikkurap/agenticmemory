# Design: On-Disk / Hybrid Storage

**Status:** First slice implemented  
**Related:** [design_roadmap.md](design_roadmap.md)

## Goal

Support episode sets that do not fit entirely in RAM. Enable:
- Append-only persistence so writes are durable without full snapshots.
- Index rebuild on load (or incremental indexing for future).
- Optional compaction to reclaim space from deleted/expired episodes.

## Constraints

- **Backward compatible:** Existing `AgentMemDB` (in-memory) and `save_to_file`/`load_from_file` remain unchanged.
- **Vertical slice:** First increment delivers a working `AgentMemDBDisk` (or similar) that uses disk for episode storage; index stays in RAM.
- **Simple first:** Append-only log + full index rebuild on open. Defer incremental indexing and compaction.

## Architecture

### Option A: Separate `AgentMemDBDisk` Type

Introduce a new type that:
- Stores episodes in an append-only log (one JSON line per episode, or a binary format).
- Keeps the vector index (HNSW or Exact) in RAM.
- On open: read log, rebuild index in memory.
- On write: append to log, insert into index.

**Pros:** Clear separation; no risk to existing `AgentMemDB`.  
**Cons:** Two types to maintain; some code duplication.

### Option B: Storage Backend Abstraction

Add a `StorageBackend` enum (InMemory, AppendOnlyLog) and make `AgentMemDB` generic over it.

**Pros:** Single type, unified API.  
**Cons:** More refactoring; trait/impl complexity.

### Recommendation: Option A

Start with `AgentMemDBDisk` as a distinct type. Keeps the in-memory path simple and allows us to iterate on disk format without touching core logic.

## Append-Only Log Format

### Format: JSONL (JSON Lines)

One episode per line. Easy to append, human-readable, debuggable.

```
{"id":"...","task_id":"...","state_embedding":[...],"reward":0.5,"metadata":null,"steps":null}
{"id":"...","task_id":"...","state_embedding":[...],"reward":0.8,"metadata":null,"steps":null}
```

**Alternatives considered:**
- Binary (bincode, msgpack): Smaller, faster parse. Defer until we need it.
- SQLite: Overkill for append-only; adds dependency.

### File Layout

```
<data_dir>/
  episodes.jsonl     # Append-only log
  meta.json          # dim, index_type, created_at (optional)
```

On first write, create `meta.json` with `dim`, `index_type` (hnsw|exact), `max_elements` (for HNSW).

## API (First Slice)

```rust
/// Disk-backed agent memory DB. Episodes stored in append-only log; index in RAM.
pub struct AgentMemDBDisk {
    dim: usize,
    index: IndexBackend,
    key_to_uuid: HashMap<usize, Uuid>,
    episodes: HashMap<Uuid, Episode>,  // Cache in RAM for now; future: page or lazy load
    path: PathBuf,
    log_file: File,  // Append handle
}

impl AgentMemDBDisk {
    /// Open or create a disk-backed DB at the given directory.
    pub fn open(path: impl AsRef<Path>, dim: usize) -> Result<Self, AgentMemError>;
    
    /// Open with options (max_elements for HNSW, exact vs HNSW).
    pub fn open_with_options(path: impl AsRef<Path>, opts: DiskOptions) -> Result<Self, AgentMemError>;
    
    // store_episode, query_similar — same signature as AgentMemDB
    // Each store appends to log and updates index.
}
```

**First slice:** Episodes stay in RAM (HashMap) for simplicity. The log is the source of truth on load; we replay it into memory + index. Future: evict cold episodes to disk, keep hot set in RAM.

## Load Sequence

1. Read `meta.json` if exists → get `dim`, `index_type`, `max_elements`.
2. If no meta: create new DB with provided `dim` and defaults.
3. Read `episodes.jsonl` line by line; deserialize each line as `Episode`.
4. Build index and `key_to_uuid` by calling `store_episode` logic (without writing to log).
5. Open log file in append mode for subsequent writes.

## Write Sequence

1. Append serialized episode (one JSON line) to `episodes.jsonl`.
2. Insert into index and `episodes` HashMap.
3. Flush log file (or use `write_all` with implicit flush; consider explicit `sync_all` for durability).

## Compaction (Future)

When we add deletion or retention:
- Write a new `episodes.jsonl` with only retained episodes.
- Atomically replace (e.g., rename temp file over original after fsync).
- Rebuild index from new log.

## Retention and Pruning (Future)

- Time-based: drop episodes older than T.
- Importance-based: drop low-reward episodes when over capacity.
- Hooks for user-defined policies.

Design these in a follow-up doc once the base disk storage works.

## Implementation Plan

| Increment | Scope | Deliverable |
|-----------|-------|-------------|
| 1 | Design | This doc |
| 2 | AgentMemDBDisk + JSONL log | New type, open/store/query, tests |
| 3 | Python bindings | AgentMemDBDisk.open(path, dim) |
| 4 | Compaction (optional) | Defer |
| 5 | Retention policies | Defer |

## Open Questions

- **Flush policy:** Flush on every write vs batch. Start with flush per write for durability.
- **Concurrent access:** Single writer assumed. Multiple readers could read log + index snapshot; defer.
- **Corruption recovery:** If log is truncated, we may have partial episode. Consider checksums or length-prefixed records later.
