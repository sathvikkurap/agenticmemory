# Design: HNSW Checkpoint (Future)

**Status:** Design only — not implemented  
**Related:** [design_roadmap.md](design_roadmap.md), [design_incremental_indexing.md](design_incremental_indexing.md)

## Goal

Extend AgentMemDBDisk checkpoint to HNSW backend, enabling fast restart for large episode sets (10k–1M+) that use HNSW instead of ExactIndex.

## Current State

- **ExactIndex checkpoint:** Implemented. `exact_checkpoint.json` + `meta.json` → skip replay on open.
- **HNSW:** No checkpoint. AgentMemDBDisk with HNSW always does full log replay on open.
- **hnswx crate:** Does not expose serialize/deserialize. Internal structure is opaque.

## Options

### Option A: Upstream hnswx Serialization

- Contribute `save`/`load` or `Serialize`/`Deserialize` to hnswx.
- **Pros:** Clean, maintained upstream.
- **Cons:** Depends on maintainer acceptance; may take time.

### Option B: Fork hnswx

- Fork, add bincode/serde persistence.
- **Pros:** Full control.
- **Cons:** Maintenance burden; drift from upstream.

### Option C: Minimal Persistence Layer

- Extract vectors + graph structure from HNSW via public API (if any).
- Rebuild HNSW from persisted vectors on load (batch insert).
- **Pros:** No fork; works with current hnswx.
- **Cons:** hnswx may not expose enough; rebuild is O(n log n), not instant load.

### Option D: Switch to Crate with Persistence

- Evaluate `usearch`, `hnsw_rs`, or others that support save/load.
- **Pros:** May get persistence + other features.
- **Cons:** API migration; possible quality/performance differences.

## Recommendation

**Short term:** Keep ExactIndex checkpoint for disk-backed DBs that need fast restart. Use HNSW for in-memory or when replay time is acceptable.

**When implementing:** Check hnswx releases for new serialization support. If none, evaluate Option C (rebuild from persisted vectors) or Option D (alternative crate).

## Interface (when implemented)

Same as ExactIndex checkpoint:

- `DiskOptions::hnsw_with_checkpoint(dim, max_elements)` — HNSW + checkpoint enabled
- `checkpoint()` — persist `hnsw_index.bin` (or equivalent)
- On open: load from checkpoint if valid, else replay

## Out of Scope

- Incremental index updates (replay-only new lines) — complex for HNSW graph
- Cross-version checkpoint compatibility
