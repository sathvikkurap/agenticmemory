# Design: Pluggable Index Backends

**Status:** Implemented (initial slice)  
**Related:** [design_roadmap.md](design_roadmap.md)

## Goal

Support multiple vector index backends so users can choose:
- **HNSW** (default): Fast approximate NN, good for large episode sets.
- **Exact**: Brute-force L2 search, correct for small sets or correctness-critical use.
- **Future:** PQ, IVF, disk-backed indexes.

## Design

### Internal Abstraction

We use an enum (not a trait object) to avoid dynamic dispatch overhead and keep the API simple:

```rust
enum IndexBackend {
    Hnsw(HnswIndex),
    Exact(ExactIndex),
}
```

Each variant wraps the underlying implementation. `AgentMemDB` holds `IndexBackend` and delegates `insert`/`search` to it.

### ExactIndex

- Stores vectors in a `Vec<Vec<f32>>`.
- `insert(vec)` appends and returns the index.
- `search(query, k)` computes L2 distance to all vectors, sorts, returns top-k indices.
- O(n) per query; suitable for n < ~10k.

### HnsqIndex (existing)

- Wraps `hnswx::HNSW<EuclideanDistance>`.
- Same `insert`/`search` interface.
- O(log n) approximate; suitable for n > 1k.

### Public API

**Backward compatible:** `AgentMemDB::new(dim)` and `AgentMemDB::new_with_max_elements(dim, n)` unchanged; both use HNSW.

**New constructors:**
- `AgentMemDB::new_exact(dim)` — uses exact search. For small episode sets or when correctness is critical.

**Future:** `AgentMemDB::new_with_backend(dim, backend)` when we add more backends.

### Persistence

Save/load format is unchanged. Episodes are serialized as JSON; on load, we rebuild the index. The backend choice is not persisted (we default to HNSW on load). Future: persist backend type in metadata if needed.

## Implementation Notes

- `ExactIndex` uses Euclidean (L2) distance to match HNSW.
- Both backends return internal indices; `AgentMemDB` maps these to UUIDs via `key_to_uuid`.
- Python bindings: `AgentMemDB.exact(dim)` (classmethod) — implemented.

## Migration

No migration needed. Existing code continues to use `AgentMemDB::new(dim)` with HNSW.
