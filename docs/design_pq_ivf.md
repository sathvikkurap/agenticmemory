# Design: PQ/IVF Index Backends (Future)

**Status:** Design only — not implemented  
**Related:** [design_roadmap.md](design_roadmap.md), [design_pluggable_index.md](design_pluggable_index.md)

## Goal

Support Product Quantization (PQ) and Inverted File (IVF) index backends for:
- **Larger scale:** Millions of episodes with sub-linear query time
- **Memory efficiency:** PQ compresses vectors; IVF reduces memory vs full HNSW
- **Trade-offs:** Lower recall than HNSW for same latency; tunable quality/speed

## When to Use

| Backend | Scale | Query | Memory | Recall |
|---------|-------|-------|--------|--------|
| Exact | <10k | O(n) | O(n·d) | 100% |
| HNSW | 10k–1M+ | O(log n) | O(n·d·M) | ~95%+ |
| IVF+PQ | 100k–10M+ | O(√n) approx | O(n·m) compressed | ~80–95% tunable |

## Approach

### Option A: IVF-Flat

- Cluster vectors into `nlist` centroids (e.g. k-means).
- Query: find nearest `nprobe` clusters, search only those vectors.
- No compression; simpler than PQ.
- **Crates:** `faer` for k-means, or integrate `usearch`, `hnswlib`-style IVF.

### Option B: IVF-PQ

- IVF for coarse filtering + PQ for compressed storage.
- PQ: split vector into `m` sub-vectors, quantize each to `k` centroids (e.g. 256).
- **Crates:** `instant-distance` (PQ), or hand-roll with `faer`.

### Option C: Integrate Existing

- `usearch` (Rust) — supports IVF, HNSW, various metrics.
- `qdrant` / `milvus` — full vector DBs; overkill for in-process.
- Prefer lightweight crates that match our `insert`/`search` interface.

## Interface

Extend `IndexBackend`:

```rust
enum IndexBackend {
    Hnsw(HnswIndex),
    Exact(ExactIndex),
    IvfPq(IvfPqIndex),  // future
}
```

`IvfPqIndex` would implement same `insert(&[f32]) -> usize` and `search(&[f32], k) -> Vec<(usize, f32)>` as today.

## Persistence

- IVF: persist centroids + assignments.
- PQ: persist codebook + codes.
- Checkpoint format TBD; likely separate from ExactIndex JSON.

## Out of Scope (for now)

- Distributed / sharded indexes
- GPU acceleration
- Mixed-precision (fp16) search
