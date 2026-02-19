# PERFORMANCE.md

## AgentMemDB Phase 0: Performance Notes

### Insert Complexity
- Each episode insert updates a HashMap (O(1)) and the HNSW index (amortized O(log n) per insert).
- HNSW insert cost grows slowly with the number of episodes.

### Query Complexity
- HNSW approximate nearest neighbor search: O(log n) per query (sublinear, but not constant).
- Filtering by reward is O(k), where k is the number of candidates returned by HNSW.

### Expected Performance (Mac M3, 10k episodes)
- Insert: < 1 ms per episode (in-memory, no disk IO).
- Query: ~1-10 ms for top-10 nearest neighbors (depends on HNSW params and embedding dim).
- Memory: ~30MB for 10k episodes with 768-dim embeddings.

### Notes
- All operations are single-threaded in Phase 0.
- For larger scales, consider sharding, disk persistence, and parallelism.
- HNSW parameters (M, ef_construction, ef_search) can be tuned for speed/recall tradeoff.

// Add benchmarks with `criterion` in future phases.
