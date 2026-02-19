# How to Size and Tune for Your Workload

## Episode Count

- **Default:** `AgentMemDB::new(dim)` supports up to 20,000 episodes.
- **Scale:** Use `AgentMemDB::new_with_max_elements(dim, n)` or `AgentMemDB.with_max_elements(dim, n)` (Python) for 50k–100k+ episodes.
- Set `max_elements` to ~1.2× your expected peak to avoid reallocation.

## Embedding Dimension

Match your embedding model (e.g., 384 for all-MiniLM, 768 for BERT-base, 1536 for OpenAI).

## Query Latency

- 10k episodes: ~200 µs per query (see [BENCHMARKS.md](../BENCHMARKS.md))
- 50k–100k: expect ~500 µs–2 ms depending on hardware
- Increase `ef_search` for higher recall at the cost of latency (HNSW config; not yet exposed in AgentMemDB)

## Insert Throughput

- ~11k inserts/sec at 10k episodes (Apple M3)
- Batch inserts via `store_episodes`; single-threaded
- For bulk load, consider building the index once rather than incremental inserts (future bulk API)

## Save/Load

- JSON serialization; index is rebuilt on load
- Load time dominates for large DBs (see [SCALE_BENCHMARKS.md](../SCALE_BENCHMARKS.md))
- For frequent persistence, consider periodic saves in the background

## Memory

- Vectors: 4 bytes × dim × n episodes
- 100k × 768 ≈ 300 MB for embeddings alone
- HNSW index adds overhead; plan for ~2× vector size at scale
