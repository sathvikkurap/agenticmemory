# Scale Benchmarks

Measurements at 50k–100k+ episodes for sizing and tuning.

## Running Scale Benchmarks

```bash
cargo bench --bench agent_mem_db_bench -- scale_ --nocapture
```

This runs `scale_insert_768d_50000eps`, `scale_insert_768d_100000eps`, `scale_query_768d_50000eps_topk10`, `scale_query_768d_100000eps_topk10`.

## Placeholder Results (fill after running)

| Scenario | Environment | p50 | p95 | Throughput | Notes |
|----------|-------------|-----|-----|------------|-------|
| Insert 50k episodes (768d) | TBD | TBD | TBD | TBD | new_with_max_elements(768, 51000) |
| Insert 100k episodes (768d) | TBD | TBD | TBD | TBD | new_with_max_elements(768, 101000) |
| Query top-10 @ 50k | TBD | TBD | TBD | TBD | |
| Query top-10 @ 100k | TBD | TBD | TBD | TBD | |

## Tuning Notes

- **max_elements:** Must be ≥ expected episode count. Use `AgentMemDB::new_with_max_elements(dim, n)` for large workloads.
- **Memory:** ~4 bytes per float × dim × n episodes. 100k × 768 ≈ 300 MB for vectors alone.
- **Save/Load:** JSON serialization and index rebuild dominate. Consider bincode or streaming for very large DBs (future).
- **ef_search:** Higher values improve recall but increase query latency. Default 32 is a reasonable trade-off.

## How to Size for Your Workload

1. **Episode count:** Estimate peak episodes. Set `max_elements` to 1.2× that.
2. **Embedding dim:** Match your embedding model (e.g., 384, 768, 1536).
3. **Query latency:** At 10k episodes, ~200 µs/query. At 100k, expect ~500 µs–2 ms depending on hardware.
4. **Insert throughput:** ~11k inserts/sec at 10k. Scales roughly linearly; 50k may take ~5× longer per batch.
