BENCHMARKS
==========

This file collects benchmark scenarios and placeholders for measured values.
Fill in the p50/p95 latencies and throughput values after running the benchmark
commands locally.

What is measured
-----------------
- Episode insert: time to insert N episodes into the in-memory DB (includes HNSW insert).
- Query: time to perform ANN queries (e.g., top-K retrieval) against the DB.
- Save/Load: time to serialize (save) N episodes to disk and to load them back and rebuild the index.
- Disk open (replay vs checkpoint): time to open `AgentMemDBDisk` with 5k episodes—replay replays the append log; checkpoint loads from `exact_checkpoint.json` (much faster).

Interpreting metrics
--------------------
- p50 / p95: the 50th and 95th percentile latencies measured by Criterion. Lower is better.
- Throughput / ops-per-sec: how many operations (inserts or queries) are completed per second.

How to run (Rust/Criterion)
---------------------------
From the repository root, run:

```bash
cargo bench --bench agent_mem_db_bench -- --nocapture
```

**To run only the 4 key scenarios** (faster): add filter `768d_10000`:
```bash
cargo bench --bench agent_mem_db_bench -- 768d_10000 --nocapture
```

**For stable results:** Run 2–3 times and use consistent numbers. Close other heavy apps. The first run warms caches.

**Output:** Criterion prints `time: [lower median upper]` (nanoseconds or ms). Use the **median** (middle value) as p50; use the **upper** value as a proxy for p95. Throughput: insert = 10000/median_sec (inserts/sec), query = 10/median_sec (queries/sec; each iter runs 10 queries), save/load = 1/median_sec (ops/sec). Results also in `target/criterion/<bench_name>/base/estimates.json`.

Results table (MacBook Pro, 768d, 10k episodes)
-----------------------------------------------

| Scenario | Environment (CPU, RAM) | p50 | p95 | Throughput (ops/sec) | Notes |
|---|---:|---:|---:|---:|---|
| Insert 10k episodes (Rust, single-threaded) | Apple M3, 18 GB RAM, macOS 14 | 924 ms | 980 ms | 10,821 | insert_768d_10000eps; HNSW max_elements=20000 |
| Query top-10 similar episodes (Rust, single-threaded) | Apple M3, 18 GB RAM, macOS 14 | 199 µs | 203 µs | 5,029 | query_768d_10000eps_topk10; per-query latency |
| Save 10k episodes to disk (JSON) | Apple M3, 18 GB RAM, macOS 14 | 227 ms | 231 ms | 4.4 | save_768d_10000eps; disk: SSD |
| Load 10k episodes from disk (rebuild index) | Apple M3, 18 GB RAM, macOS 14 | 1.26 s | 1.38 s | 0.79 | load_768d_10000eps; includes index reinsertion |
| Disk open (replay 5k eps) | — | — | — | — | open_replay_5k_eps; exact index, replay append log |
| Disk open (from checkpoint 5k eps) | — | — | — | — | open_from_checkpoint_5k_eps; exact index, load checkpoint |

Notes
-----
- For insert-heavy workloads consider increasing HNSW `max_elements` or using a bulk/batch index construction.
- When comparing numbers, ensure the environment (CPU model, core count, RAM, and OS) is recorded in the Environment column.
- If you're measuring Python-level end-to-end throughput via the PyO3 bindings, note whether the Python process is single-threaded and whether the embedding generation is included in the timing.

