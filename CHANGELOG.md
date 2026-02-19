# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **ExactIndex checkpoint:** `DiskOptions::exact_with_checkpoint(dim)` for fast restart; `checkpoint()` persists `exact_checkpoint.json`; skip log replay on open when valid
- **Server disk persistence:** When `AGENT_MEM_DATA_DIR` set, per-tenant AgentMemDBDisk; `POST /v1/checkpoint` for disk-backed tenants
- **Helm persistence:** `persistence.enabled` auto-sets `AGENT_MEM_DATA_DIR` to mount path
- **AgentMemDBDisk Node bindings:** `AgentMemDbDisk.open`, `openExactWithCheckpoint`, `storeEpisode`, `querySimilar`, `checkpoint`, `prune*`
- **AgentMemDBDisk Go bindings:** C API + `DiskDB` with `OpenDisk`, `OpenDiskExactWithCheckpoint`, `Store`, `Query`, `Checkpoint`, `Prune*`
- **`make test-all`:** Runs Rust + Node + Go + Python tests in one command
- **Design docs:** `design_pq_ivf.md`, `design_hnsw_checkpoint.md`, `design_horizontal_scaling.md` for future work
- **Multi-replica deployment:** Helm `persistence.accessMode=ReadWriteMany` for NFS-backed shared storage; `deploy/README.md` multi-replica section
- **Disk open benchmark:** `bench_disk_open_replay_vs_checkpoint` compares open time with replay vs checkpoint (5k episodes)

### Changed

- **Python 3.14:** `make python-dev` uses `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for compatibility
- **Python tests:** `make python-test` uses `.venv/bin/python` for correct package resolution
- **Examples:** `examples/disk_checkpoint.rs`; `make disk-checkpoint` target

## [0.2.1] - 2026-02-16

### Changed

- **Eval harness:** Task-type tagging (short vs long), per-segment stats, tokens/turn and latency/turn metrics
- **AgentMemDB tuning:** `top_k=25` on long tasks (56 episodes) to ensure key episode is retrieved; `top_k=10` on short tasks
- **Results:** AgentMemDB 97.5% vs naive 85%; on long-history tasks AgentMemDB 88.9% vs naive 33.3%
- **Docs:** `docs/eval_results.md` with full tables; `docs/eval_case_studies.md` for tuning rationale
- **Reproducibility:** `make eval` target; `eval_results.json` output; CONTRIBUTING.md eval instructions

## [0.2.0] - 2026-02-16

### Added

- **Agent integration:** `python/examples/agents/` with no_memory, naive, and agent_mem_db variants
- **A/B evaluation:** `run_eval.py` harness; 40 tasks (preference/QA domain); AgentMemDB beats no-memory, competitive with naive on long-history tasks
- **Scale support:** `AgentMemDB::new_with_max_elements(dim, n)` (Rust) and `AgentMemDB.with_max_elements(dim, n)` (Python) for 50k–100k+ episodes
- **Scale benchmarks:** `scale_insert_*`, `scale_query_*` for 50k and 100k episodes
- **Synthetic episode generator:** `python/examples/synthetic_episodes.py`
- **Documentation:** `docs/agent_integration.md`, `docs/eval_results.md`, `docs/research.md`, `docs/tuning.md`, `docs/design_notes_*.md`, `SCALE_BENCHMARKS.md`
- **CI:** GitHub Actions for Rust (fmt, clippy, test) and Python (maturin develop, pytest, agent eval)
- **AGENTS_GUIDE.md:** Conventions and pointers for contributors

## [0.1.0] - 2026-02-16

### Added

- Initial public release of `agent_mem_db`
- Rust core: in-memory episodic memory with HNSW approximate nearest-neighbor search
- Python bindings via PyO3 (maturin): `agent_mem_db_py` package
- `Episode` and `EpisodeStep` types for storing agent experiences
- `AgentMemDB`: store, query_similar, query_similar_batch, store_episodes, save/load (JSON)
- Batch APIs for efficient multi-episode and multi-query operations
- Examples: `agent_sim` (Rust), `basic.py`, `langgraph_agent.py` (Python)
- Benchmarks: insert, query, save, load (Criterion)
- Documentation: README, overview, BENCHMARKS.md, CONTRIBUTING.md
