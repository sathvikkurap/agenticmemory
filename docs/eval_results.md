# Evaluation Results

**Last run:** 2026-02 (stub policy, 40 tasks, preference/QA domain)

## Setups

| Setup | Description |
|-------|-------------|
| **no_memory** | Stateless agent. Each turn has no context; cannot answer recall questions. |
| **naive** | Full conversation history with truncation to last 256 tokens. Succeeds when key info is in the recent window; fails when truncation drops it. |
| **agent_mem_db** | Episodic store with HNSW retrieval. Stores each "remember" as an episode; retrieves similar episodes before answering. Key info is found by similarity regardless of position in history. |

## Results

| Setup | Success Rate | Tokens/turn | Latency/turn (ms) |
|-------|-------------|-------------|-------------------|
| no_memory | 7.5% | 47 | 0.00 |
| naive | 85.0% | 52 | 0.00 |
| agent_mem_db | **97.5%** | 56 | 0.11 |

### Latency Percentiles (per-task total ms)

| Setup | p50 | p95 | p99 |
|-------|-----|-----|-----|
| no_memory | 0.00 | 0.00 | 0.00 |
| naive | 0.00 | 0.00 | 0.01 |
| agent_mem_db | 0.24 | 8.16 | 8.80 |

AgentMemDB adds retrieval latency (HNSW query); p95/p99 capture tail latency on long tasks.

### By Task Type

| Setup | Short tasks success | Long tasks success |
|-------|---------------------|--------------------|
| no_memory | 9.7% | 0.0% |
| naive | 100.0% | 33.3% |
| agent_mem_db | 100.0% | **88.9%** |

## Where AgentMemDB Helps

**Long-history tasks:** The key statement is at the *start* of a 56-statement history. Naive truncation keeps only the last 256 tokens, dropping the key. AgentMemDB retrieves by similarity, so it finds the relevant episode regardless of position. Result: **88.9% vs 33.3%** on long tasks.

**Short tasks:** Both naive and agent_mem_db achieve 100%—the full history fits in the context window.

**Overall:** AgentMemDB (97.5%) clearly outperforms naive (85%) and no_memory (7.5%). It maintains comparable token efficiency (56 vs 52 tokens/turn) with a small latency cost (0.11 ms/turn for retrieval).

## Task Set

- 40 tasks total
- **Short:** 10 templates × ~3 runs each (1 store statement, 1 recall question)
- **Long:** 3 templates × ~3 runs each (1 key statement + 55 filler statements; key at start)

## Reproducing

```bash
cd agent_mem_db/python
# Build Python bindings first (see python/README.md)
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1  # if Python 3.13+
maturin develop --release
python -m examples.agents.run_eval
```

Or from repo root: `make eval` (requires `python/.venv` with maturin-built extension).

Output: stdout summary + `eval_results.json` at repo root. See [eval_case_studies.md](eval_case_studies.md) for tuning rationale and failure analysis.

**Analysis:** Run `make eval-analyze` to generate [eval_analysis_report.md](eval_analysis_report.md). For interactive exploration, use `python/notebooks/eval_analysis.ipynb`.
