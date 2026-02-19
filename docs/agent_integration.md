# Agent Integration

This document describes the agent integration examples and how to run them.

## Overview

We provide three agent variants for A/B comparison:

| Variant | Memory | Use Case |
|---------|--------|----------|
| **no_memory** | None | Baseline: stateless, cannot answer recall questions |
| **naive** | Full history, truncated to 256 tokens | Baseline: works when context fits; fails when key info is truncated |
| **agent_mem_db** | Episodic store with HNSW retrieval | Retrieves relevant past episodes regardless of position in history |

## Domain

**Preference & multi-turn QA:** Tasks where the agent must recall information from earlier in the session (e.g., "Remember: user prefers X" followed by "What does user prefer?").

## Running the Examples

From the `python/` directory:

```bash
cd python
maturin develop --release   # if not already built
python -m examples.agents.run_eval
```

## Running the Evaluation

The same command runs the full A/B evaluation:

```bash
python -m examples.agents.run_eval
```

Results are printed to stdout. See [eval_results.md](eval_results.md) for typical numbers.

## File Layout

```
python/examples/agents/
├── base.py              # Tasks, stub policy, embedding
├── no_memory.py         # No-memory baseline
├── naive_memory.py      # Naive (truncated history) baseline
├── agent_mem_db_agent.py # AgentMemDB-backed agent
├── run_eval.py          # Evaluation harness
└── __init__.py
```

## Design Notes

- **Stub policy:** Deterministic pattern-matching to avoid LLM API costs. Real LLM integration can be layered on top.
- **Embedding:** Bag-of-words style for reproducibility. Production use would substitute a real embedding model.
- **Long-history tasks:** Some tasks put the key statement first, then 55+ filler statements. Naive truncation drops the key; AgentMemDB retrieves it by similarity.
