# Design: Config-Driven Evaluation

**Status:** First slice in progress  
**Related:** [design_roadmap.md](design_roadmap.md)

## Goal

Make the evaluation framework configurable via YAML/JSON so users can:
- Select which variants to run (no_memory, naive, agent_mem_db, future: exact, disk, etc.)
- Tune variant parameters (e.g., top_k for agent_mem_db)
- Control task count and mix (short vs long)
- Reproduce experiments without code changes

## Config Format (YAML)

```yaml
variants:
  - no_memory
  - naive
  - agent_mem_db
num_tasks: 40
# Optional per-variant overrides (future)
# agent_mem_db:
#   top_k_short: 10
#   top_k_long: 25
```

## Implementation

1. **Variant registry** — Map name → run_fn. Extensible for new variants.
2. **Config loader** — Read YAML; validate variant names.
3. **run_eval --config path** — Use config; fallback to default (current hardcoded).
4. **Default config** — `eval_config.yaml` in repo for reproducibility.

## First Slice

- Config specifies `variants` (list) and `num_tasks`
- run_eval accepts `--config eval_config.yaml` (optional)
- No per-variant params yet (keep agent_mem_db tuning in code for now)
