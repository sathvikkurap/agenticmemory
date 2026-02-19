# Design Notes: Agent Integration

**Status:** Implemented (2026-02)  
**Related:** [docs/agent_integration.md](agent_integration.md), [examples/agents/](../examples/agents/)

## Goal

Integrate AgentMemDB into at least one real agent stack and demonstrate it improves behavior over no-memory and naive-memory baselines.

## Architecture Choices

### Domain: Preference & Multi-Turn QA

We use a **personal-assistant** domain where memory clearly matters:

- **Store phase:** "Remember: user prefers X" / "User said their favorite color is Y"
- **Recall phase:** "What does user prefer?" / "What's my favorite color?"

Tasks require retrieving information from earlier in the session. A no-memory agent cannot succeed; a naive (full-history) agent can but scales poorly; AgentMemDB provides semantic retrieval over episodes.

### Three Variants

| Variant | Memory | Behavior |
|---------|--------|----------|
| **no_memory** | None | Each turn is stateless. Cannot answer recall questions. |
| **naive** | Full conversation history (truncated to last N tokens) | Can answer if info is in recent context. Fails when history exceeds window. |
| **agent_mem_db** | Episodic store with HNSW retrieval | Stores each "remember" as an episode. Retrieves similar episodes before answering. |

### Stub LLM

To avoid API costs and enable reproducible evaluation, we use a **deterministic stub** that:

- Given context + query, outputs a response by pattern-matching (e.g., if "favorite color" appears in context, extract and return it)
- Simulates success/failure based on whether the right information is in the provided context

This lets us run A/B experiments without real LLM calls while preserving the memory-retrieval pattern.

### Embedding

Deterministic hash-based embedding: `hash(text) → seed → random vector`. Same text → same vector. Enables reproducible benchmarks.

## File Layout

```
examples/agents/
├── base.py          # Shared: task definitions, stub policy, embedding
├── no_memory.py     # Baseline: no memory
├── naive_memory.py  # Baseline: full history with truncation
├── agent_mem_db.py  # AgentMemDB-backed agent
├── run_eval.py      # Run all three and collect metrics
└── README.md        # How to run
```

## Design Decisions Log

1. **Python-only for agent examples:** Agent frameworks (LangGraph, etc.) are Python-centric. Rust examples remain in `examples/agent_sim.rs`.
2. **Stub over real LLM:** Enables cheap, reproducible eval. Real LLM integration can be added as an optional layer.
3. **Single domain first:** Preference/QA is simple and clearly memory-dependent. Coding tasks can be added later.
