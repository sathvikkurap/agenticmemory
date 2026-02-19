# Design Notes: Evaluation Harness

**Status:** Implemented (2026-02)  
**Related:** [docs/eval_results.md](eval_results.md), [examples/agents/](../examples/agents/)

## Goal

Provide quantitative evidence that AgentMemDB improves agent behavior over baselines.

## Task Design

Inspired by MemoryAgentBench (emergentmind.com/topics/memoryagentbench) but simplified:

- **Accurate Retrieval (AR):** Answer questions that require recalling information from earlier in the session.
- **Multi-turn:** Store phase (N "remember" statements) followed by recall phase (M questions).

We do *not* implement Conflict Resolution, Test-Time Learning, or Long-Range Understanding in v1; those can be added later.

## Task Set

20–50 tasks, each consisting of:

1. **Setup:** 3–10 "remember" statements (e.g., "Remember: user prefers dark mode")
2. **Query:** 1–3 recall questions (e.g., "What does user prefer for display?")
3. **Ground truth:** Expected answers for grading

Tasks are generated programmatically to ensure variety and controllability.

## Metrics

| Metric | Definition |
|--------|------------|
| **Success rate** | % of recall questions answered correctly |
| **Avg tokens per turn** | Total context + response tokens / num turns (proxy for efficiency) |
| **Latency per step** | Time per agent step (ms) |

## Harness Flow

1. Load task set (JSON or Python structure)
2. For each variant (no_memory, naive, agent_mem_db):
   - Run each task
   - Record success, tokens, latency
3. Aggregate and write `docs/eval_results.md`

## Stub Policy Grading

The stub policy extracts answers from context via simple patterns. Grading:

- **Exact match:** Response contains the expected substring
- **Fuzzy:** Optional semantic similarity (future)

For v1 we use exact substring match.
