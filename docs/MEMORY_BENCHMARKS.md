# Memory Benchmarks — Integration & Emulation

**Status:** Design — documents current eval and known benchmarks for future integration.

## Current Eval: Preference & Recall QA

Our built-in eval (`make eval`) uses a **preference/recall QA** format:

- **Store phase:** "Remember: user prefers X" statements
- **Recall phase:** Questions like "What does user prefer?"
- **Task types:** Short (key in recent window) vs long (key at start + 55 filler statements)

See [case_study_preference_qa.md](case_study_preference_qa.md) and [eval_results.md](eval_results.md).

## Known Benchmarks (Literature)

| Benchmark | Focus | Format | Link |
|-----------|-------|--------|------|
| **LongMemEval** | Long-term chat memory, 500 questions | Multi-session, temporal reasoning | [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) |
| **LoCoMo** | Very long-term (35 sessions, ~9K tokens) | QA, summarization, multi-modal | [arXiv:2402.17753](https://arxiv.org/abs/2402.17753) |
| **MemGPT** | OS-inspired memory, virtual context | Document analysis, multi-session chat | [memgpt.ai](https://research.memgpt.ai/) |

## Emulation Strategy

To integrate with external benchmarks:

1. **Adapter pattern:** Convert benchmark format → our `Task` (store_statements, recall_questions, expected_answers)
2. **Embedding:** Use benchmark-provided embeddings or our stub; swap for real embedder when comparing
3. **Metrics:** Reuse our success-rate, latency, token-count reporting

## Task Format (Internal)

Our `Task` dataclass:

```python
class Task:
    task_id: str
    store_statements: list[str]   # "Remember: ..."
    recall_questions: list[str]  # "What does user prefer?"
    expected_answers: list[list[str]]  # Acceptable answers
    task_type: str  # "short" | "long"
```

To add a benchmark: implement `load_<benchmark>_tasks() -> list[Task]` and register in `run_eval` config.

## Future Work

- [ ] LongMemEval adapter: parse their format, map to Task
- [ ] LoCoMo adapter: session decomposition, multi-turn recall
- [ ] MemGPT-style: compare with episodic memory backend
