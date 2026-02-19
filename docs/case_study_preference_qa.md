# Case Study: Preference & Recall QA

**Domain:** Agents that must remember user preferences and answer recall questions across long conversations.

**Result:** AgentMemDB achieves **97.5%** success vs **85%** for naive (truncated history). On long-history tasks where the key info is at the start of a 56-statement context, AgentMemDB reaches **88.9%** vs **33.3%** for naive.

## The Problem

Many agent applications need to recall information from earlier in a session:

- "Remember: user prefers dark mode" → later: "What does user prefer for display?"
- "Remember: user's secret code is 42" (amid 55 filler statements) → "What is the secret code?"

**Naive approach:** Keep full conversation history, truncate to last N tokens (e.g., 256) to fit the context window. Works when the key statement is recent; fails when truncation drops it.

**AgentMemDB approach:** Store each "remember" as an episode (state embedding + metadata). At recall time, embed the question and retrieve similar episodes. Key info is found by *similarity*, not position.

## When AgentMemDB Wins

| Scenario | Naive | AgentMemDB |
|---------|-------|------------|
| Short history (key in recent window) | 100% | 100% |
| Long history (key at start, 55 fillers) | 33.3% | **88.9%** |
| Overall (mixed tasks) | 85% | **97.5%** |

**Takeaway:** Use AgentMemDB when your agent must recall information that may be far from the current turn—e.g., preference bots, personal assistants, support agents with long threads.

## Implementation Pattern

```python
# Store phase
for stmt in store_statements:
    emb = embed(stmt)
    db.store_episode(Episode(task_id=..., state_embedding=emb, reward=1.0, metadata={"text": stmt}))

# Recall phase
emb = embed(question)
similar = db.query_similar(emb, min_reward=0.0, top_k=25)  # higher k for long histories
context = "\n".join(ep.metadata["text"] for ep in similar)
answer = policy(context, question)
```

**Tuning:** For long histories (many episodes), increase `top_k` so the relevant episode is in the retrieved set. See [eval_case_studies.md](eval_case_studies.md).

## Reproducing

```bash
make eval
```

See [eval_results.md](eval_results.md) for full metrics and [agent_integration.md](agent_integration.md) for the eval setup.
