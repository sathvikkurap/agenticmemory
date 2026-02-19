# Research Context: Long-Term Memory for LLM Agents

This document positions AgentMemDB within current research on long-term memory for LLM agents and cites influential work.

## Episodic vs Semantic Memory

**Semantic memory** stores general facts and knowledge (e.g., "Paris is the capital of France"). Traditional RAG over documents is semantic: you index a corpus and retrieve relevant chunks at query time.

**Episodic memory** stores specific events and experiences ("Last Tuesday, the user asked me to use dark mode"). AgentMemDB is designed for episodic memory: each `Episode` is a recorded interaction with `state_embedding`, `reward`, and optional `steps` (trajectory). Retrieval is by *similarity of state* and *reward filtering*, enabling "recall similar past experiences" rather than "recall similar documents."

As noted in [Augmenting LLM Agents with Long-Term Memory](https://www.rohan-paul.com/p/augmenting-llm-agents-with-long-term) (Rohan Paul, 2025), episodic memory modules are crucial for agent coherence and continual learning: they excel at remembering unique, non-repeating events and maintaining context across long gaps. AgentMemDB implements a lightweight episodic store with HNSW-based retrieval.

## Design Influences

### Vector DB and Episodic Memory

The Rohan Paul article distinguishes:

- **Vector DB memory:** Semantic retrieval over embedded chunks. Good for factual lookup.
- **Episodic memory modules:** Store traces of interactions as distinct units, with mechanisms for storage and retrieval. Often combine embedding-based similarity with temporal or structural cues.

AgentMemDB sits at the intersection: we use vector similarity (HNSW) over state embeddings, but the stored units are *episodes* (experiences with reward, metadata, steps) rather than document chunks. The `min_reward` filter at query time lets agents preferentially recall high-reward experiences—a simple form of "what worked before."

### MemoryAgentBench

[MemoryAgentBench](https://www.emergentmind.com/topics/memoryagentbench) (Hu et al., 2025) evaluates LLM agents on four competencies:

1. **Accurate Retrieval (AR):** Locate and extract information from long interaction histories.
2. **Test-Time Learning (TTL):** Apply new rules/labels provided during dialogue.
3. **Long-Range Understanding (LRU):** Form coherent summaries over extended narratives.
4. **Conflict Resolution (CR):** Overwrite outdated facts when contradictions appear.

Our evaluation harness focuses on **Accurate Retrieval** in a simplified setting: multi-turn preference/QA tasks where the agent must recall past "remember" statements. We do not yet implement TTL, LRU, or CR; those are natural extensions.

### Memory Architectures

MemoryAgentBench compares long-context agents, simple RAG (BM25), embedding RAG, and structure-augmented RAG. Findings:

- Embedding RAG outperforms long-context on retrieval but struggles with global summarization.
- No single paradigm dominates all competencies.
- Conflict resolution remains a major bottleneck.

AgentMemDB is an embedding-based episodic store. We aim to show that for *episodic* retrieval (past experiences, not documents), a compact HNSW index is effective and efficient.

## Relation to This Project

| Concept | AgentMemDB Implementation |
|---------|---------------------------|
| Episodic storage | `Episode` with `state_embedding`, `reward`, `steps`, `metadata` |
| Retrieval | `query_similar(embedding, min_reward, top_k)` |
| Persistence | `save_to_file` / `load_from_file` (JSON) |
| Scale | HNSW with configurable `max_elements`; see BENCHMARKS.md |

## References

- [Augmenting LLM Agents with Long-Term Memory](https://www.rohan-paul.com/p/augmenting-llm-agents-with-long-term) — Rohan Paul, 2025. Survey of vector DB, memory transformers, episodic memory.
- [MemoryAgentBench](https://www.emergentmind.com/topics/memoryagentbench) — Hu et al., 2025. Unified benchmark for memory in LLM agents.
- [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://www.emergentmind.com/papers/2507.05257) — Hu et al., 2025. Paper underlying MemoryAgentBench.
