# Agentic Memory DB — Overview

[← Back to README](../README.md)

This crate provides a compact in-memory episodic memory with approximate nearest
neighbour search (HNSW) and JSON persistence. It is intended as a lightweight
building block for agent systems that need to store and recall past episodes.

Key concepts
- Episode: a single recorded interaction or outcome. Includes `task_id`,
  `state_embedding: Vec<f32>`, `reward: f32`, optional `steps`, and `metadata`.
- EpisodeStep: optional per-step trace information attached to an Episode.
- AgentMemDB: the main in-memory store; supports store, query_similar, and
  save/load (JSON) operations.

Design notes
- HNSW is used for nearest-neighbour search (via the `hnswx` crate). The index
  is configured with a conservative default `max_elements` — increase this if you
  plan to store many tens of thousands of episodes.
- The Python bindings (in `python/`) intentionally expose a minimal, ergonomic
  surface so the extension builds across common CPython versions using PyO3.

Usage
- Rust: use `AgentMemDB::new(dim)` and `store_episode`/`query_similar`.
- Python: build the PyO3 extension in `python/` (see `python/README.md`), then
  import `agent_mem_db_py` and use `AgentMemDB` and `Episode`.

Next steps / roadmap
- Add a bulk-insert API and parallelized index construction for large-scale
  dataset ingestion.
- Add batch Python APIs for efficiency when storing many episodes from Python.
- Provide CI/maturin steps for building/publishing the Python wheel to PyPI.

For examples and tests, see the `examples/` and `python/examples/` directories. For agent integration and A/B evaluation, see [agent_integration.md](agent_integration.md). See [eval_results.md](eval_results.md) for evaluation results (AgentMemDB beats naive on long-history tasks). For LangChain and LangGraph integrations, see [integrations/](integrations/).

RAG over documents vs RAG over episodes (experiences)
----------------------------------------------------

- RAG over documents (traditional):
  - Index a corpus of documents (text chunks, passages).
  - At query time, embed a user query and retrieve nearest document chunks.
  - Use retrieved documents as context for an LLM to generate an answer.

- RAG over episodes (agent experiences):
  - Index past agent episodes by their state embeddings and metadata (reward,
    actions, observations).
  - At decision time, embed the current agent state and retrieve similar
    episodes (preferably high-reward ones).
  - Use retrieved episodes as context for an agent policy or LLM to choose an
    action or to predict outcomes. This is especially useful for learning from
    prior trajectories and reusing successful behaviours.

Simple example (conceptual):

- Documents RAG: user asks "How to cook rice?" → retrieve text passages about rice-cooking → LLM composes answer.
- Episodes RAG: agent faces state S → retrieve past episodes where states similar to S had high reward → use those episodes to bias action selection.

How to use this with an agent loop (5–10 line pseudocode)
-------------------------------------------------------

1. For each timestep/task:
   - Embed current state: `emb = embed(state)`
   - Retrieve similar episodes: `hits = db.query_similar(emb, min_reward=threshold, top_k=K)`
   - Construct context from `hits` (e.g., recent high-reward trajectories)
   - Decide action: `action = policy(state, context)` (policy can be an LLM or a learned policy)
   - Execute action, observe reward and next_state
   - Store episode: `db.store_episode(Episode::new(task_id, emb, reward))`

The above pattern keeps a compact episodic memory and uses RAG-like retrieval
to bootstrap action selection from past successful experiences.
