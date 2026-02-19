# Design: LangGraph Memory Adapter

**Status:** First slice in progress  
**Related:** [design_roadmap.md](design_roadmap.md), [design_langchain_integration.md](design_langchain_integration.md)

## Goal

Provide a LangGraph `BaseStore` implementation backed by AgentMemDB for episodic long-term memory. Enables LangGraph agents to use semantic search over past experiences (episodes) with HNSW retrieval.

## Context

LangGraph supports two memory types:
- **Short-term:** Thread-scoped state via checkpointers
- **Long-term:** Cross-thread document store with `put`, `get`, `search` (namespace, key-value, optional semantic search)

AgentMemDB is episodic memory: stores experiences (state_embedding, reward, metadata) and retrieves by similarity. The LangGraph store adapter bridges these: key-value semantics for get/put, AgentMemDB for semantic search.

## Approach

Implement `AgentMemDBStore` extending `langgraph.store.base.BaseStore`:

- **put(namespace, key, value)** — Embed value (via configured fields), store in AgentMemDB + key-value index
- **get(namespace, key)** — Lookup from key-value index
- **search(namespace_prefix, query=..., filter=..., limit=...)** — If `query`: embed, query AgentMemDB, filter by namespace/filter; else: filter key-value by namespace/filter
- **delete(namespace, key)** — Remove from key-value; episodes remain in AgentMemDB but are filtered out of search

## Dependencies

- `langgraph` — BaseStore, Op types
- `agent_mem_db_py` — episodic memory backend
- User provides `index={"embed": Embeddings, "dims": N}` for semantic search

## API

```python
from langchain_core.embeddings import Embeddings
from agent_mem_db_langgraph import AgentMemDBStore

store = AgentMemDBStore(
    index={
        "dims": 768,
        "embed": my_embeddings,
        "fields": ["text"],  # optional, default ["$"]
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
item = store.get(("user_123", "memories"), "1")
results = store.search(("user_123", "memories"), query="What foods do I like?", limit=5)
```

## Implementation Notes

- **Dual storage:** `_data[(namespace, key)]` = Item for get/delete; AgentMemDB for vector search
- **Episode metadata:** `{"__namespace": [...], "__key": key, **value}` — internal fields prefixed to avoid collisions
- **Delete:** Remove from `_data`; search results cross-check against `_data` to exclude deleted items
- **Score:** AgentMemDB returns L2-sorted episodes; we pass `score=None` (order preserved)

## Optional Extras

- Async `abatch` for async LangGraph graphs
- AgentMemDBDisk backend for persistence
- TTL support (future)
