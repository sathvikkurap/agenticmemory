# Design: LangChain Integration

**Status:** First slice in progress  
**Related:** [design_roadmap.md](design_roadmap.md)

## Goal

Provide a drop-in LangChain VectorStore backend using AgentMemDB. Enables RAG chains and retrieval to use episodic memory instead of (or alongside) traditional vector DBs.

## Approach

Implement `AgentMemDBVectorStore` that subclasses `langchain_core.vectorstores.VectorStore`:
- **add_texts(texts, metadatas, ids)** — embed texts, store as episodes, return ids
- **similarity_search(query, k)** — embed query, query_similar, return Documents
- **similarity_search_with_score** — optional; return (doc, distance)

## Dependencies

- `langchain-core` — VectorStore, Document, Embeddings interface
- `agent_mem_db_py` — our Python bindings
- User provides an `Embeddings` implementation (e.g., OpenAIEmbeddings, HuggingFaceEmbeddings)

## API

```python
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from agent_mem_db_langchain import AgentMemDBVectorStore

# User provides embedding model
store = AgentMemDBVectorStore(embedding=my_embeddings, dim=768)
store.add_texts(["Alice likes cats", "Bob prefers dogs"])
docs = store.similarity_search("pets", k=2)
```

## Implementation Notes

- Episode metadata stores `{"text": original_text, **metadatas}` for Document reconstruction
- IDs: use Episode.id (UUID string)
- Embedding dimension must match the embedding model output

## Optional Extras

- `similarity_search_by_vector` — for pre-embedded queries
- `delete(ids)` — not in AgentMemDB; document as unsupported or add tombstone
- Filter support via QueryOptions (tags_any, time_after) — future
