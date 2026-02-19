# Comparison with Alternatives

How Agent Memory DB compares to similar tools and when to use it.

## Agent Memory DB vs. Vector Databases

| | Agent Memory DB | Pinecone, Weaviate, Qdrant, Milvus |
|--|-----------------|-------------------------------------|
| **Purpose** | Episodic memory for agents | General-purpose vector search |
| **Data model** | Episodes (embedding + reward + metadata) | Vectors + optional metadata |
| **Deployment** | Library or single-binary server | Managed service or self-hosted cluster |
| **Scale** | 10K–100K episodes typical | Millions–billions of vectors |
| **Use case** | Agent recall, preference QA, trajectory reuse | RAG over documents, semantic search |

**Choose Agent Memory DB when:** You need lightweight, in-process or single-node episodic memory for an agent. No external DB, minimal ops.

**Choose a vector DB when:** You need large-scale document RAG, multi-tenant SaaS, or distributed search.

---

## Agent Memory DB vs. Conversation History / Context Window

| | Agent Memory DB | Truncated history (naive) |
|--|-----------------|---------------------------|
| **Recall** | Similarity search over all stored episodes | Only recent N tokens |
| **Long history** | Retrieves by similarity regardless of position | Drops early context when truncated |
| **Eval result** | 97.5% success (preference QA) | 85% naive; 33% on long tasks |

**Choose Agent Memory DB when:** Key information may be anywhere in history (e.g. "user said X at the start"). See [case_study_preference_qa.md](case_study_preference_qa.md).

**Choose naive when:** Context fits in window and recency matters more than similarity.

---

## Agent Memory DB vs. LangChain/LangGraph Memory

| | Agent Memory DB | LangChain BufferMemory, LangGraph checkpointer |
|--|-----------------|-----------------------------------------------|
| **Storage** | HNSW index, similarity search | In-memory or external store |
| **Retrieval** | `query_similar(embedding)` — semantic | Sliding window or full history |
| **Integration** | AgentMemDBVectorStore, AgentMemDBStore | Native to framework |

Agent Memory DB **integrates with** LangChain and LangGraph — it's a memory backend, not a replacement. Use it when you want similarity-based recall instead of (or in addition to) recency-based memory.

---

## Summary

- **Lightweight agent memory** → Agent Memory DB
- **Document RAG at scale** → Vector DB (Pinecone, etc.)
- **Simple conversation buffer** → Framework built-in memory
- **Best of both** → Agent Memory DB + LangChain/LangGraph
