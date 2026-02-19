# LangChain Integration

AgentMemDB as a **VectorStore** backend for LangChain. Use episodic memory in RAG chains, retrieval-augmented generation, and document stores.

## Install

```bash
pip install agent_mem_db_py langchain-core
# From repo:
pip install -e integrations/langchain
```

## Quick Start

```python
from langchain_core.embeddings import FakeEmbeddings  # or OpenAIEmbeddings, HuggingFaceEmbeddings
from agent_mem_db_langchain import AgentMemDBVectorStore

embedding = FakeEmbeddings(size=768)
store = AgentMemDBVectorStore(embedding=embedding, dim=768)

store.add_texts(["Alice likes cats", "Bob prefers dogs"])
docs = store.similarity_search("pets", k=2)
```

## API

| Method | Description |
|--------|-------------|
| `AgentMemDBVectorStore(embedding, dim, use_exact=False, max_elements=20000)` | Create store. `use_exact=True` for brute-force search (small datasets). |
| `add_texts(texts, metadatas=None, ids=None)` | Add texts; returns episode IDs. |
| `similarity_search(query, k=4, filter=None)` | Search by query string. Filter supports `tags_any`, `time_after`, `time_before`. |
| `similarity_search_by_vector(embedding, k=4)` | Search by pre-computed embedding. |
| `from_texts(texts, embedding, metadatas=None, **kwargs)` | Create and populate in one call. |

## RAG Chain Example

```python
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from agent_mem_db_langchain import AgentMemDBVectorStore

embedding = FakeEmbeddings(size=16)
store = AgentMemDBVectorStore(embedding=embedding, dim=16)
store.add_texts(["Project deadline: March 15", "Team uses Python"])

retriever = store.as_retriever(k=2)
# Use with LCEL: retriever | format_docs | llm
```

## Run Example

```bash
make langchain-example
```

## See Also

- [Design doc](../design_langchain_integration.md)
- [integrations/langchain/README.md](../../integrations/langchain/README.md)
