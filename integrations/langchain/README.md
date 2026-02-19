# agent_mem_db_langchain

LangChain VectorStore backend using AgentMemDB for episodic memory with HNSW retrieval.

## Install

```bash
pip install agent_mem_db_py langchain-core
# From repo: pip install -e integrations/langchain
```

## Usage

```python
from langchain_core.embeddings import FakeEmbeddings  # or OpenAIEmbeddings, etc.
from agent_mem_db_langchain import AgentMemDBVectorStore

embedding = FakeEmbeddings(size=16)
store = AgentMemDBVectorStore(embedding=embedding, dim=16)
store.add_texts(["Alice likes cats", "Bob prefers dogs"])
docs = store.similarity_search("pets", k=2)
```

### With RAG

```python
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from agent_mem_db_langchain import AgentMemDBVectorStore

embedding = FakeEmbeddings(size=16)
store = AgentMemDBVectorStore(embedding=embedding, dim=16)
store.add_texts(["Key fact: the project deadline is March 15"])

retriever = store.as_retriever(k=2)
# Use with LCEL: retriever | format_docs | llm
```
