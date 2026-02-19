"""Example: AgentMemDB as LangChain VectorStore."""

from langchain_core.embeddings import FakeEmbeddings
from agent_mem_db_langchain import AgentMemDBVectorStore

# Use FakeEmbeddings for demo (deterministic)
embedding = FakeEmbeddings(size=16)
store = AgentMemDBVectorStore(embedding=embedding, dim=16)

# Add texts
ids = store.add_texts(
    ["Alice likes cats", "Bob prefers dogs", "Charlie has a parrot"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
)
print("Stored", len(ids), "documents")

# Search
docs = store.similarity_search("pets", k=2)
for d in docs:
    print("-", d.page_content, d.metadata)
