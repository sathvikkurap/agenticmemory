"""Tests for AgentMemDBVectorStore."""

import pytest
from langchain_core.embeddings import Embeddings

from agent_mem_db_langchain import AgentMemDBVectorStore


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 + (sum(ord(c) for c in t) % 10) / 100.0] * 8 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def store():
    return AgentMemDBVectorStore(embedding=MockEmbeddings(), dim=8, use_exact=True)


def test_add_texts_returns_ids(store):
    ids = store.add_texts(["doc one", "doc two"], metadatas=[{}, {}])
    assert len(ids) == 2
    assert all(id is not None for id in ids)


def test_similarity_search_returns_docs(store):
    store.add_texts(["Alice likes cats", "Bob prefers dogs"], metadatas=[{}, {}])
    docs = store.similarity_search("pets", k=2)
    assert len(docs) >= 1
    assert all(hasattr(d, "page_content") and hasattr(d, "metadata") for d in docs)


def test_similarity_search_by_vector(store):
    store.add_texts(["Python", "Rust"], metadatas=[{}, {}])
    vec = store.embedding.embed_query("code")
    docs = store.similarity_search_by_vector(vec, k=2)
    assert len(docs) >= 1


def test_from_texts(store):
    vs = AgentMemDBVectorStore.from_texts(
        ["a", "b", "c"],
        embedding=MockEmbeddings(),
        dim=8,
        use_exact=True,
    )
    docs = vs.similarity_search("a", k=3)
    assert len(docs) >= 1


def test_metadatas_preserved(store):
    store.add_texts(
        ["Hello"],
        metadatas=[{"source": "test", "key": "value"}],
    )
    docs = store.similarity_search("Hello", k=1)
    assert len(docs) == 1
    assert docs[0].metadata.get("source") == "test"
    assert docs[0].metadata.get("key") == "value"
