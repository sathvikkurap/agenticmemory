"""Tests for AgentMemDBStore."""

import pytest
from langchain_core.embeddings import Embeddings

from agent_mem_db_langgraph import AgentMemDBStore


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 + (sum(ord(c) for c in t) % 10) / 100.0] * 8 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def store():
    return AgentMemDBStore(
        index={
            "dims": 8,
            "embed": MockEmbeddings(),
            "fields": ["text"],
        }
    )


def test_put_get(store):
    store.put(("user_1", "memories"), "m1", {"text": "I love pizza"})
    item = store.get(("user_1", "memories"), "m1")
    assert item is not None
    assert item.value["text"] == "I love pizza"
    assert item.key == "m1"


def test_get_missing(store):
    item = store.get(("user_1", "memories"), "nonexistent")
    assert item is None


def test_search_with_query(store):
    store.put(("user_1", "memories"), "m1", {"text": "I love pizza"})
    store.put(("user_1", "memories"), "m2", {"text": "Python is great"})
    results = store.search(("user_1", "memories"), query="food", limit=5)
    assert len(results) >= 1
    keys = {r.key for r in results}
    assert "m1" in keys or "m2" in keys


def test_search_with_filter(store):
    store.put(("user_1", "memories"), "m1", {"text": "Pizza", "type": "food"})
    store.put(("user_1", "memories"), "m2", {"text": "Python", "type": "code"})
    results = store.search(("user_1", "memories"), filter={"type": "food"}, limit=5)
    assert len(results) == 1
    assert results[0].value["type"] == "food"


def test_delete(store):
    store.put(("user_1", "memories"), "m1", {"text": "temp"})
    assert store.get(("user_1", "memories"), "m1") is not None
    store.put(("user_1", "memories"), "m1", None)
    assert store.get(("user_1", "memories"), "m1") is None
