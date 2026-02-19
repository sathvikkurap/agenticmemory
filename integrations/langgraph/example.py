"""Example: AgentMemDBStore as LangGraph long-term memory."""

from langchain_core.embeddings import Embeddings


# Simple mock embedding for demo (no API key needed)
# Uses hash of text to vary embeddings so semantic search returns results
class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        h = sum(ord(c) for c in text) % 100
        return [0.1 + h / 1000.0] * 8


def main() -> None:
    from agent_mem_db_langgraph import AgentMemDBStore

    store = AgentMemDBStore(
        index={
            "dims": 8,
            "embed": MockEmbeddings(),
            "fields": ["text"],
        }
    )

    # Store memories
    store.put(("user_1", "memories"), "m1", {"text": "I love pizza"})
    store.put(("user_1", "memories"), "m2", {"text": "Python is my favorite language"})
    store.put(("user_1", "memories"), "m3", {"text": "I prefer dark mode"})

    # Get by key
    item = store.get(("user_1", "memories"), "m1")
    print("Get m1:", item.value if item else None)

    # Semantic search (mock embeddings return same vector, so order may vary)
    results = store.search(
        ("user_1", "memories"),
        query="programming preferences",
        limit=3,
    )
    print("Search results:", [r.key for r in results])

    # Filter search
    store.put(("user_1", "memories"), "m4", {"text": "Coffee lover", "type": "food"})
    filtered = store.search(
        ("user_1", "memories"),
        filter={"type": "food"},
        limit=5,
    )
    print("Filtered (type=food):", [r.key for r in filtered])

    print("Done.")


if __name__ == "__main__":
    main()
