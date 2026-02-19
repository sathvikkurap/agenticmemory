"""Example: LangGraph chatbot with AgentMemDBStore for long-term memory.

This agent remembers facts the user shares and retrieves them when answering.
Uses mock LLM and embeddings (no API key required).
"""

from typing import Annotated, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

try:
    from langgraph.graph import START, END, StateGraph
    from langgraph.config import get_store
except ImportError as e:
    raise ImportError("Install langgraph: pip install langgraph") from e

from agent_mem_db_langgraph import AgentMemDBStore


# --- Mock components (no API key) ---
class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        h = sum(ord(c) for c in text) % 100
        return [0.1 + h / 1000.0] * 8


def mock_llm(messages: list[BaseMessage], memory_context: str) -> str:
    """Simulate LLM response using memory context."""
    last = messages[-1].content if messages else ""
    if memory_context:
        return f"[Using memory: {memory_context[:80]}...] Response to: {last}"
    return f"Response to: {last}"


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    user_id: str


# --- Agent node ---
def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    user_id = state.get("user_id", "default")
    if not messages:
        return {"messages": [AIMessage(content="Hello! How can I help?")]}

    last_msg = messages[-1]
    if not isinstance(last_msg, HumanMessage):
        return {"messages": []}

    user_text = last_msg.content
    store = get_store()

    # Search memories for relevant context
    namespace = (user_id, "memories")
    results = store.search(namespace, query=user_text, limit=3)
    memory_context = " | ".join(r.value.get("text", str(r.value)) for r in results)

    # Generate response
    response = mock_llm(messages, memory_context)

    # If user seems to share a fact (not ask), save it (simplified heuristic)
    fact_phrases = ("i like ", "i love ", "i prefer ", "my favorite ", "i'm ", "i am ")
    is_question = user_text.strip().endswith("?")
    if not is_question and any(p in user_text.lower() for p in fact_phrases):
        key = f"m_{hash(user_text) % 10000}"
        store.put(namespace, key, {"text": user_text})

    return {"messages": [AIMessage(content=response)]}


# --- Build graph ---
def main() -> None:
    store = AgentMemDBStore(
        index={
            "dims": 8,
            "embed": MockEmbeddings(),
            "fields": ["text"],
        }
    )

    graph = (
        StateGraph(AgentState)
        .add_node("agent", agent_node)
        .add_edge(START, "agent")
        .add_edge("agent", END)
    )
    app = graph.compile(store=store)

    # Simulate conversation
    config = {"configurable": {"user_id": "user_1"}}
    for msg in [
        "I love pizza",
        "What foods do I like?",
        "My favorite language is Python",
        "What programming language do I prefer?",
    ]:
        result = app.invoke(
            {"messages": [HumanMessage(content=msg)], "user_id": "user_1"},
            config=config,
        )
        out = result["messages"][-1].content
        print(f"User: {msg}")
        print(f"Agent: {out}\n")

    print("Done. Memories stored:", len(store.search(("user_1", "memories"), limit=10)))


if __name__ == "__main__":
    main()
