# LangGraph Integration

AgentMemDB as a **BaseStore** for LangGraph long-term memory. Enables semantic search over agent memories with HNSW retrieval.

## Install

```bash
pip install agent_mem_db_py langgraph langchain-core
# From repo:
pip install -e integrations/langgraph
```

## Quick Start

```python
from langchain_core.embeddings import FakeEmbeddings
from agent_mem_db_langgraph import AgentMemDBStore

store = AgentMemDBStore(
    index={
        "dims": 768,
        "embed": FakeEmbeddings(size=768),
        "fields": ["text"],  # optional, default ["$"] embeds entire value
    }
)

store.put(("user_123", "memories"), "m1", {"text": "I love pizza"})
item = store.get(("user_123", "memories"), "m1")
results = store.search(("user_123", "memories"), query="What foods do I like?", limit=5)
```

## Using the Store in a Graph

Pass the store when compiling your graph:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_store

def agent_node(state):
    store = get_store()
    results = store.search(("user_id", "memories"), query=state["message"], limit=5)
    # Use results to augment response
    ...

graph = StateGraph(State).add_node("agent", agent_node).add_edge(START, "agent").add_edge("agent", END)
app = graph.compile(store=store)
```

## API

| Method | Description |
|--------|-------------|
| `put(namespace, key, value)` | Store item. Value must be `dict[str, Any]` (JSON-serializable). |
| `get(namespace, key)` | Retrieve by key. Returns `Item` or `None`. |
| `search(namespace_prefix, query=None, filter=None, limit=10, offset=0)` | Search. Use `query` for semantic search, `filter` for key-value filters. |
| `delete(namespace, key)` | Remove item. |

## Index Config

```python
index={
    "dims": 768,           # required: embedding dimension
    "embed": Embeddings,   # required: LangChain Embeddings or callable
    "fields": ["text"],   # optional: fields to embed (default ["$"] = entire value)
}
```

## Run Examples

```bash
make langgraph-example   # Store/search demo
make langgraph-agent     # Chatbot that remembers user facts
```

## See Also

- [Design doc](../design_langgraph_memory.md)
- [integrations/langgraph/README.md](../../integrations/langgraph/README.md)
