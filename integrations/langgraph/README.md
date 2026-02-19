# AgentMemDB LangGraph Integration

LangGraph `BaseStore` backend using AgentMemDB for episodic long-term memory with HNSW semantic search.

## Installation

```bash
pip install -e .  # from this directory
# Also install agent_mem_db_py (from agent_mem_db/python)
```

## Usage

```python
from langchain_core.embeddings import Embeddings
from agent_mem_db_langgraph import AgentMemDBStore

# Use with any LangChain Embeddings (e.g., OpenAI, HuggingFace)
store = AgentMemDBStore(
    index={
        "dims": 768,
        "embed": my_embeddings,
        "fields": ["text"],  # optional, default ["$"] embeds entire value
    }
)

# Store memories
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "Python is my favorite language"})

# Get by key
item = store.get(("user_123", "memories"), "1")

# Semantic search
results = store.search(
    ("user_123", "memories"),
    query="What programming languages do I like?",
    limit=5
)
```

## LangGraph Integration

Pass the store when compiling your graph:

```python
from langgraph.graph import StateGraph

graph = StateGraph(State).add_node("agent", agent_node).add_edge(...)
app = graph.compile(store=store)  # AgentMemDBStore
```

Inside nodes, use `get_store()` to access the store:

```python
from langgraph.config import get_store

def agent_node(state):
    store = get_store()
    results = store.search(("user_id", "memories"), query=state["message"], limit=5)
    ...
```

## Example: Chatbot with Memory

Run the full agent example (chatbot that remembers user facts):

```bash
make langgraph-agent
```
