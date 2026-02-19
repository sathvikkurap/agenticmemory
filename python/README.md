db = AgentMemDB(dim=768)
ep = Episode(task_id="foo", state_embedding=np.zeros(768), reward=1.0)
db.store_episode(ep)
results = db.query_similar(np.zeros(768), min_reward=0.0, top_k=5)
db.save_to_file("mem.json")

# Python bindings for agent_mem_db

## Install

```bash
cd python
maturin develop  # or: pip install maturin && maturin develop
```

**Python 3.13+:** PyO3 supports up to 3.12. If you use 3.13 or 3.14, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before maturin:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin develop --release
```

## Usage

```python
import agent_mem_db_py as agent_mem_db

# Create DB
db = agent_mem_db.AgentMemDB(16)

# Create an episode with steps
steps = [
	agent_mem_db.EpisodeStep(index=0, action="move_a", observation="obs1", step_reward=0.2),
	agent_mem_db.EpisodeStep(index=1, action="move_b", observation="obs2", step_reward=0.3),
]
ep = agent_mem_db.Episode(
	task_id="t1",
	state_embedding=[0.1]*16,
	reward=1.0,
	steps=steps
)
db.store_episode(ep)

# Query similar
eps = db.query_similar([0.1]*16, min_reward=0.0, top_k=2)
for ep in eps:
	print(ep.task_id, ep.reward, len(ep.steps) if ep.steps else 0)

# Batch query
results = db.query_similar_batch([[0.1]*16, [0.2]*16], min_reward=0.0, top_k=2)

# Save/load
db.save_to_file("/tmp/py_mem.json")
db2 = agent_mem_db.AgentMemDB.load_from_file("/tmp/py_mem.json")
```

## Async API

For async contexts (e.g. async web servers, LangChain/LangGraph async chains), use `AgentMemDBAsync`:

```python
import asyncio
import agent_mem_db_py as agent_mem_db

async def main():
    db = agent_mem_db.AgentMemDB(16)
    async_db = agent_mem_db.AgentMemDBAsync(db)
    ep = agent_mem_db.Episode(task_id="t1", state_embedding=[0.1]*16, reward=1.0)
    await async_db.store_episode_async(ep)
    results = await async_db.query_similar_async([0.1]*16, min_reward=0.0, top_k=5)
    await async_db.save_to_file_async("/tmp/async_mem.json")
    db2 = await agent_mem_db.AgentMemDBAsync.load_from_file_async("/tmp/async_mem.json")

asyncio.run(main())
```

## Example: LangGraph-style Agent Loop

A simple agent loop using a stub LLM, a math tool, and AgentMemDB for memory:

```bash
python examples/langgraph_agent.py
```

This script demonstrates:
- Building a state string from user message + tool outputs
- Embedding state (with a deterministic stub)
- Querying AgentMemDB for similar episodes before acting
- Storing new episodes with steps after each task
- Running for 20 tasks in a loop

## Notes
- Embeddings must be list[float] (numpy.ndarray support planned).

## Testing

After building the Python bindings (see above), run tests with:

```bash
pytest tests/
```

This will run basic DB, query, and persistence tests.

## Publishing to PyPI

From the `python/` directory:

```bash
# Build wheel (release mode)
maturin build --release

# Publish to PyPI (requires PyPI token)
maturin publish
```

For PyPI upload you need:
- `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<pypi-token>` (or use `maturin publish` which handles auth)
- Or configure `~/.pypirc` with your credentials

To build without publishing: `maturin build --release` produces wheels in `target/wheels/`.
