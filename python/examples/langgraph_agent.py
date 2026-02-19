"""Simple LangGraph-style agent demo using the minimal PyO3 bindings.

This demo is intentionally small: it stubs an LLM and a tool, shows where
`query_similar` is called to recall past episodes, and where `store_episode`
is called to persist a new episode.
"""

import agent_mem_db_py as agent_mem_db
import random
import hashlib

# --- Stub LLM and Tool ---
def fake_llm(prompt):
    # Simulate an LLM call (replace with OpenAI-compatible call if desired)
    return f"LLM response to: {prompt}"

def math_tool(x):
    return x * x

def embed_state(text, dim=16):
    # Deterministic pseudo-random embedding based on text hash
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    random.seed(h)
    return [random.uniform(-1, 1) for _ in range(dim)]

# --- Agent Loop ---
# Build the in-memory DB (16-dim embeddings)
db = agent_mem_db.AgentMemDB(16)
num_tasks = 20
for i in range(num_tasks):
    user_msg = f"Task {i}: What is {i} squared?"
    tool_out = math_tool(i)
    state = f"user: {user_msg}\ntool: {tool_out}"
    embedding = embed_state(state)
    # Query memory before acting
    # <-- Retrieval point: call `query_similar` to get relevant past episodes
    similar = db.query_similar(embedding, min_reward=0.0, top_k=3)
    n_sim = len(similar)
    best_reward = max((ep.reward for ep in similar), default=float('nan'))
    print(f"[Task {i}] Found {n_sim} similar, best_reward={best_reward}")
    # LLM "decides" next action
    llm_out = fake_llm(state)
    # Store metadata instead of explicit EpisodeStep objects (simpler Python binding)
    metadata = {"tool_out": tool_out, "llm_out": llm_out}
    ep = agent_mem_db.Episode(
        task_id=f"task_{i}",
        state_embedding=embedding,
        reward=1.0,
        metadata=metadata,
    )
    # <-- Storage point: persist this new episode into memory
    db.store_episode(ep)
print("Done. DB now contains:", len(db.query_similar(embed_state(""), 0.0, 100)), "episodes.")
