import agent_mem_db_py as agent_mem_db

# Minimal Python example for the PyO3 bindings.
# Build and install the Python extension locally (see python/README.md) before running.

# Create DB (16-dim embeddings)
db = agent_mem_db.AgentMemDB(16)

# Create and store two episodes (the minimal bindings do not expose EpisodeStep)
ep1 = agent_mem_db.Episode(
    task_id="t1",
    state_embedding=[0.1] * 16,
    reward=1.0,
    metadata={"foo": 1},
)
ep2 = agent_mem_db.Episode(
    task_id="t2",
    state_embedding=[0.2] * 16,
    reward=0.5,
    metadata={"bar": 2},
)
db.store_episode(ep1)
db.store_episode(ep2)

# Query similar episodes (min_reward=0.0, top_k=2)
eps = db.query_similar([0.1] * 16, min_reward=0.0, top_k=2)
for ep in eps:
    print(f"Episode: id={ep.id}, task_id={ep.task_id}, reward={ep.reward}")

# Save and load the DB (local path)
db.save_to_file("/tmp/py_mem.json")
db2 = agent_mem_db.AgentMemDB.load_from_file("/tmp/py_mem.json")
print(f"Loaded DB size: {len(db2.query_similar([0.1] * 16, 0.0, 10))} episodes.")
