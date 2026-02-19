#!/usr/bin/env python3
"""
Synthetic episode generator for scale testing.
Generates episodes with deterministic embeddings for reproducible benchmarks.
"""

import hashlib
import random
from typing import Iterator


def embed_text(text: str, dim: int = 16) -> list[float]:
    """Deterministic embedding from text."""
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    random.seed(h)
    return [random.uniform(-1, 1) for _ in range(dim)]


def generate_episodes(n: int, dim: int = 768) -> Iterator[dict]:
    """Yield n synthetic episodes as dicts (task_id, state_embedding, reward)."""
    for i in range(n):
        text = f"episode_{i}_task_{i % 100}_reward_{random.uniform(-1, 1):.2f}"
        yield {
            "task_id": f"task_{i}",
            "state_embedding": embed_text(text, dim),
            "reward": random.uniform(-1.0, 1.0),
        }


def main():
    import agent_mem_db_py as db
    n = 10_000
    dim = 16
    mem = db.AgentMemDB(dim)
    for ep_data in generate_episodes(n, dim):
        ep = db.Episode(
            task_id=ep_data["task_id"],
            state_embedding=ep_data["state_embedding"],
            reward=ep_data["reward"],
        )
        mem.store_episode(ep)
    print(f"Stored {n} episodes. Query test:")
    q = embed_text("episode_5_task_5", dim)
    results = mem.query_similar(q, min_reward=0.0, top_k=3)
    for r in results:
        print(f"  {r.task_id} reward={r.reward}")


if __name__ == "__main__":
    main()
