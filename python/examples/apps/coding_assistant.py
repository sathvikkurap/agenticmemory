#!/usr/bin/env python3
"""
Coding assistant with episodic memory.

Remembers code snippets, preferences, and patterns. When you ask a question,
it retrieves similar past interactions to inform the response.

Usage:
  python -m examples.apps.coding_assistant
  # Or: make coding-assistant (from repo root)

Example session:
  > remember: I prefer Python 3.11 and use type hints
  > remember: here's how I sort a list: sorted(items, key=lambda x: x.name)
  > how do I sort a list?
  [Recalls similar episodes and responds with context]
  > prune: 50          # keep only 50 most recent memories
  > prune reward: 20   # keep only 20 highest-reward memories
  > stats              # show episode count
"""

import sys
import time
from pathlib import Path

# Add parent for agent_mem_db_py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import agent_mem_db_py as agent_mem_db
except ImportError:
    print("Build Python bindings first: cd python && maturin develop")
    sys.exit(1)

DIM = 16
MEM_PATH = Path.home() / ".agent_mem_db" / "coding_assistant.json"


def embed(text: str) -> list[float]:
    """Simple deterministic embedding (no API key)."""
    import hashlib
    words = [w.lower() for w in text.split() if len(w) > 1]
    vec = [0.0] * DIM
    for i, w in enumerate(words):
        h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
        for j in range(DIM):
            vec[j] += ((h >> (j + i)) & 1) * 0.2 - 0.1
    norm = (sum(x * x for x in vec)) ** 0.5
    if norm > 1e-6:
        vec = [x / norm for x in vec]
    return vec


def load_db() -> agent_mem_db.AgentMemDB:
    """Load or create the memory DB."""
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MEM_PATH.exists():
        return agent_mem_db.AgentMemDB.load_from_file(str(MEM_PATH))
    return agent_mem_db.AgentMemDB(DIM)


def save_db(db: agent_mem_db.AgentMemDB) -> None:
    """Persist the memory DB."""
    db.save_to_file(str(MEM_PATH))


def main() -> None:
    db = load_db()
    print("Coding assistant (type 'quit' to exit, 'remember: ...' to store)")
    print("Commands: stats | prune: <n> | prune reward: <n>")
    print("=" * 50)

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() == "quit":
            save_db(db)
            print("Goodbye.")
            break

        if line.lower() == "stats":
            count = len(db.query_similar(embed(""), min_reward=-1.0, top_k=10000))
            print(f"Memories: {count} episodes")
            continue

        if line.lower().startswith("prune:"):
            try:
                n = int(line[6:].strip())
                removed = db.prune_keep_newest(n)
                save_db(db)
                print(f"✓ Pruned to {n} most recent. Removed {removed}.")
            except ValueError:
                print("Usage: prune: <n>  (e.g. prune: 50)")
            continue

        if line.lower().startswith("prune reward:"):
            try:
                n = int(line[13:].strip())
                removed = db.prune_keep_highest_reward(n)
                save_db(db)
                print(f"✓ Pruned to {n} highest-reward. Removed {removed}.")
            except ValueError:
                print("Usage: prune reward: <n>  (e.g. prune reward: 20)")
            continue

        if line.lower().startswith("remember:"):
            content = line[9:].strip()
            if content:
                ep = agent_mem_db.Episode(
                    task_id=f"mem_{hash(content) % 100000}",
                    state_embedding=embed(content),
                    reward=1.0,
                    metadata={"text": content, "type": "snippet"},
                    timestamp=int(time.time() * 1000),
                )
                db.store_episode(ep)
                save_db(db)
                print("✓ Stored.")
            else:
                print("Usage: remember: <your text>")
        else:
            # Query similar memories
            query_emb = embed(line)
            similar = db.query_similar(query_emb, min_reward=0.0, top_k=3)
            if similar:
                context = " | ".join(
                    ep.metadata.get("text", str(ep.metadata))[:60]
                    for ep in similar
                    if isinstance(ep.metadata, dict)
                )
                print(f"[Context: {context}...]")
                print(f"Response: Based on your preferences, try: {line[:60]}...")
            else:
                print("No relevant memories. Ask me to remember something first.")


if __name__ == "__main__":
    main()
