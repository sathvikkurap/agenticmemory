#!/usr/bin/env python3
"""
Personal assistant with episodic memory.

Remembers preferences (I like X, I prefer Y) and retrieves them when
answering questions. Uses AgentMemDB for long-term storage.

Usage:
  python -m examples.apps.personal_assistant
  # Or: make personal-assistant (from repo root)

Example session:
  > I prefer dark mode
  > I'm vegetarian
  > what do I prefer for display?
  [Recalls "dark mode" and responds]
  > prune: 30    # keep 30 most recent preferences
  > stats        # show preference count
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import agent_mem_db_py as agent_mem_db
except ImportError:
    print("Build Python bindings first: cd python && maturin develop")
    sys.exit(1)

DIM = 16
MEM_PATH = Path.home() / ".agent_mem_db" / "personal_assistant.json"


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
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MEM_PATH.exists():
        return agent_mem_db.AgentMemDB.load_from_file(str(MEM_PATH))
    return agent_mem_db.AgentMemDB(DIM)


def save_db(db: agent_mem_db.AgentMemDB) -> None:
    db.save_to_file(str(MEM_PATH))


def main() -> None:
    db = load_db()
    print("Personal assistant (type 'quit' to exit)")
    print("Share preferences: 'I prefer X', 'I like Y', 'I'm Z'")
    print("Commands: stats | prune: <n>")
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
            print(f"Preferences: {count} stored")
            continue

        if line.lower().startswith("prune:"):
            try:
                n = int(line[6:].strip())
                removed = db.prune_keep_newest(n)
                save_db(db)
                print(f"✓ Kept {n} most recent. Removed {removed}.")
            except ValueError:
                print("Usage: prune: <n>")
            continue

        # Detect if user is sharing a preference (store) or asking (query)
        is_question = line.strip().endswith("?") or any(
            w in line.lower() for w in ("what", "how", "when", "which", "do i", "does")
        )
        preference_phrases = ("i like ", "i love ", "i prefer ", "i'm ", "i am ", "my ")

        if not is_question and any(line.lower().startswith(p) for p in preference_phrases):
            # Store preference
            ep = agent_mem_db.Episode(
                task_id=f"pref_{hash(line) % 100000}",
                state_embedding=embed(line),
                reward=1.0,
                metadata={"text": line, "type": "preference"},
                timestamp=int(time.time() * 1000),
            )
            db.store_episode(ep)
            save_db(db)
            print("✓ Remembered.")
        else:
            # Query similar memories
            query_emb = embed(line)
            similar = db.query_similar(query_emb, min_reward=0.0, top_k=3)
            if similar:
                answers = []
                for ep in similar:
                    if isinstance(ep.metadata, dict) and "text" in ep.metadata:
                        answers.append(ep.metadata["text"])
                if answers:
                    print(f"Based on what you've told me: {'; '.join(answers[:2])}")
                else:
                    print("I don't have relevant memories for that.")
            else:
                print("I don't have any preferences stored yet. Share something!")


if __name__ == "__main__":
    main()
