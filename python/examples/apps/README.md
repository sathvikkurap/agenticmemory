# Full Application Examples

Standalone CLI applications that use AgentMemDB for episodic memory.

## Coding Assistant

Remembers code snippets, preferences, and patterns. Retrieves similar context when you ask questions.

```bash
make coding-assistant
# Or: cd python && .venv/bin/python -m examples.apps.coding_assistant
```

- **Store:** `remember: I prefer Python 3.11 and use type hints`
- **Store:** `remember: here's how I sort: sorted(items, key=lambda x: x.name)`
- **Query:** `how do I sort a list?`
- **Retention:** `prune: 50` (keep 50 most recent), `prune reward: 20` (keep 20 highest-reward), `stats` (show count)

Memory persists to `~/.agent_mem_db/coding_assistant.json`.

## Personal Assistant

Remembers preferences (I like X, I prefer Y) and retrieves them when answering questions.

```bash
make personal-assistant
# Or: cd python && .venv/bin/python -m examples.apps.personal_assistant
```

- **Store:** `I prefer dark mode`
- **Store:** `I'm vegetarian`
- **Query:** `what do I prefer for display?`
- **Retention:** `prune: 30` (keep 30 most recent), `stats` (show count)

Memory persists to `~/.agent_mem_db/personal_assistant.json`.

## Prerequisites

- Python bindings built: `cd python && maturin develop`
- Or use `make python-dev` from repo root
