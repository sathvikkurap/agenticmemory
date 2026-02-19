# Guide for AI Agents and Contributors

This document helps both human and AI contributors work effectively on the Agent Memory DB project.

## Coding Conventions

- **Rust:** `cargo fmt`, `cargo clippy --all-targets --all-features`
- **Python:** PEP 8; run `pytest` before committing
- **Docs:** Markdown; design docs in `docs/design_*.md`, integration guides in `docs/integrations/`

## Key Files

| File | Purpose |
|------|---------|
| `docs/design_roadmap.md` | Master roadmap, execution log |
| `SESSION_SUMMARY.txt` | Log of recent changes and next steps |
| `docs/design_*.md` | Design decisions and rationale |
| `docs/integrations/` | LangChain, LangGraph guides |
| `docs/research.md` | Research context |
| `CHANGELOG.md` | Version history |

## Project Structure

| Path | Contents |
|------|----------|
| `src/` | Rust core (lib.rs, index.rs, disk.rs) |
| `python/` | PyO3 bindings, examples, tests |
| `node/` | napi-rs Node bindings |
| `integrations/langchain/` | AgentMemDBVectorStore |
| `integrations/langgraph/` | AgentMemDBStore |
| `examples/` | Rust examples |
| `docs/` | Design docs, overview, eval, integrations |
| `benches/` | Criterion benchmarks |

## Agent Integration Patterns

**When to use episodic memory:** Tasks where the agent must recall information from earlier in a session (e.g., "Remember: user prefers X" followed by "What does user prefer?"). See [docs/agent_integration.md](docs/agent_integration.md).

**Rust:** `AgentMemDB::new(dim)`, `store_episode`, `query_similar`  
**Python:** `agent_mem_db_py.AgentMemDB`, `Episode`, `store_episode`, `query_similar`  
**LangChain:** `AgentMemDBVectorStore` for RAG  
**LangGraph:** `AgentMemDBStore` for long-term memory

## Running Tests and Examples

```bash
make test          # Rust (incl. proptest)
make python-test   # Python
make node-test     # Node
make langgraph-test
make eval          # Agent A/B evaluation
make bench         # Criterion benchmarks
```

## How to Log New Insights

1. Update `SESSION_SUMMARY.txt` after significant work
2. Add or update `docs/design_*.md` before large changes
3. Update `docs/design_roadmap.md` execution log when completing slices
4. Keep `CHANGELOG.md` updated for releases

## Constraints

- Do not break existing public APIs (Rust, Python, Node) without deprecation
- Prefer vertical slices: feature → example → tests → docs
- Keep benchmark code small and clear
