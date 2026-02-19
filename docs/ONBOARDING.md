# Onboarding — Get Started with Agent Memory DB

**New to Agent Memory DB?** This guide gets you from zero to a working agent with episodic memory in ~10 minutes.

## What You'll Build

An agent that remembers past interactions and retrieves similar experiences when making decisions — "RAG over agent experiences" instead of documents.

## Prerequisites

- **Rust** (for the core library) or **Python 3.8+** (for bindings)
- No API keys required for local development (examples use stub embeddings)

## Option A: Python (Fastest)

```bash
# 1. Clone and build Python bindings
cd agent_mem_db/python
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --release

# 2. Run the personal assistant (interactive)
python -m examples.apps.personal_assistant
# Try: "I prefer dark mode" then "what do I prefer?"
```

## Option B: Rust

```bash
cd agent_mem_db
cargo run --example agent_sim
```

## Option C: LangGraph Integration

```bash
cd agent_mem_db
make langgraph-agent
# Chatbot that remembers facts you share
```

## Next Steps

| Goal | Doc |
|------|-----|
| Understand the concept | [overview.md](overview.md) |
| Run evaluation | [eval_results.md](eval_results.md), `make eval` |
| Use with LangChain | [integrations/langchain.md](integrations/langchain.md) |
| Use with LangGraph | [integrations/langgraph.md](integrations/langgraph.md) |
| Deploy the HTTP server | [design_hosted_memory.md](design_hosted_memory.md), `make docker-build` |
| Contribute | [CONTRIBUTING.md](../CONTRIBUTING.md) |

## Common Issues

- **Python 3.13+**: Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before `maturin develop`
- **Dimension mismatch**: Embedding length must match `AgentMemDB(dim)` — e.g. 384 for many sentence embedders
- **No episodes on query**: Store at least one episode before querying
