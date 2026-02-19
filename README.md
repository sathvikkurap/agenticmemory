# Agent Memory DB

**Episodic memory for AI agents.** Store experiences, retrieve by similarity—so your agent remembers what matters instead of losing it to context truncation.

```text
97.5%  recall success (vs 85% truncated history) · 88.9% on long-history tasks (vs 33.3%)
~200µs query latency · 10k+ inserts/sec · Rust · Python · Node · Go · LangChain · LangGraph
```

---

## Why this exists

LLMs have fixed context windows. Most apps keep only the **last N tokens**. When the user said the important thing 50 turns ago, it’s already gone—and the model can’t answer.

Agent Memory DB gives your agent **episodic memory**: store each experience as a vector, retrieve the **most similar** past experiences when answering. So recall works by **relevance**, not recency.

| Approach | Recall success (40-task eval) | Long-history tasks |
|----------|------------------------------|---------------------|
| No memory | 7.5% | 0% |
| Truncated history (last 256 tokens) | 85% | 33.3% |
| **Agent Memory DB** | **97.5%** | **88.9%** |

[Full eval results](docs/eval_results.md) · [When it helps](docs/case_study_preference_qa.md)

---

## Quick start

**Rust**

```rust
use agent_mem_db::{AgentMemDB, Episode};

let mut db = AgentMemDB::new(768);  // 768-dim embeddings
let ep = Episode::new("task_1", vec![0.0; 768], 1.0);
db.store_episode(ep).unwrap();
let similar = db.query_similar(&vec![0.0; 768], 0.0, 5).unwrap();
```

**Python** (after `cd python && maturin develop --release`)

```python
import agent_mem_db_py as agent_mem_db

db = agent_mem_db.AgentMemDB(768)
db.store_episode(agent_mem_db.Episode(task_id="task_1", state_embedding=[0.0] * 768, reward=1.0))
similar = db.query_similar([0.0] * 768, min_reward=0.0, top_k=5)
```

You bring your own embeddings (OpenAI, Cohere, local model—any 768-d vector). The library handles storage and similarity search.

---

## Features

- **HNSW vector search** — Fast approximate nearest neighbor; exact index for small datasets
- **Disk persistence** — Append log + optional checkpoint for fast restart
- **Retention** — Prune by time, keep newest N, or keep top by reward
- **Filters** — Query by tags, time range, task_id prefix
- **Bindings** — Rust, Python, Node.js, Go
- **Integrations** — [LangChain](integrations/langchain/) VectorStore, [LangGraph](integrations/langgraph/) memory store
- **HTTP server** — Multi-tenant API with auth and rate limiting; Docker & Helm

---

## Install

| Stack | Command |
|-------|---------|
| **Rust** | `cargo add agent_mem_db` (or clone and use path dep) |
| **Python** | `cd python && maturin develop --release` |
| **Node** | `cd node && npm install && npm run build` |
| **Go** | Build C API first: `cargo build -p agent_mem_db_capi --release` then `cd go && go build ./...` |

See [Onboarding](docs/ONBOARDING.md) for a 10-minute setup. See [Architecture](docs/architecture.md) and [Comparison](docs/COMPARISON.md) (vs vector DBs, naive memory, frameworks).

---

## Performance (proven)

*MacBook Pro (M3), 10k episodes, 768 dimensions.*

| Metric | Value |
|--------|--------|
| Query (top-10) | **~200 µs** (~5k queries/sec) |
| Insert 10k episodes | **~925 ms** (~11k inserts/sec) |
| Save to disk | **~227 ms** |
| Load from disk | **~1.26 s** (index rebuild) |

```bash
cargo bench --bench agent_mem_db_bench -- --nocapture
```

Details: [BENCHMARKS.md](BENCHMARKS.md).

---

## Examples

```bash
cargo run --example agent_sim          # Rust agent loop
cargo run --example disk_checkpoint   # Disk + checkpoint
make langchain-example                 # LangChain VectorStore
make langgraph-agent                  # LangGraph memory
make coding-assistant                  # Full CLI app (Python)
```

---

## Documentation

- [Onboarding](docs/ONBOARDING.md) — Get running in ~10 minutes
- [Architecture](docs/architecture.md) — Components and data flow
- [Comparison](docs/COMPARISON.md) — vs vector DBs, naive memory, LangChain/LangGraph
- [Eval results](docs/eval_results.md) — Success rates, task types, reproducibility
- [Integrations](docs/integrations/) — LangChain, LangGraph
- [Deploy](deploy/README.md) — Docker, Helm, Terraform

Full index: [docs/README.md](docs/README.md).

---

## License

MIT
