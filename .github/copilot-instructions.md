You are helping build a Rust crate called `agent_mem_db` on macOS Apple Silicon (M3).

Core goal:
- Implement a fast in-memory agent memory database:
  - Store `Episode` structs with (id, task_id, state_embedding: Vec<f32>, reward, metadata).
  - Use an HNSW index for vector search.
  - Provide `store_episode` and `query_similar` APIs.

Constraints:
- Code must compile with stable Rust on aarch64-apple-darwin.
- Use the dependencies listed in Cargo.toml (serde, uuid, thiserror, hnswx, anyhow, rand).
- Favor clear, small functions and exhaustive error handling.
- Do not add network or cloud dependencies.

Your job:
- When I ask for changes, update existing code in-place instead of rewriting entire files.
- Prefer adding tests and examples to validate behavior.
- Suggest concrete next steps at the end of each response.

Here’s a long-term “mission prompt” you can give VS Code/Copilot/Cursor as project instructions so it has a clear north star and can keep executing toward it.

You can paste this into your project-level instructions (or a PROJECT_GOAL.md you reference often):

Long-Term Project Goal for agent_mem_db

You are helping build a production-quality Agent Memory Database (AMDB) in Rust that runs on my Mac M3 and later in real deployments.

Vision
Create an embeddable database engine that lets LLM-based agents:

Store their experiences as structured episodes:

Situation (state embedding),

Actions taken,

Rewards/outcomes,

Metadata (user, task, tools used).

Query those episodes later to answer:

“What happened in similar situations before?”

“What did I try that failed?”

“What worked best last time?”

This is like RAG for experiences instead of documents, and it will be used together with normal RAG (docs) in agents.

Core Requirements
Over time, the system should:

Core Rust library

Expose stable public API:

Episode

AgentMemDB::new(dim: usize)

store_episode(...)

query_similar(...)

Use:

In-memory storage for episodes.

HNSW or similar approximate nearest neighbor index for embeddings.

Be fully compatible with macOS Apple Silicon (M3) and Linux.

Query behavior

Given a query embedding and filters (e.g., min_reward, task_id), quickly return the top-k most relevant episodes.

Correctness:

Validate embedding dimensions.

Handle errors explicitly (custom error type).

Performance:

Reasonable latency (< 50–100 ms for ~10k episodes on M3).

Quality & safety

Clean, idiomatic Rust.

All code compiles with cargo build on aarch64-apple-darwin.

cargo fmt, cargo clippy, and cargo test should pass.

Clear tests for:

Storing and retrieving episodes.

Reward filtering.

Dimension mismatch handling.

Developer experience

Examples showing:

How to create a DB instance.

How to add episodes.

How to query similar high-reward episodes.

Simple documentation in README.md explaining the concept and basic usage.

Autonomy Instructions
From now on, when I ask for changes or improvements, you should:

Respect the long-term goal above.

Work incrementally:

Improve the existing code instead of rewriting entire files unless explicitly asked.

After each change or suggestion, propose the next 1–2 most important steps to move toward:

Better performance,

Better tests,

Better ergonomics (e.g., persistence, Python bindings, LangChain/LangGraph integration).

Do not introduce network/cloud dependencies or GPU-specific code unless I explicitly request it.

Your priorities, in order:

Make the current crate correct and well-tested.

Make it fast and clean.

Make it easy to integrate with LLM agents later.