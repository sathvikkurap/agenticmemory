# Design: Async APIs (Rust + Python)

**Status:** Implemented (first slice)  
**Related:** [design_roadmap.md](design_roadmap.md)

## Goal

Provide async variants of AgentMemDB operations so callers can avoid blocking the event loop when performing CPU-bound work (HNSW search, index insert) or I/O (save/load).

## Current State

- **Rust:** All methods are synchronous. `store_episode`, `query_similar`, `save_to_file`, `load_from_file` block the calling thread.
- **Python:** PyO3 bindings expose sync methods. Calling from an async context requires `asyncio.to_thread()` or `run_in_executor`.

## Use Cases

- **Async web servers** — Axum/Actix handlers that call AgentMemDB without blocking the async runtime.
- **Async Python agents** — LangChain/LangGraph async chains that need non-blocking memory access.
- **Concurrent queries** — Multiple queries in flight without blocking each other.

## Approach

### Rust

**Option A: `spawn_blocking` wrapper crate**

Create `agent_mem_db_async` (or feature-gated async in main crate) that wraps sync API:

```rust
pub async fn store_episode_async(db: Arc<RwLock<AgentMemDB>>, ep: Episode) -> Result<(), AgentMemError> {
    tokio::task::spawn_blocking(move || {
        let mut guard = db.write().unwrap();
        guard.store_episode(ep)
    }).await.unwrap()
}
```

- **Pros:** No changes to core; clear separation.
- **Cons:** Caller must wrap DB in `Arc<RwLock<>>`; extra crate or feature.

**Option B: Async trait + default impl**

```rust
#[async_trait]
pub trait AgentMemDBAsync {
    async fn store_episode(&self, ep: Episode) -> Result<(), AgentMemError>;
    async fn query_similar(&self, emb: &[f32], opts: QueryOptions) -> Result<Vec<Episode>, AgentMemError>;
}
```

Implement for `Arc<RwLock<AgentMemDB>>` using `spawn_blocking`.

**Option C: `AgentMemDB` with `tokio::sync::RwLock`**

Replace `std::sync::Mutex` in Node bindings-style with `tokio::sync::RwLock`. Async methods would `.read().await` / `.write().await` and then call sync logic. But the sync logic (HNSW) still blocks — we'd need `spawn_blocking` inside. So this doesn't simplify much.

**Recommendation:** Option A — minimal async wrapper. Keep core sync; async is a thin layer.

### Python

**Option A: `asyncio.to_thread` wrapper**

Add async methods that delegate to sync:

```python
async def store_episode_async(self, episode: Episode) -> None:
    await asyncio.to_thread(self.store_episode, episode)
```

- **Pros:** Simple, no changes to Rust extension.
- **Cons:** Requires Python 3.9+ for `to_thread`. For 3.8: `loop.run_in_executor(None, lambda: self.store_episode(episode))`.

**Option B: Separate async class**

`AgentMemDBAsync` that holds a reference to sync `AgentMemDB` and wraps all methods. Same idea as Option A.

**Recommendation:** Option A — add `store_episode_async`, `query_similar_async`, `save_to_file_async`, `load_from_file_async` to `AgentMemDB` (or a mixin). Use `asyncio.to_thread` when available.

## API Surface (First Slice)

### Rust

- New crate or feature: `agent_mem_db::async_api` or `agent_mem_db_async`
- `store_episode_async(db: Arc<RwLock<AgentMemDB>>, ep: Episode) -> Result<(), AgentMemError>`
- `query_similar_async(db: Arc<RwLock<AgentMemDB>>, emb: Vec<f32>, opts: QueryOptions) -> Result<Vec<Episode>, AgentMemError>`
- `save_to_file_async(db: Arc<RwLock<AgentMemDB>>, path: PathBuf) -> Result<(), AgentMemError>`
- `load_from_file_async(path: PathBuf) -> Result<Arc<RwLock<AgentMemDB>>, AgentMemError>`

Dependencies: `tokio` with `rt` and `sync` features.

### Python

- `async def store_episode_async(self, episode: Episode) -> None`
- `async def query_similar_async(self, state_embedding, min_reward, top_k, **opts) -> List[Episode]`
- `async def save_to_file_async(self, path: str) -> None`
- `@classmethod async def load_from_file_async(cls, path: str) -> AgentMemDB`

Implementation: `asyncio.to_thread(self.store_episode, episode)` etc.

## Backward Compatibility

- Sync API unchanged.
- Async is additive. No breaking changes.

## Implementation (First Slice)

### Rust

- Feature `async` in `agent_mem_db` crate; `agent_mem_db::async_api` module.
- `store_episode_async`, `query_similar_async`, `save_to_file_async`, `load_from_file_async`.
- Example: `cargo run --example async_example --features async`.

### Python

- `AgentMemDBAsync` class in `agent_mem_db_py.async_api`; re-exported from `agent_mem_db_py`.
- Wraps sync `AgentMemDB`; uses `asyncio.to_thread` (Python 3.9+) or `run_in_executor` (3.8).
- Test: `tests/test_basic.py::test_async_api`.

## Out of Scope

- Async index implementations (HNSW is sync; we wrap with spawn_blocking).
- Async I/O in core (save/load use std::fs; spawn_blocking covers that).
- Node.js async — napi-rs can expose Promises; separate design.
