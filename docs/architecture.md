# Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent / Application                            │
│  (embeds state → query_similar → policy → store_episode)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentMemDB (Rust core)                           │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │  episodes   │  │  key_to_uuid     │  │  IndexBackend               │  │
│  │  HashMap    │  │  (key → id)      │  │  ├─ HNSW (hnswx) [default]  │  │
│  │  (id → ep)  │  │                  │  │  └─ Exact (brute-force)      │  │
│  └─────────────┘  └──────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ├── save/load (JSON)
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AgentMemDBDisk (optional)  │  Append-only JSONL + meta.json             │
│  Index rebuilt on open       │  For datasets larger than RAM              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Bindings & Integrations

```
                    ┌─────────────────┐
                    │   Rust core     │
                    │  (agent_mem_db) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Python       │   │  Node.js      │   │  (future: Go) │
│  (PyO3)       │   │  (napi-rs)   │   │               │
└───────┬───────┘   └───────────────┘   └───────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  LangChain    │  │  LangGraph   │  │  Raw Python   │
│  VectorStore  │  │  BaseStore   │  │  agent_mem_db │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Data Flow: Agent Loop

```
  ┌──────────┐     embed      ┌─────────────┐    query_similar    ┌─────────────┐
  │  State   │ ─────────────► │  embedding  │ ──────────────────► │  Episodes   │
  └──────────┘                 └─────────────┘                    │  (top_k)    │
       │                              │                             └──────┬──────┘
       │                              │                                    │
       │                              │ store_episode                      │
       │                              │ ◄──────────────────────────────────┘
       ▼                              │
  ┌──────────┐     policy      ┌──────┴──────┐
  │  Action  │ ◄────────────── │   Context   │  (retrieved episodes + state)
  └──────────┘                 └────────────┘
```

## Episode Structure

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Unique identifier |
| task_id | str | User-defined task label |
| state_embedding | Vec<f32> | Vector for similarity search |
| reward | f32 | Scalar outcome |
| metadata | JSON | Arbitrary key-value |
| timestamp | i64? | Unix ms (optional) |
| tags | Vec<String>? | For filtering (optional) |
| steps | Vec<EpisodeStep>? | Per-step trace (optional) |

## Index Backends

| Backend | Use case | Query complexity |
|---------|----------|------------------|
| HNSW | Default, large datasets | O(log n) approximate |
| Exact | Small datasets, correctness-critical | O(n) exact |

See [design_pluggable_index.md](design_pluggable_index.md) for details.
