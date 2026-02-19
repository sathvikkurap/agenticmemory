# Design: Hosted Memory Cloud

**Status:** Phase 1 implemented  
**Related:** [design_roadmap.md](design_roadmap.md) § III

**Implementation:** `server/` — `make server`, `make server-run`, or `make docker-build` / `make docker-run`

## Goal

Multi-tenant, horizontally scalable memory service exposing AgentMemDB over HTTP and/or gRPC. Enables teams to use episodic memory without deploying their own infrastructure.

## Core API

Mirror the Rust core operations:

| Operation | HTTP | gRPC | Description |
|-----------|------|------|-------------|
| StoreEpisode | `POST /v1/episodes` | `StoreEpisode` | Store one episode |
| StoreEpisodes | `POST /v1/episodes/batch` | `StoreEpisodes` | Batch store |
| QuerySimilar | `POST /v1/query` | `QuerySimilar` | Similarity search |
| Save | `POST /v1/save` | `Save` | Persist to backend storage |
| Load | `POST /v1/load` | `Load` | Load from backend |
| PruneOlderThan | `POST /v1/prune/older-than` | — | Remove episodes older than cutoff |
| PruneKeepNewest | `POST /v1/prune/keep-newest` | — | Keep only n most recent episodes |
| PruneKeepHighestReward | `POST /v1/prune/keep-highest-reward` | — | Keep only n highest-reward episodes |
| Checkpoint | `POST /v1/checkpoint` | — | Persist ExactIndex checkpoint (disk mode only) |

### Request/Response Schemas (JSON)

**StoreEpisode**
```json
{
  "task_id": "string",
  "state_embedding": [0.1, ...],
  "reward": 0.9,
  "metadata": {},
  "timestamp": 1234567890,
  "tags": ["coding"]
}
```
Response: `{"id": "uuid"}`

**QuerySimilar**
```json
{
  "query_embedding": [0.1, ...],
  "min_reward": 0.0,
  "top_k": 5,
  "tags_any": ["coding"],
  "tags_all": ["coding", "python"],
  "task_id_prefix": "task_",
  "time_after": 1234567890,
  "time_before": 1234567999
}
```
Response: `{"episodes": [{...}, ...]}`

**PruneOlderThan**
```json
{ "timestamp_cutoff_ms": 1700000000000 }
```
Response: `{"removed": 42}`

**PruneKeepNewest**
```json
{ "n": 1000 }
```
Response: `{"removed": 150}`

**PruneKeepHighestReward**
```json
{ "n": 500 }
```
Response: `{"removed": 200}`

## Multi-Tenancy

- **Namespace:** Each tenant has a `tenant_id` (or `api_key` → tenant). All operations are scoped to that tenant.
- **Isolation:** Episodes are stored per-tenant; queries never cross tenants.
- **Rate limiting:** Per-tenant limits on requests/sec and storage.

## Authentication

- **API key:** `Authorization: Bearer <key>` or `X-API-Key: <key>`
- **Future:** OAuth2, JWT for user-level access within a tenant

## Storage Backend

- **In-memory (default):** Per-tenant AgentMemDB in RAM. Save/Load to JSON files.
- **Disk-backed:** When `AGENT_MEM_DATA_DIR` is set, each tenant uses AgentMemDBDisk with ExactIndex checkpoint. Data stored under `data_dir/<tenant_id>/` (episodes.jsonl, meta.json, exact_checkpoint.json). Call `POST /v1/checkpoint` to persist checkpoint for fast restart.
- **Future:** Distributed storage (e.g., S3 for episodes, Redis for index), sharding by tenant.

## Implementation Phases

| Phase | Scope |
|-------|-------|
| 1 ✓ | HTTP server (axum), in-process AgentMemDB per tenant, API key auth |
| 2 ✓ | Rate limiting, metrics |
| 3 ✓ | Disk persistence when AGENT_MEM_DATA_DIR set; POST /v1/checkpoint |
| 4 | Horizontal scaling — Option A (multi-replica + NFS) via Helm; see [design_horizontal_scaling.md](design_horizontal_scaling.md) |

## Rate Limiting

When `AGENT_MEM_RATE_LIMIT` is set, per-tenant rate limiting is enabled. Uses fixed-window: N requests per tenant per window. Returns 429 Too Many Requests when exceeded.

## Metrics & Logging

- **`GET /metrics`** — Prometheus-style metrics: `agent_mem_requests_total`, `agent_mem_store_episodes_total`, `agent_mem_query_total`, `agent_mem_tenants_active`
- **`GET /dashboard`** — Simple web UI: health, usage (requests, episodes, queries, tenants), config (dim, rate limit, audit, data dir)
- **Request logging** — TraceLayer logs method, URI, status, latency (set `RUST_LOG=info`)

## Audit Log

When `AGENT_MEM_AUDIT_LOG` is set to a file path, the server appends JSONL entries for sensitive operations:

- **store_episode** — task_id, episode_count=1
- **store_episodes** — episode_count
- **query** — read access
- **save** — path
- **load** — path

Each line: `{"ts":"...","tenant_id":"...","op":"...","task_id":"...","episode_count":...,"path":"..."}` (fields omitted when not applicable).

## Docker

```bash
make docker-build   # Build image agent-mem-server
make docker-run    # Run with AGENT_MEM_API_KEY=dev-secret
# Or: docker run -p 8080:8080 -e AGENT_MEM_API_KEY=secret agent-mem-server
```

## Kubernetes (Helm, Terraform)

- **Helm:** `deploy/helm/agent-mem-server/` — Deployment, Service, optional PVC, Ingress
- **Terraform:** `deploy/terraform/` — Deploys Helm chart to existing cluster
- See `deploy/README.md` for usage

## Environment (Phase 1)

| Var | Default | Description |
|-----|---------|-------------|
| `AGENT_MEM_API_KEY` | (none) | Required API key; if unset, all keys accepted (dev only) |
| `AGENT_MEM_DIM` | 384 | Default embedding dimension for new tenants |
| `AGENT_MEM_DATA_DIR` | (none) | When set, use disk-backed storage per tenant (AgentMemDBDisk + checkpoint) |
| `AGENT_MEM_RATE_LIMIT` | (none) | Max requests per tenant per window (e.g. 100) |
| `AGENT_MEM_RATE_WINDOW_SECS` | 60 | Rate limit window in seconds |
| `AGENT_MEM_AUDIT_LOG` | (none) | File path for JSONL audit log (store, query, save, load) |

## Out of Scope (First Slice)

- Real-time replication
- Cross-tenant search
- Custom embedding models (client sends embeddings)
