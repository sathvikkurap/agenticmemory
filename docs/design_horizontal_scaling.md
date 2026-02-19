# Design: Horizontal Scaling (Phase 4)

**Status:** Option A (shared storage) — Helm supports ReadWriteMany PVC for multi-replica  
**Related:** [design_roadmap.md](design_roadmap.md), [design_hosted_memory.md](design_hosted_memory.md)

## Goal

Scale the Hosted Memory Cloud beyond a single node: multiple server replicas, shared storage, tenant sharding. Support high availability and larger tenant counts.

## Current State (Phase 1–3)

- Single-node HTTP server
- Per-tenant AgentMemDB or AgentMemDBDisk in process
- Optional disk persistence via `AGENT_MEM_DATA_DIR`
- Stateless HTTP; state in memory or local disk

## Scaling Dimensions

| Dimension | Approach |
|-----------|----------|
| **More tenants** | Shard tenants across nodes; route by tenant_id |
| **Larger tenant data** | Offload episodes to object store; index in Redis or dedicated vector DB |
| **Higher QPS** | Multiple replicas behind load balancer; read replicas |
| **HA** | Replication, failover, shared storage |

## Architecture Options

### Option A: Stateless + Shared Storage

- Episodes in S3/GCS (or NFS); index rebuilt on pod start from object store.
- **Pros:** Simple; no new dependencies.
- **Cons:** Cold start slow for large tenants; no cross-pod caching.

### Option B: Tenant Sharding

- Route tenant_id → shard (hash or consistent hashing).
- Each shard = one or more pods; shard owns a subset of tenants.
- **Pros:** Bounded memory per shard; horizontal scale.
- **Cons:** Rebalancing on scale; need routing layer.

### Option C: External Vector DB

- Episodes + embeddings in Qdrant, Milvus, Pinecone, etc.
- Server becomes thin API layer.
- **Pros:** Battle-tested scale; managed options.
- **Cons:** New dependency; possible latency/cost; different semantics (e.g., no in-process prune).

### Option D: Redis + Object Store

- Episodes in S3; HNSW/vectors in Redis (e.g., Redis Stack with vector search).
- **Pros:** Fast index; familiar stack.
- **Cons:** Redis memory limits; need to sync episodes ↔ index.

## Recommendation

**First slice:** Option A — stateless pods + shared storage (S3/NFS). Each pod replays from shared `AGENT_MEM_DATA_DIR` (NFS) or fetches from S3 on tenant first access. No new infra; extends current disk mode.

**Later:** Option B (sharding) when tenant count grows; Option C if teams prefer managed vector DB.

## Implementation Sketch (Option A)

1. **Shared data dir:** Mount S3 via s3fs, or NFS, at `AGENT_MEM_DATA_DIR`.
2. **Pod identity:** Optional pod label for logging; no stateful identity.
3. **Load balancer:** Round-robin or least-connections; any pod can serve any tenant.
4. **Cold start:** On first request for tenant, load from shared path. Replay or checkpoint load as today.

## Out of Scope

- Strong consistency across replicas (eventual consistency acceptable)
- Cross-tenant queries
- Embedding model hosting
