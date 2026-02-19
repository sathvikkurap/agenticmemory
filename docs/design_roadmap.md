# Agent Memory DB — Master Roadmap

**Status:** Living document  
**Last updated:** 2026-02

This roadmap guides the evolution of Agent Memory DB from a library into a complete agent memory platform. Work proceeds in vertical slices: each increment delivers working code, examples, tests, and docs.

## Principles

- **Backward compatibility:** Preserve existing APIs; document breaking changes and migration paths.
- **Vertical slices:** Each effort ends with working code + examples + tests + docs.
- **Design first:** For larger initiatives, create `docs/design_*.md` before heavy implementation.
- **Always buildable:** Keep the repo in a buildable, testable state.

---

## I. Core Engine Evolution (Rust + Python)

### 1.1 Pluggable Index Backends

**Goal:** Support multiple index backends (HNSW, exact, future: PQ, IVF, disk-backed).

**Status:** Done — [design_pluggable_index.md](design_pluggable_index.md).

**Increments:**
- [x] Design interface
- [x] IndexBackend enum; wrap HNSW and Exact
- [x] Add exact (brute-force) backend; `AgentMemDB::new_exact(dim)`
- [x] Python: `AgentMemDB.exact(dim)`
- [ ] Add PQ/IVF backends (future) — [design_pq_ivf.md](design_pq_ivf.md)

### 1.2 On-Disk / Hybrid Storage

**Goal:** Support episode sets that do not fit entirely in RAM.

**Status:** First slice done — [design_on_disk_storage.md](design_on_disk_storage.md).

**Increments:**
- [x] Design append-only log + index rebuild
- [x] AgentMemDBDisk: open/store/query, JSONL log, meta.json
- [x] Python bindings for AgentMemDBDisk
- [x] Incremental indexing first slice: ExactIndex checkpoint — [design_incremental_indexing.md](design_incremental_indexing.md)
- [x] Configurable retention and pruning (time-based, count-based, importance-based) — [design_retention_pruning.md](design_retention_pruning.md)

### 1.3 Rich Data Model and Query APIs

**Goal:** Structured metadata, multiple embedding spaces, expressive queries.

**Status:** First slice done — [design_rich_metadata.md](design_rich_metadata.md).

**Increments:**
- [x] Extend Episode: timestamp, tags (optional fields)
- [x] Add query filters: tags_any, time_after, time_before; QueryOptions
- [x] query_similar_with_options; Python kwargs (tags_any, time_after, time_before)
- [x] source, user_id (optional Episode fields, query filters)
- [x] tags_all, task_id_prefix (QueryOptions filters)
- [x] Combine similarity + constraints (recency tie-breaker when distance equal)
- [x] Async APIs (Rust + Python) — [design_async_apis.md](design_async_apis.md)

### 1.4 Reliability and Correctness

**Goal:** Property-based testing, chaos/stress tests, fuzzing.

**Increments:**
- [x] Proptest/QuickCheck for save/load invariants (tests/proptest_persist.rs, load_from_file_exact)
- [x] Concurrent read/write stress tests (tests/concurrent_stress.rs)
- [x] Fuzzing of episode inputs and metadata (tests/fuzz_episode.rs, proptest)

---

## II. Ecosystem SDKs & Integrations

### 2.1 Multi-Language Bindings

**Goal:** TypeScript/Node, Go, possibly Java/Kotlin.

**Status:** TypeScript/Node first slice done — [design_typescript_bindings.md](design_typescript_bindings.md).

**Increments:**
- [x] TypeScript/Node (napi-rs): AgentMemDb, AgentMemDbDisk, Episode, createEpisode, storeEpisode, querySimilar, prune*, checkpoint
- [x] Go (cgo + C-ABI): capi/ crate, go/ package, New/Store/Query/Save/Load/Prune*, DiskDB (OpenDisk, OpenDiskExactWithCheckpoint, Checkpoint)
- [x] Per-language README, examples, tests (Node test.js, LangGraph tests, Go example, main README bindings table)

### 2.2 Agent Framework Integrations

**Goal:** First-class integrations with LangChain/LangGraph, Autogen, OpenAI/Anthropic SDKs.

**Status:** LangChain VectorStore first slice done — [design_langchain_integration.md](design_langchain_integration.md).

**Increments:**
- [x] LangChain VectorStore: AgentMemDBVectorStore (add_texts, similarity_search, from_texts)
- [x] LangGraph memory adapter: AgentMemDBStore (put/get/search, BaseStore)
- [x] Example agents using AgentMemDB vs baselines (agent_example.py, langgraph-agent)
- [x] Docs per framework (docs/integrations/langchain.md, langgraph.md)

---

## III. Hosted Memory Cloud

**Goal:** Multi-tenant, horizontally scalable memory service.

**Status:** Design done — [design_hosted_memory.md](design_hosted_memory.md).

**Increments:**
- [x] Design HTTP/gRPC API (StoreEpisode, QuerySimilar, Save/Load)
- [x] Implement server using Rust core (Phase 1: axum, in-process per tenant, API key auth)
- [x] Multi-tenant isolation, auth, rate limiting (per-tenant fixed-window, AGENT_MEM_RATE_LIMIT)
- [x] Metrics, logging (Prometheus /metrics, TraceLayer request logging)
- [x] Auditing (audit log for sensitive ops)
- [x] Docker for deployment (Dockerfile, make docker-build, make docker-run)
- [x] Helm, Terraform for deployment (deploy/helm/, deploy/terraform/)

---

## IV. Evaluation Suite & Research

**Goal:** Robust eval framework, multiple benchmarks, documented wins.

**Status:** Config-driven eval first slice done — [design_config_driven_eval.md](design_config_driven_eval.md).

**Increments:**
- [x] Config-driven eval (eval_config.json/yaml, --config, VARIANT_REGISTRY)
- [x] Integrate/emulate known memory benchmarks (docs/MEMORY_BENCHMARKS.md, adapter design)
- [x] Extended metrics: p50/p95/p99 latency (per-task, in run_eval and JSON report)
- [x] Analysis scripts and notebooks (scripts/analyze_eval.py, notebooks/eval_analysis.ipynb)
- [x] Case study docs (agents where AgentMemDB wins) — docs/case_study_preference_qa.md

---

## V. Developer Experience & Community

**Goal:** Excellent docs, rich examples, clear contribution guidance.

**Increments:**
- [x] Reorganize docs: Getting Started, Integrations, Operations, Evaluation (docs/README.md index)
- [x] Architectural diagrams (docs/architecture.md)
- [x] Full application examples (coding assistant, personal assistant) — python/examples/apps/
- [x] Flesh out CONTRIBUTING, AGENTS_GUIDE.md (Makefile targets, project structure, integration patterns)

---

## VI. Business & Productization

**Goal:** Open-core model, hosted offering, team UX.

**Increments:**
- [x] Define OSS vs commercial boundaries (docs/OSS_COMMERCIAL.md)
- [x] Roadmap for commercial features (docs/COMMERCIAL_ROADMAP.md)
- [x] Simple web dashboard (usage, health, config) — GET /dashboard
- [x] Onboarding materials, comparison with alternatives (docs/ONBOARDING.md, docs/COMPARISON.md)

---

## Execution Log

| Date | Slice | Outcome |
|------|-------|---------|
| 2026-02 | Master roadmap | Created design_roadmap.md |
| 2026-02 | Pluggable index (design) | Created design_pluggable_index.md |
| 2026-02 | Exact backend | Added IndexBackend::Exact, AgentMemDB::new_exact |
| 2026-02 | On-disk design | Created design_on_disk_storage.md |
| 2026-02 | AgentMemDBDisk | Append-only JSONL log, open/store/query, tests |
| 2026-02 | Rich metadata (design) | Created design_rich_metadata.md |
| 2026-02 | Rich metadata (slice 1) | Episode timestamp/tags, QueryOptions, query filters |
| 2026-02 | TypeScript/Node bindings | napi-rs, node/, AgentMemDb, createEpisode, example |
| 2026-02 | LangChain integration | AgentMemDBVectorStore, integrations/langchain/, example |
| 2026-02 | Config-driven eval | eval_config.json/yaml, --config, VARIANT_REGISTRY, load_config() |
| 2026-02 | LangGraph memory adapter | AgentMemDBStore, integrations/langgraph/, BaseStore put/get/search |
| 2026-02 | Per-language README, tests | Node test.js, LangGraph test_store.py, main README bindings table |
| 2026-02 | Docs per framework | docs/integrations/langchain.md, langgraph.md, README |
| 2026-02 | Extended metrics | p50/p95/p99 latency in run_eval, eval_results.json |
| 2026-02 | Reorganize docs | docs/README.md index (Getting Started, Integrations, Operations, Evaluation) |
| 2026-02 | Proptest save/load | tests/proptest_persist.rs, load_from_file_exact for deterministic roundtrip |
| 2026-02 | Flesh out CONTRIBUTING, AGENTS_GUIDE | Makefile table, project structure, integration patterns |
| 2026-02 | Architectural diagrams | docs/architecture.md (component overview, data flow, bindings) |
| 2026-02 | Case study docs | docs/case_study_preference_qa.md (when AgentMemDB wins) |
| 2026-02 | Design Hosted Memory Cloud | docs/design_hosted_memory.md (HTTP/gRPC API, multi-tenant) |
| 2026-02 | Hosted Memory Cloud Phase 1 | server/ — axum HTTP API, per-tenant AgentMemDB, API key auth |
| 2026-02 | Concurrent read/write stress tests | tests/concurrent_stress.rs (exact + HNSW, Arc&lt;RwLock&lt;&gt;&gt;) |
| 2026-02 | Fuzzing of episode inputs and metadata | tests/fuzz_episode.rs (proptest, metadata/tags/steps) |
| 2026-02 | Analysis scripts and notebooks | python/scripts/analyze_eval.py, notebooks/eval_analysis.ipynb |
| 2026-02 | Full application examples | python/examples/apps/ (coding_assistant, personal_assistant) |
| 2026-02 | Docker for server deployment | Dockerfile, .dockerignore, make docker-build/docker-run |
| 2026-02 | Metrics, logging | GET /metrics (Prometheus), TraceLayer request logging |
| 2026-02 | Rate limiting | Per-tenant fixed-window, AGENT_MEM_RATE_LIMIT env |
| 2026-02 | OSS vs commercial boundaries | docs/OSS_COMMERCIAL.md |
| 2026-02 | Onboarding, comparison | docs/ONBOARDING.md, docs/COMPARISON.md |
| 2026-02 | Commercial roadmap | docs/COMMERCIAL_ROADMAP.md |
| 2026-02 | Memory benchmarks | docs/MEMORY_BENCHMARKS.md (LongMemEval, LoCoMo, MemGPT) |
| 2026-02 | Auditing | AGENT_MEM_AUDIT_LOG, JSONL audit for store/query/save/load |
| 2026-02 | Helm, Terraform | deploy/helm/agent-mem-server, deploy/terraform, deploy/README.md |
| 2026-02 | Go bindings | capi/ (C-ABI), go/ (cgo), New/Store/Query/Save/Load, make go-build |
| 2026-02 | Simple web dashboard | GET /dashboard — usage, health, config |
| 2026-02 | source, user_id | Episode optional fields, QueryOptions filters, Python/Node/server/Go |
| 2026-02 | tags_all, task_id_prefix | QueryOptions filters (tags_all, task_id_prefix) |
| 2026-02 | Combine similarity + constraints | Recency tie-breaker (design_combine_similarity_constraints.md) |
| 2026-02 | Async APIs (first slice) | Python AgentMemDBAsync (asyncio.to_thread), Rust async_api (spawn_blocking) |
| 2026-02 | Retention and pruning (first slice) | prune_older_than for AgentMemDB and AgentMemDBDisk, log compaction |
| 2026-02 | Retention and pruning (count-based) | prune_keep_newest(n) for AgentMemDB, AgentMemDBDisk, Python |
| 2026-02 | Prune in Node and Go | pruneOlderThan, pruneKeepNewest in Node; PruneOlderThan, PruneKeepNewest in Go |
| 2026-02 | Prune HTTP endpoints | POST /v1/prune/older-than, POST /v1/prune/keep-newest |
| 2026-02 | Importance-based pruning | prune_keep_highest_reward(n) — Rust, Python, Node, Go, HTTP |
| 2026-02 | Retention in app examples | prune, stats commands in coding_assistant, personal_assistant |
| 2026-02 | Design incremental indexing | design_incremental_indexing.md — checkpoint for fast restart |
| 2026-02 | ExactIndex checkpoint (first slice) | DiskOptions::exact_with_checkpoint, checkpoint(), Python open_exact_with_checkpoint |
| 2026-02 | Hosted Memory Phase 3: disk persistence | AGENT_MEM_DATA_DIR → AgentMemDBDisk per tenant; POST /v1/checkpoint |
| 2026-02 | Helm persistence → AGENT_MEM_DATA_DIR | When persistence.enabled, auto-set AGENT_MEM_DATA_DIR to mount path |
| 2026-02 | AgentMemDBDisk Node bindings | AgentMemDbDisk.open, openExactWithCheckpoint, storeEpisode, querySimilar, checkpoint, prune* |
| 2026-02 | AgentMemDBDisk Go bindings | C API + DiskDB: OpenDisk, OpenDiskExactWithCheckpoint, Store, Query, Checkpoint, Prune* |
| 2026-02 | Design PQ/IVF backends | design_pq_ivf.md — future index options for scale |
| 2026-02 | Python 3.14 + test-all | PYO3_USE_ABI3_FORWARD_COMPATIBILITY in make python-dev; make test-all |
| 2026-02 | Design HNSW checkpoint | design_hnsw_checkpoint.md — options when hnswx adds serialization |
| 2026-02 | Design horizontal scaling | design_horizontal_scaling.md — Phase 4 options (shared storage, sharding) |
| 2026-02 | Phase 4 Option A: multi-replica + NFS | Helm persistence.accessMode=ReadWriteMany; deploy/README multi-replica section |
