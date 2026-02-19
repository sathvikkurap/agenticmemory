# OSS vs Commercial Boundaries

**Status:** Draft — defines intended boundaries for Agent Memory DB.

## Open Source (MIT)

The following are and will remain open source under the MIT license:

- **Core library** (`agent_mem_db` crate): AgentMemDB, AgentMemDBDisk, Episode, QueryOptions, HNSW/exact index backends
- **Python bindings** (`python/`): PyO3 bindings, AgentMemDB, AgentMemDBDisk
- **Node.js bindings** (`node/`): napi-rs bindings
- **Integrations**: LangChain VectorStore, LangGraph AgentMemDBStore
- **HTTP server** (`server/`): axum API, auth, rate limiting, metrics
- **Evaluation suite**: run_eval, config-driven eval, analysis scripts
- **Examples**: coding assistant, personal assistant, agent examples
- **Documentation**: design docs, architecture, integration guides

## Commercial (Future)

Potential commercial offerings, if pursued:

- **Hosted Memory Cloud**: Managed, multi-tenant service with SLA, auto-scaling, enterprise support
- **Advanced features**: Cross-tenant analytics, compliance/audit dashboards, premium embeddings
- **Support**: Paid support, training, custom integrations

## Principles

1. **Core stays free**: The library and server you can self-host are always OSS.
2. **No artificial limits**: OSS version has no usage caps or feature gates.
3. **Transparent**: Commercial features, if any, are clearly documented and separate.
4. **Community first**: Contributions, issues, and docs focus on the OSS experience.

## Current State

As of 2026-02, Agent Memory DB is fully open source. No commercial offering exists. This document serves as a placeholder for future productization decisions.
