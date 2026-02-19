# Roadmap for Commercial Features

**Status:** Draft — potential commercial product direction. No commercial offering exists today.

See [OSS_COMMERCIAL.md](OSS_COMMERCIAL.md) for OSS vs commercial boundaries.

## Phase 1: Hosted Memory Cloud (Managed)

- **Managed API**: Hosted HTTP/gRPC endpoint; no self-hosting
- **SLA**: Uptime, latency guarantees
- **Auto-scaling**: Per-tenant resource allocation
- **Tiers**: Free tier (limited), paid tiers (higher limits, support)

## Phase 2: Enterprise Features

- **Audit logging**: Immutable log of store/query/load operations
- **SSO / SAML**: Enterprise identity integration
- **Compliance**: SOC2, GDPR-ready data handling
- **Cross-tenant analytics**: Aggregated usage dashboards (opt-in)

## Phase 3: Premium Capabilities

- **Premium embeddings**: Managed embedding pipelines (e.g. OpenAI, Cohere)
- **Advanced retention**: Time-based, importance-based pruning
- **Replication**: Multi-region, disaster recovery

## Dependencies

- Phase 1 builds on the OSS server (design_hosted_memory.md)
- Phase 2 requires auditing, auth enhancements
- Phase 3 requires distributed storage design

## Non-Goals (Stay OSS)

- Core AgentMemDB library
- Python/Node bindings
- LangChain/LangGraph integrations
- Evaluation suite
- Self-hosted server
