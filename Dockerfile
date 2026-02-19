# Agent Memory DB — HTTP server (Hosted Memory Cloud Phase 1)
# Build: docker build -t agent-mem-server .
# Run:   docker run -p 8080:8080 -e AGENT_MEM_API_KEY=secret agent-mem-server

FROM rust:1.75-bookworm AS builder
WORKDIR /app

# Copy workspace (all member crates required for Cargo to resolve)
COPY . .

# Build release binary (only server + agent_mem_db lib are built)
RUN cargo build -p agent_mem_db_server --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/agent-mem-server /usr/local/bin/

EXPOSE 8080
ENV RUST_LOG=info

CMD ["agent-mem-server"]
