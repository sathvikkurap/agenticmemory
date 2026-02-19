# Contributing

Thanks for considering contributing to `agent_mem_db`.

## Developer Quickstart

From repo root, use the Makefile or run commands directly:

| Target | Command |
|--------|---------|
| All tests | `make test-all` (Rust + Node + Go + Python) |
| Rust tests | `make test` or `cargo test` |
| Rust fmt/clippy | `make fmt` `make clippy` |
| Python dev | `make python-dev` (builds PyO3 extension; Python 3.14 compat) |
| Python tests | `make python-test` |
| Node tests | `make node-test` |
| LangGraph tests | `make langgraph-test` |
| Eval | `make eval` |
| Benchmarks | `make bench` |

## Full Setup

**Rust:**
```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test
cargo run --example agent_sim
```

**Python** (requires Rust toolchain):
```bash
cd python
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip maturin pytest
maturin develop --release    # Python 3.13+: PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
pytest tests/
python -m examples.basic
```

**Node.js:**
```bash
cd node && npm install && npm run build && npm test
```

**Integrations:**
```bash
make langchain-example
make langgraph-example
make langgraph-agent
```

## Before Submitting a PR

1. Run `cargo fmt` and `cargo clippy --all-targets --all-features`
2. Run `make test-all` (or `cargo test` + `make python-test` + `make node-test` + `make go-test` as needed)
3. Do not include build artifacts (check `.gitignore`)

## Documentation

- [docs/README.md](docs/README.md) — Full doc index
- [docs/design_roadmap.md](docs/design_roadmap.md) — Roadmap and execution log
- [AGENTS_GUIDE.md](AGENTS_GUIDE.md) — Conventions and project structure

## Reporting Issues

Open a GitHub issue with a short title, steps to reproduce, and expected vs actual behaviour. Attach logs if relevant.
