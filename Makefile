.PHONY: test test-all bench fmt clippy python-dev python-test eval eval-analyze build-node node-test langchain-example langgraph-example langgraph-agent langgraph-test server server-run docker-build docker-run helm-install helm-uninstall terraform-init terraform-apply go-build go-test go-example coding-assistant personal-assistant disk-checkpoint

test:
	cargo test

# Run all tests (Rust, Node, Go, Python). Builds Python bindings first.
test-all: test python-dev
	$(MAKE) node-test
	$(MAKE) go-test
	$(MAKE) python-test

bench:
	cargo bench --bench agent_mem_db_bench -- --nocapture

fmt:
	cargo fmt

clippy:
	cargo clippy --all-targets --all-features

python-dev:
	cd python && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release

python-test:
	cd python && .venv/bin/python -m pytest tests/

eval:
	cd python && .venv/bin/python -m examples.agents.run_eval

eval-analyze:
	cd python && python3 scripts/analyze_eval.py --input ../eval_results.json --output ../docs/eval_analysis_report.md

build-node:
	cd node && npm install && npm run build

node-test:
	cd node && npm run build && npm test

langchain-example:
	cd python && .venv/bin/pip install -q -e ../integrations/langchain langchain-core && .venv/bin/python ../integrations/langchain/example.py

langgraph-example:
	cd python && .venv/bin/pip install -q -e ../integrations/langgraph && .venv/bin/python ../integrations/langgraph/example.py

langgraph-agent:
	cd python && .venv/bin/pip install -q -e ../integrations/langgraph && .venv/bin/python ../integrations/langgraph/agent_example.py

coding-assistant:
	cd python && .venv/bin/python -m examples.apps.coding_assistant

personal-assistant:
	cd python && .venv/bin/python -m examples.apps.personal_assistant

langgraph-test:
	cd python && .venv/bin/pip install -q -e ../integrations/langgraph pytest && .venv/bin/pytest ../integrations/langgraph/tests/ -v

server:
	cargo build -p agent_mem_db_server --release

server-run:
	AGENT_MEM_API_KEY=dev-secret AGENT_MEM_DIM=16 cargo run -p agent_mem_db_server

docker-build:
	docker build -t agent-mem-server .

docker-run:
	docker run -p 8080:8080 -e AGENT_MEM_API_KEY=dev-secret -e AGENT_MEM_DIM=16 agent-mem-server

# Deployment (Helm, Terraform)
helm-install:
	helm install agent-mem ./deploy/helm/agent-mem-server --set image.repository=agent-mem-server --set image.tag=latest
helm-uninstall:
	helm uninstall agent-mem
terraform-init:
	cd deploy/terraform && terraform init
terraform-apply:
	cd deploy/terraform && terraform apply -var="api_key_secret=$${AGENT_MEM_API_KEY:-dev-secret}" -auto-approve

# Go bindings (requires C library built first)
go-build:
	cargo build -p agent_mem_db_capi --release && cd go && go build ./...
go-test:
	cargo build -p agent_mem_db_capi --release && cd go && go test -v ./...
go-example:
	cargo build -p agent_mem_db_capi --release && cd go && go run ./cmd/example

disk-checkpoint:
	cargo run --example disk_checkpoint
