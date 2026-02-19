# Agent Memory DB — Deployment

Deploy the Hosted Memory Cloud HTTP server to Kubernetes.

## Helm

### Prerequisites

- Kubernetes cluster (minikube, kind, EKS, GKE, etc.)
- `kubectl` configured
- Docker image `agent-mem-server` built and available to the cluster

### Install

```bash
# Build the image
cd agent_mem_db && make docker-build

# Load into kind (if using kind)
kind load docker-image agent-mem-server:latest

# Create API key secret (required for production)
kubectl create secret generic agent-mem-api-key --from-literal=api-key=your-secret

# Install with Helm
helm install agent-mem ./deploy/helm/agent-mem-server \
  --set image.repository=agent-mem-server \
  --set image.tag=latest \
  --set apiKey.secretName=agent-mem-api-key \
  --set apiKey.secretKey=api-key
```

### Dev (no API key)

```bash
helm install agent-mem ./deploy/helm/agent-mem-server \
  --set image.repository=agent-mem-server \
  --set image.tag=latest
```

### Port-forward

```bash
# Service name: <release>-agent-mem-server (e.g. agent-mem-agent-mem-server)
kubectl port-forward svc/agent-mem-agent-mem-server 8080:8080
# Then: curl http://localhost:8080/health
```

### Multi-replica with shared storage

For horizontal scaling, run multiple replicas with a shared data directory. Use a `ReadWriteMany` PVC (e.g. NFS-backed):

```bash
helm install agent-mem ./deploy/helm/agent-mem-server \
  --set replicaCount=2 \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set persistence.storageClass=nfs-client \
  --set persistence.accessMode=ReadWriteMany \
  --set apiKey.secretName=agent-mem-api-key
```

Ensure your cluster has a StorageClass that supports `ReadWriteMany`. Each pod shares the same mount; any pod can serve any tenant. See [design_horizontal_scaling.md](../docs/design_horizontal_scaling.md).

### Customize

See `deploy/helm/agent-mem-server/values.yaml` for:

- `replicaCount`, `image`, `service`
- `env` — AGENT_MEM_DIM, AGENT_MEM_DATA_DIR, AGENT_MEM_RATE_LIMIT, etc.
- `persistence` — PVC for data and audit log. When enabled, sets `AGENT_MEM_DATA_DIR` to the mount path for disk-backed storage.
- `ingress` — expose via Ingress controller

## Terraform

Deploy via Terraform using the Helm chart.

### Prerequisites

- Terraform >= 1.0
- `kubectl` configured (KUBECONFIG or ~/.kube/config)

### Apply

```bash
cd deploy/terraform
terraform init
terraform plan -var="api_key_secret=your-secret"
terraform apply -var="api_key_secret=your-secret"
```

### Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `kube_config_path` | (auto) | Path to kubeconfig |
| `namespace` | default | Kubernetes namespace |
| `release_name` | agent-mem-server | Helm release name |
| `image_repository` | agent-mem-server | Docker image |
| `image_tag` | latest | Image tag |
| `api_key_secret` | "" | API key (creates K8s secret) |
| `replica_count` | 1 | Number of replicas |
