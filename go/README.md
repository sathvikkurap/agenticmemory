# AgentMemDB — Go Bindings

Go bindings for AgentMemDB via C-API (cgo).

## Prerequisites

- Go 1.21+
- Rust toolchain (for building the C library)
- C compiler (gcc/clang)

## Build

```bash
# From repo root: build the C library first
cargo build -p agent_mem_db_capi --release

# Then build the Go package
cd go && go build ./...
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/agent-mem-db/agent_mem_db"
)

func main() {
    db := agentmemdb.New(768)
    defer db.Free()

    emb := []float32{0.1, 0.2, ...}  // 768-dim
    db.Store("task1", emb, 0.9)

    results, _ := db.Query(emb, 0.0, 5)
    for _, ep := range results {
        fmt.Println(ep.TaskID, ep.Reward)
    }

    db.Save("/tmp/mem.json")
    db2, _ := agentmemdb.Load("/tmp/mem.json")
    defer db2.Free()
}
```

## API

### In-memory (DB)

- `New(dim int) *DB` — create in-memory DB
- `Load(path string) (*DB, error)` — load from JSON file
- `db.Free()` — release resources
- `db.Store(taskID string, embedding []float32, reward float32) error`
- `db.Query(embedding []float32, minReward float32, topK int) ([]Episode, error)`
- `db.Save(path string) error`
- `db.PruneOlderThan`, `db.PruneKeepNewest`, `db.PruneKeepHighestReward`
- `db.Dim() int` — embedding dimension

### Disk-backed (DiskDB)

- `OpenDisk(path string, dim int) (*DiskDB, error)` — disk-backed (HNSW)
- `OpenDiskExactWithCheckpoint(path string, dim int) (*DiskDB, error)` — disk-backed with checkpoint
- `db.Store`, `db.Query`, `db.Checkpoint()`, `db.PruneOlderThan`, `db.PruneKeepNewest`, `db.PruneKeepHighestReward`

## Run Example

```bash
cargo build -p agent_mem_db_capi --release
cd go && go run ./cmd/example
```

## Test

```bash
cd go && go test -v
```
