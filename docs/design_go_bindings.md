# Design: Go Bindings

**Status:** First slice done  
**Related:** [design_roadmap.md](design_roadmap.md) § II.2.1  
**Implementation:** `capi/`, `go/` — `make go-build`, `make go-test`, `make go-example`

## Goal

Provide idiomatic Go bindings for AgentMemDB via a C-ABI wrapper. Enables Go agent frameworks to use episodic memory without Python or Node.

## Approach

- **C-ABI:** Rust crate compiles to `cdylib`; exposes `#[no_mangle] extern "C"` functions.
- **cgo:** Go package uses cgo to call the C API.
- **Layout:** `capi/` (Rust C-API crate), `go/` (Go module).

## C API (First Slice)

```c
// Opaque handle
typedef void* AgentMemDBHandle;

// Create/destroy
AgentMemDBHandle agent_mem_db_new(size_t dim);
void agent_mem_db_free(AgentMemDBHandle h);

// Store (task_id, embedding, reward)
int agent_mem_db_store(AgentMemDBHandle h, const char* task_id,
                       const float* embedding, size_t dim, float reward);

// Query — returns JSON array of episodes (caller frees via agent_mem_db_free_string)
char* agent_mem_db_query(AgentMemDBHandle h, const float* embedding,
                         float min_reward, size_t top_k);

// Persistence
int agent_mem_db_save(AgentMemDBHandle h, const char* path);
AgentMemDBHandle agent_mem_db_load(const char* path);  // new handle, or NULL

// Error reporting
char* agent_mem_db_last_error(void);
void agent_mem_db_free_string(char* s);
```

## Go API (First Slice)

```go
db := agentmemdb.New(768)
defer db.Free()

db.Store("task1", []float32{0.1, ...}, 0.9)
episodes, err := db.Query([]float32{0.1, ...}, 0.0, 5)
db.Save("/tmp/mem.json")
```

## Build

- `cargo build -p agent_mem_db_capi --release` — produces `libagent_mem_db_capi.so` (Linux), `.dylib` (Mac), `.dll` (Windows)
- `cd go && go build` — links via cgo (CGO_LDFLAGS or build script)
- Makefile: `make go-build`, `make go-test`

## AgentMemDBDisk (Implemented)

- `OpenDisk(path, dim)` — disk-backed DB (HNSW)
- `OpenDiskExactWithCheckpoint(path, dim)` — disk-backed with checkpoint
- `Store`, `Query`, `Checkpoint`, `PruneOlderThan`, `PruneKeepNewest`, `PruneKeepHighestReward`

## Future

- Query options (tags_any, time_after, time_before)
- Episode struct with metadata
