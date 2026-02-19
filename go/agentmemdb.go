// Package agentmemdb provides Go bindings for AgentMemDB via C-API.
package agentmemdb

/*
#cgo CFLAGS: -I${SRCDIR}/../capi/include
#cgo darwin LDFLAGS: -L${SRCDIR}/../target/release -lagent_mem_db_capi
#cgo linux LDFLAGS: -L${SRCDIR}/../target/release -lagent_mem_db_capi -lpthread -ldl
#cgo windows LDFLAGS: -L${SRCDIR}/../target/release -lagent_mem_db_capi -luserenv -lbcrypt

#include "agent_mem_db.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"
)

// DB is an in-memory agent memory database with HNSW vector search.
type DB struct {
	handle C.AgentMemDBHandle
	dim    int
}

// Episode represents a stored agent experience (returned from Query).
type Episode struct {
	ID             string    `json:"id"`
	TaskID         string    `json:"task_id"`
	StateEmbedding []float32 `json:"state_embedding"`
	Reward         float32   `json:"reward"`
	Metadata       any       `json:"metadata,omitempty"`
	Timestamp      *int64    `json:"timestamp,omitempty"`
	Tags           []string  `json:"tags,omitempty"`
	Source         *string   `json:"source,omitempty"`
	UserID         *string   `json:"user_id,omitempty"`
}

// New creates a new AgentMemDB for the given embedding dimension.
func New(dim int) *DB {
	h := C.agent_mem_db_new(C.size_t(dim))
	if h == nil {
		return nil
	}
	return &DB{handle: h, dim: dim}
}

// Dim returns the embedding dimension.
func (db *DB) Dim() int {
	return db.dim
}

// Free releases the database. Call when done.
func (db *DB) Free() {
	if db != nil && db.handle != nil {
		C.agent_mem_db_free(db.handle)
		db.handle = nil
	}
}

// Store adds an episode. taskID and embedding are required.
func (db *DB) Store(taskID string, embedding []float32, reward float32) error {
	if db == nil || db.handle == nil {
		return fmt.Errorf("db is nil or freed")
	}
	if len(embedding) != db.dim {
		return fmt.Errorf("embedding dimension mismatch: expected %d, got %d", db.dim, len(embedding))
	}
	ctask := C.CString(taskID)
	defer C.free(unsafe.Pointer(ctask))
	emb := (*C.float)(unsafe.Pointer(&embedding[0]))
	r := C.agent_mem_db_store(db.handle, ctask, emb, C.size_t(len(embedding)), C.float(reward))
	if r != 0 {
		return fmt.Errorf("store failed: %s", lastError())
	}
	return nil
}

// Query returns similar episodes. Caller frees the DB when done.
func (db *DB) Query(embedding []float32, minReward float32, topK int) ([]Episode, error) {
	if db == nil || db.handle == nil {
		return nil, fmt.Errorf("db is nil or freed")
	}
	if len(embedding) != db.dim {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d", db.dim, len(embedding))
	}
	emb := (*C.float)(unsafe.Pointer(&embedding[0]))
	jsonStr := C.agent_mem_db_query(db.handle, emb, C.size_t(len(embedding)), C.float(minReward), C.size_t(topK))
	if jsonStr == nil {
		return nil, fmt.Errorf("query failed: %s", lastError())
	}
	defer C.agent_mem_db_free_string(jsonStr)
	s := C.GoString(jsonStr)
	var episodes []Episode
	if err := json.Unmarshal([]byte(s), &episodes); err != nil {
		return nil, fmt.Errorf("parse query result: %w", err)
	}
	return episodes, nil
}

// Save persists the database to a JSON file.
func (db *DB) Save(path string) error {
	if db == nil || db.handle == nil {
		return fmt.Errorf("db is nil or freed")
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	r := C.agent_mem_db_save(db.handle, cpath)
	if r != 0 {
		return fmt.Errorf("save failed: %s", lastError())
	}
	return nil
}

// Load creates a new DB from a JSON file. Caller must call Free when done.
func Load(path string) (*DB, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	h := C.agent_mem_db_load(cpath)
	if h == nil {
		return nil, fmt.Errorf("load failed: %s", lastError())
	}
	dim := int(C.agent_mem_db_dim(h))
	return &DB{handle: h, dim: dim}, nil
}

// PruneOlderThan removes episodes with timestamp older than cutoff (Unix ms). Episodes without timestamp are kept. Returns number removed.
func (db *DB) PruneOlderThan(timestampCutoffMs int64) int {
	if db == nil || db.handle == nil {
		return 0
	}
	return int(C.agent_mem_db_prune_older_than(db.handle, C.int64_t(timestampCutoffMs)))
}

// PruneKeepNewest keeps only the n most recent episodes (by timestamp). Returns number removed.
func (db *DB) PruneKeepNewest(n int) int {
	if db == nil || db.handle == nil {
		return 0
	}
	return int(C.agent_mem_db_prune_keep_newest(db.handle, C.size_t(n)))
}

// PruneKeepHighestReward keeps only the n episodes with highest reward. Returns number removed.
func (db *DB) PruneKeepHighestReward(n int) int {
	if db == nil || db.handle == nil {
		return 0
	}
	return int(C.agent_mem_db_prune_keep_highest_reward(db.handle, C.size_t(n)))
}

func lastError() string {
	p := C.agent_mem_db_last_error()
	if p == nil {
		return "unknown error"
	}
	return C.GoString(p)
}

// DiskDB is a disk-backed agent memory database. Episodes stored in append-only log.
type DiskDB struct {
	handle C.AgentMemDBDiskHandle
	dim    int
}

// OpenDisk opens or creates a disk-backed DB at the given directory. Uses HNSW by default.
func OpenDisk(path string, dim int) (*DiskDB, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	h := C.agent_mem_db_disk_open(cpath, C.size_t(dim))
	if h == nil {
		return nil, fmt.Errorf("open disk failed: %s", lastError())
	}
	return &DiskDB{handle: h, dim: dim}, nil
}

// OpenDiskExactWithCheckpoint opens with exact index and checkpoint for fast restart.
func OpenDiskExactWithCheckpoint(path string, dim int) (*DiskDB, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	h := C.agent_mem_db_disk_open_exact_with_checkpoint(cpath, C.size_t(dim))
	if h == nil {
		return nil, fmt.Errorf("open disk failed: %s", lastError())
	}
	return &DiskDB{handle: h, dim: dim}, nil
}

// Free releases the disk DB. Call when done.
func (db *DiskDB) Free() {
	if db != nil && db.handle != nil {
		C.agent_mem_db_disk_free(db.handle)
		db.handle = nil
	}
}

// Dim returns the embedding dimension.
func (db *DiskDB) Dim() int {
	return db.dim
}

// Store adds an episode.
func (db *DiskDB) Store(taskID string, embedding []float32, reward float32) error {
	if db == nil || db.handle == nil {
		return fmt.Errorf("db is nil or freed")
	}
	if len(embedding) != db.dim {
		return fmt.Errorf("embedding dimension mismatch: expected %d, got %d", db.dim, len(embedding))
	}
	ctask := C.CString(taskID)
	defer C.free(unsafe.Pointer(ctask))
	emb := (*C.float)(unsafe.Pointer(&embedding[0]))
	r := C.agent_mem_db_disk_store(db.handle, ctask, emb, C.size_t(len(embedding)), C.float(reward))
	if r != 0 {
		return fmt.Errorf("store failed: %s", lastError())
	}
	return nil
}

// Query returns similar episodes.
func (db *DiskDB) Query(embedding []float32, minReward float32, topK int) ([]Episode, error) {
	if db == nil || db.handle == nil {
		return nil, fmt.Errorf("db is nil or freed")
	}
	if len(embedding) != db.dim {
		return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d", db.dim, len(embedding))
	}
	emb := (*C.float)(unsafe.Pointer(&embedding[0]))
	jsonStr := C.agent_mem_db_disk_query(db.handle, emb, C.size_t(len(embedding)), C.float(minReward), C.size_t(topK))
	if jsonStr == nil {
		return nil, fmt.Errorf("query failed: %s", lastError())
	}
	defer C.agent_mem_db_free_string(jsonStr)
	s := C.GoString(jsonStr)
	var episodes []Episode
	if err := json.Unmarshal([]byte(s), &episodes); err != nil {
		return nil, fmt.Errorf("parse query result: %w", err)
	}
	return episodes, nil
}

// Checkpoint persists the ExactIndex checkpoint for fast restart. No-op for HNSW.
func (db *DiskDB) Checkpoint() error {
	if db == nil || db.handle == nil {
		return fmt.Errorf("db is nil or freed")
	}
	r := C.agent_mem_db_disk_checkpoint(db.handle)
	if r != 0 {
		return fmt.Errorf("checkpoint failed: %s", lastError())
	}
	return nil
}

// PruneOlderThan removes episodes with timestamp older than cutoff (Unix ms). Returns number removed, or error.
func (db *DiskDB) PruneOlderThan(timestampCutoffMs int64) (int, error) {
	if db == nil || db.handle == nil {
		return 0, fmt.Errorf("db is nil or freed")
	}
	r := C.agent_mem_db_disk_prune_older_than(db.handle, C.int64_t(timestampCutoffMs))
	if r < 0 {
		return 0, fmt.Errorf("prune failed: %s", lastError())
	}
	return int(r), nil
}

// PruneKeepNewest keeps only the n most recent episodes. Returns number removed, or error.
func (db *DiskDB) PruneKeepNewest(n int) (int, error) {
	if db == nil || db.handle == nil {
		return 0, fmt.Errorf("db is nil or freed")
	}
	r := C.agent_mem_db_disk_prune_keep_newest(db.handle, C.size_t(n))
	if r < 0 {
		return 0, fmt.Errorf("prune failed: %s", lastError())
	}
	return int(r), nil
}

// PruneKeepHighestReward keeps only the n episodes with highest reward. Returns number removed, or error.
func (db *DiskDB) PruneKeepHighestReward(n int) (int, error) {
	if db == nil || db.handle == nil {
		return 0, fmt.Errorf("db is nil or freed")
	}
	r := C.agent_mem_db_disk_prune_keep_highest_reward(db.handle, C.size_t(n))
	if r < 0 {
		return 0, fmt.Errorf("prune failed: %s", lastError())
	}
	return int(r), nil
}
