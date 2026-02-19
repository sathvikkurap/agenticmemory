package agentmemdb_test

import (
	"fmt"
	"os"

	"github.com/agent-mem-db/agent_mem_db"
)

func ExampleDB() {
	db := agentmemdb.New(8)
	if db == nil {
		panic("New failed")
	}
	defer db.Free()

	emb := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	if err := db.Store("task1", emb, 0.9); err != nil {
		panic(err)
	}

	results, err := db.Query(emb, 0.0, 5)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Found %d episodes\n", len(results))
	for _, ep := range results {
		fmt.Printf("  %s: reward=%.1f\n", ep.TaskID, ep.Reward)
	}
	// Output:
	// Found 1 episodes
	//   task1: reward=0.9
}

func ExampleDB_PruneKeepNewest() {
	db := agentmemdb.New(8)
	if db == nil {
		panic("New failed")
	}
	defer db.Free()

	emb := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	db.Store("a", emb, 0.9)
	db.Store("b", emb, 0.8)
	db.Store("c", emb, 0.7)

	removed := db.PruneKeepNewest(2)
	results, _ := db.Query(emb, 0.0, 5)
	fmt.Printf("Pruned %d, kept %d\n", removed, len(results))
	// Output:
	// Pruned 1, kept 2
}

func ExampleDiskDB() {
	dir, _ := os.MkdirTemp("", "agent_mem_go_disk_*")
	defer os.RemoveAll(dir)

	db, err := agentmemdb.OpenDiskExactWithCheckpoint(dir, 8)
	if err != nil {
		panic(err)
	}
	defer db.Free()

	emb := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	db.Store("task1", emb, 0.9)
	db.Store("task2", emb, 0.8)
	db.Checkpoint()

	db2, _ := agentmemdb.OpenDiskExactWithCheckpoint(dir, 8)
	defer db2.Free()
	results, _ := db2.Query(emb, 0.5, 5)
	fmt.Printf("Found %d episodes\n", len(results))
	// Output:
	// Found 2 episodes
}
