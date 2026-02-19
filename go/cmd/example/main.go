// Example usage of AgentMemDB Go bindings.
package main

import (
	"fmt"
	"log"

	"github.com/agent-mem-db/agent_mem_db"
)

func main() {
	db := agentmemdb.New(8)
	if db == nil {
		log.Fatal("New failed")
	}
	defer db.Free()

	emb := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	if err := db.Store("task1", emb, 0.9); err != nil {
		log.Fatal(err)
	}
	if err := db.Store("task2", []float32{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, 0.8); err != nil {
		log.Fatal(err)
	}

	results, err := db.Query(emb, 0.0, 5)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Query returned %d episodes:\n", len(results))
	for _, ep := range results {
		fmt.Printf("  %s (id=%s) reward=%.1f\n", ep.TaskID, ep.ID, ep.Reward)
	}

	path := "/tmp/agent_mem_go_example.json"
	if err := db.Save(path); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Saved to %s\n", path)

	db2, err := agentmemdb.Load(path)
	if err != nil {
		log.Fatal(err)
	}
	defer db2.Free()
	fmt.Printf("Loaded DB with dim=%d\n", db2.Dim())
}
