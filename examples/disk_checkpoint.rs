//! Example: AgentMemDBDisk with ExactIndex checkpoint for fast restart.
//!
//! Use `DiskOptions::exact_with_checkpoint(dim)` when you need:
//! - Durable storage (append-only log)
//! - Fast restarts (skip full log replay via checkpoint)
//!
//! Run: cargo run --example disk_checkpoint

use agent_mem_db::{AgentMemDBDisk, DiskOptions, Episode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("agent_mem_db_checkpoint_example");
    std::fs::create_dir_all(&dir)?;
    let dim = 8;

    // First run: create DB, store episodes, checkpoint
    {
        let mut db = AgentMemDBDisk::open_with_options(&dir, DiskOptions::exact_with_checkpoint(dim))?;
        db.store_episode(Episode::new("task_1", vec![0.1; dim], 0.8))?;
        db.store_episode(Episode::new("task_2", vec![0.2; dim], 0.9))?;
        db.checkpoint()?;
        println!("Stored 2 episodes and checkpointed.");
    }

    // Second run: open loads from checkpoint (no full replay)
    let db = AgentMemDBDisk::open_with_options(&dir, DiskOptions::exact_with_checkpoint(dim))?;
    let results = db.query_similar(&vec![0.15; dim], 0.5, 5)?;
    println!("Query returned {} episodes.", results.len());
    for ep in &results {
        println!("  - {} (reward: {})", ep.task_id, ep.reward);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
