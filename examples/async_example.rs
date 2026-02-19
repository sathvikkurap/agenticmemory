//! Example: async API for AgentMemDB
//!
//! Run with: cargo run --example async_example --features async

use agent_mem_db::async_api::{
    load_from_file_async, query_similar_async, save_to_file_async, store_episode_async,
};
use agent_mem_db::{AgentMemDB, Episode, QueryOptions};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Arc::new(RwLock::new(AgentMemDB::new(8)));
    let ep = Episode::new("task_1", vec![0.1; 8], 1.0);

    store_episode_async(db.clone(), ep).await?;
    let results = query_similar_async(db.clone(), vec![0.1; 8], QueryOptions::new(0.0, 5)).await?;
    assert_eq!(results.len(), 1);
    println!("Query returned {} episode(s)", results.len());

    let path = PathBuf::from("/tmp/agent_mem_async_example.json");
    save_to_file_async(db.clone(), path.clone()).await?;
    let loaded = load_from_file_async(path).await?;
    let guard = loaded.read().unwrap();
    println!("Loaded DB has dim={}", guard.dim());
    Ok(())
}
