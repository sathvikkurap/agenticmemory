//! Async API for AgentMemDB — non-blocking wrappers using `tokio::task::spawn_blocking`.
//!
//! Enable with the `async` feature: `agent_mem_db = { version = "0.1", features = ["async"] }`
//!
//! The caller must wrap the DB in `Arc<RwLock<AgentMemDB>>` so it can be shared across async tasks.

use crate::{AgentMemDB, AgentMemError, Episode, QueryOptions};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;

/// Store an episode without blocking the async runtime.
pub async fn store_episode_async(
    db: Arc<RwLock<AgentMemDB>>,
    ep: Episode,
) -> Result<(), AgentMemError> {
    tokio::task::spawn_blocking(move || {
        let mut guard = db.write().unwrap();
        guard.store_episode(ep)
    })
    .await
    .map_err(|e| AgentMemError::HnswError(format!("spawn_blocking: {e}")))?
}

/// Query similar episodes without blocking the async runtime.
pub async fn query_similar_async(
    db: Arc<RwLock<AgentMemDB>>,
    emb: Vec<f32>,
    opts: QueryOptions,
) -> Result<Vec<Episode>, AgentMemError> {
    tokio::task::spawn_blocking(move || {
        let guard = db.read().unwrap();
        guard.query_similar_with_options(&emb, opts)
    })
    .await
    .map_err(|e| AgentMemError::HnswError(format!("spawn_blocking: {e}")))?
}

/// Save DB to file without blocking the async runtime.
pub async fn save_to_file_async(
    db: Arc<RwLock<AgentMemDB>>,
    path: PathBuf,
) -> Result<(), AgentMemError> {
    tokio::task::spawn_blocking(move || {
        let guard = db.read().unwrap();
        guard.save_to_file(&path)
    })
    .await
    .map_err(|e| AgentMemError::HnswError(format!("spawn_blocking: {e}")))?
}

/// Load DB from file without blocking the async runtime.
pub async fn load_from_file_async(path: PathBuf) -> Result<Arc<RwLock<AgentMemDB>>, AgentMemError> {
    let db = tokio::task::spawn_blocking(move || AgentMemDB::load_from_file(&path))
        .await
        .map_err(|e| AgentMemError::HnswError(format!("spawn_blocking: {e}")))??;
    Ok(Arc::new(RwLock::new(db)))
}
