//! Node.js bindings for agent_mem_db.

use agent_mem_db::{
    AgentMemDB as RustAgentMemDB, AgentMemDBDisk as RustAgentMemDBDisk, DiskOptions,
    Episode as RustEpisode, QueryOptions,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::path::Path;

fn f64_to_f32(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

fn f32_to_f64(v: Vec<f32>) -> Vec<f64> {
    v.into_iter().map(|x| x as f64).collect()
}

/// Episode for agent memory. Pass to storeEpisode.
#[napi(object)]
pub struct Episode {
    pub id: String,
    pub task_id: String,
    pub state_embedding: Vec<f64>,
    pub reward: f64,
    pub metadata: Option<serde_json::Value>,
    pub timestamp: Option<i64>,
    pub tags: Option<Vec<String>>,
    pub source: Option<String>,
    pub user_id: Option<String>,
}

impl From<RustEpisode> for Episode {
    fn from(ep: RustEpisode) -> Self {
        let metadata = if ep.metadata == serde_json::Value::Null {
            None
        } else {
            Some(ep.metadata)
        };
        Self {
            id: ep.id.to_string(),
            task_id: ep.task_id,
            state_embedding: f32_to_f64(ep.state_embedding),
            reward: ep.reward as f64,
            metadata,
            timestamp: ep.timestamp,
            tags: ep.tags,
            source: ep.source,
            user_id: ep.user_id,
        }
    }
}

impl From<Episode> for RustEpisode {
    fn from(ep: Episode) -> Self {
        let mut rust =
            RustEpisode::new(ep.task_id, f64_to_f32(ep.state_embedding), ep.reward as f32);
        rust.metadata = ep.metadata.unwrap_or(serde_json::Value::Null);
        rust.timestamp = ep.timestamp;
        rust.tags = ep.tags;
        rust.source = ep.source;
        rust.user_id = ep.user_id;
        rust
    }
}

/// Query options for similarity search.
#[napi(object)]
pub struct QueryOptionsJs {
    pub min_reward: f64,
    pub top_k: u32,
    pub tags_any: Option<Vec<String>>,
    pub tags_all: Option<Vec<String>>,
    pub task_id_prefix: Option<String>,
    pub time_after: Option<i64>,
    pub time_before: Option<i64>,
    pub source: Option<String>,
    pub user_id: Option<String>,
}

/// In-memory agent memory DB with HNSW vector search.
#[napi]
pub struct AgentMemDB {
    inner: std::sync::Mutex<RustAgentMemDB>,
}

#[napi]
impl AgentMemDB {
    #[napi(constructor)]
    pub fn new(dim: u32) -> Self {
        Self {
            inner: std::sync::Mutex::new(RustAgentMemDB::new(dim as usize)),
        }
    }

    /// Create with exact (brute-force) search. O(n) per query.
    #[napi(factory)]
    pub fn exact(dim: u32) -> Self {
        Self {
            inner: std::sync::Mutex::new(RustAgentMemDB::new_exact(dim as usize)),
        }
    }

    /// Create with custom max_elements for scale.
    #[napi(factory)]
    pub fn with_max_elements(dim: u32, max_elements: u32) -> Self {
        Self {
            inner: std::sync::Mutex::new(RustAgentMemDB::new_with_max_elements(
                dim as usize,
                max_elements as usize,
            )),
        }
    }

    /// Store an episode.
    #[napi]
    pub fn store_episode(&self, episode: Episode) -> Result<()> {
        let rust_ep: RustEpisode = episode.into();
        self.inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?
            .store_episode(rust_ep)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Query for similar episodes. embedding: number[], min_reward, top_k. Optional opts for filters.
    #[napi]
    pub fn query_similar(
        &self,
        embedding: Vec<f64>,
        min_reward: f64,
        top_k: u32,
        opts: Option<QueryOptionsJs>,
    ) -> Result<Vec<Episode>> {
        let db = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?;
        let query_opts = opts
            .map(|o| {
                let mut q = QueryOptions::new(o.min_reward as f32, o.top_k as usize);
                q.tags_any = o.tags_any;
                q.tags_all = o.tags_all;
                q.task_id_prefix = o.task_id_prefix;
                q.time_after = o.time_after;
                q.time_before = o.time_before;
                q.source = o.source;
                q.user_id = o.user_id;
                q
            })
            .unwrap_or_else(|| QueryOptions::new(min_reward as f32, top_k as usize));
        let emb_f32 = f64_to_f32(embedding);
        let results = db
            .query_similar_with_options(&emb_f32, query_opts)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(results.into_iter().map(Episode::from).collect())
    }

    /// Save to JSON file.
    #[napi]
    pub fn save_to_file(&self, path: String) -> Result<()> {
        let db = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?;
        db.save_to_file(Path::new(&path))
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Load from JSON file.
    #[napi(factory)]
    pub fn load_from_file(path: String) -> Result<Self> {
        let db = RustAgentMemDB::load_from_file(Path::new(&path))
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self {
            inner: std::sync::Mutex::new(db),
        })
    }

    /// Prune episodes with timestamp older than cutoff (Unix ms). Episodes without timestamp are kept.
    #[napi]
    pub fn prune_older_than(&self, timestamp_cutoff_ms: i64) -> u32 {
        self.inner
            .lock()
            .map(|mut db| db.prune_older_than(timestamp_cutoff_ms) as u32)
            .unwrap_or(0)
    }

    /// Prune to keep only the n most recent episodes (by timestamp).
    #[napi]
    pub fn prune_keep_newest(&self, n: u32) -> u32 {
        self.inner
            .lock()
            .map(|mut db| db.prune_keep_newest(n as usize) as u32)
            .unwrap_or(0)
    }

    /// Prune to keep only the n episodes with highest reward.
    #[napi]
    pub fn prune_keep_highest_reward(&self, n: u32) -> u32 {
        self.inner
            .lock()
            .map(|mut db| db.prune_keep_highest_reward(n as usize) as u32)
            .unwrap_or(0)
    }
}

/// Disk-backed agent memory DB. Episodes stored in append-only log; index in RAM.
#[napi]
pub struct AgentMemDBDisk {
    inner: std::sync::Mutex<RustAgentMemDBDisk>,
}

#[napi]
impl AgentMemDBDisk {
    /// Open or create a disk-backed DB at the given directory. Uses HNSW by default.
    #[napi(factory)]
    pub fn open(path: String, dim: u32) -> Result<Self> {
        let db = RustAgentMemDBDisk::open(Path::new(&path), dim as usize)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self {
            inner: std::sync::Mutex::new(db),
        })
    }

    /// Open with exact index and checkpoint enabled for fast restart. Call checkpoint() after stores.
    #[napi(factory)]
    pub fn open_exact_with_checkpoint(path: String, dim: u32) -> Result<Self> {
        let db = RustAgentMemDBDisk::open_with_options(
            Path::new(&path),
            DiskOptions::exact_with_checkpoint(dim as usize),
        )
        .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self {
            inner: std::sync::Mutex::new(db),
        })
    }

    /// Store an episode.
    #[napi]
    pub fn store_episode(&self, episode: Episode) -> Result<()> {
        let rust_ep: RustEpisode = episode.into();
        self.inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?
            .store_episode(rust_ep)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Query for similar episodes.
    #[napi]
    pub fn query_similar(
        &self,
        embedding: Vec<f64>,
        min_reward: f64,
        top_k: u32,
        opts: Option<QueryOptionsJs>,
    ) -> Result<Vec<Episode>> {
        let db = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?;
        let query_opts = opts
            .map(|o| {
                let mut q = QueryOptions::new(o.min_reward as f32, o.top_k as usize);
                q.tags_any = o.tags_any;
                q.tags_all = o.tags_all;
                q.task_id_prefix = o.task_id_prefix;
                q.time_after = o.time_after;
                q.time_before = o.time_before;
                q.source = o.source;
                q.user_id = o.user_id;
                q
            })
            .unwrap_or_else(|| QueryOptions::new(min_reward as f32, top_k as usize));
        let emb_f32 = f64_to_f32(embedding);
        let results = db
            .query_similar_with_options(&emb_f32, query_opts)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(results.into_iter().map(Episode::from).collect())
    }

    /// Persist checkpoint for fast restart (ExactIndex only). No-op for HNSW.
    #[napi]
    pub fn checkpoint(&self) -> Result<()> {
        self.inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?
            .checkpoint()
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Prune episodes with timestamp older than cutoff (Unix ms).
    #[napi]
    pub fn prune_older_than(&self, timestamp_cutoff_ms: i64) -> Result<u32> {
        self.inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?
            .prune_older_than(timestamp_cutoff_ms)
            .map(|n| n as u32)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Prune to keep only the n most recent episodes.
    #[napi]
    pub fn prune_keep_newest(&self, n: u32) -> Result<u32> {
        self.inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?
            .prune_keep_newest(n as usize)
            .map(|r| r as u32)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Prune to keep only the n episodes with highest reward.
    #[napi]
    pub fn prune_keep_highest_reward(&self, n: u32) -> Result<u32> {
        self.inner
            .lock()
            .map_err(|e| Error::from_reason(format!("lock: {e}")))?
            .prune_keep_highest_reward(n as usize)
            .map(|r| r as u32)
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}

/// Create a new Episode. id is auto-generated.
#[napi]
pub fn create_episode(
    task_id: String,
    state_embedding: Vec<f64>,
    reward: f64,
    metadata: Option<serde_json::Value>,
    timestamp: Option<i64>,
    tags: Option<Vec<String>>,
    source: Option<String>,
    user_id: Option<String>,
) -> Episode {
    let mut rust = RustEpisode::new(
        task_id.clone(),
        f64_to_f32(state_embedding.clone()),
        reward as f32,
    );
    rust.metadata = metadata.clone().unwrap_or(serde_json::Value::Null);
    rust.timestamp = timestamp;
    rust.tags = tags.clone();
    rust.source = source.clone();
    rust.user_id = user_id.clone();
    Episode {
        id: rust.id.to_string(),
        task_id,
        state_embedding,
        reward,
        metadata: if rust.metadata == serde_json::Value::Null {
            None
        } else {
            Some(rust.metadata)
        },
        timestamp,
        tags,
        source,
        user_id,
    }
}
