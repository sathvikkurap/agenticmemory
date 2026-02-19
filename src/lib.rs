/// A single step in an agent trajectory.
///
/// `EpisodeStep` is optional metadata attached to an `Episode` that records a
/// per-step trace (action, observation, and per-step reward). This is useful
/// when an agent wants to store a short trajectory alongside the episode-level
/// summary (e.g., total `reward`).
///
/// Example:
///
/// ```rust
/// use agent_mem_db::EpisodeStep;
/// let step = EpisodeStep { index: 0, action: "move".into(), observation: "obs".into(), step_reward: 0.1 };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeStep {
    /// Step index (0-based)
    pub index: u32,
    /// Action taken by the agent
    pub action: String,
    /// Observation received
    pub observation: String,
    /// Reward for this step
    pub step_reward: f32,
}
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
#[derive(Serialize, Deserialize)]
struct PersistedDB {
    dim: usize,
    episodes: Vec<Episode>,
}

mod disk;
mod index;
pub use disk::{AgentMemDBDisk, DiskOptions};

#[cfg(feature = "async")]
pub mod async_api;
use index::{ExactIndex, HnswIndex, IndexBackend};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

/// A recorded agent experience (episode).
///
/// `Episode` contains a unique `id`, a `task_id` (user-defined), the
/// `state_embedding` vector used for similarity search, a scalar `reward`,
/// optional `metadata` (arbitrary JSON), and an optional sequence of
/// `steps` (see `EpisodeStep`).
///
/// Construct with `Episode::new(...)` which generates a UUID v4 automatically:
///
/// ```rust
/// use agent_mem_db::Episode;
/// let ep = Episode::new("task_x", vec![0.0f32; 16], 1.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique episode id (UUID v4)
    pub id: Uuid,
    /// Task identifier (user-defined)
    pub task_id: String,
    /// State embedding vector (e.g., 768-dim)
    pub state_embedding: Vec<f32>,
    /// Reward for this episode (e.g., -1.0 to 1.0)
    pub reward: f32,
    /// Optional metadata (arbitrary JSON)
    pub metadata: Value,
    /// Optional sequence of steps (trajectory)
    pub steps: Option<Vec<EpisodeStep>>,
    /// Optional Unix timestamp (milliseconds) for time-based filtering
    #[serde(default)]
    pub timestamp: Option<i64>,
    /// Optional tags for categorical filtering
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Optional source (e.g., "api", "cli")
    #[serde(default)]
    pub source: Option<String>,
    /// Optional user id for multi-tenant isolation
    #[serde(default)]
    pub user_id: Option<String>,
}
impl Episode {
    /// Create a new episode with a random UUID and empty metadata.
    pub fn new(task_id: impl Into<String>, state_embedding: Vec<f32>, reward: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            task_id: task_id.into(),
            state_embedding,
            reward,
            metadata: Value::Null,
            steps: None,
            timestamp: None,
            tags: None,
            source: None,
            user_id: None,
        }
    }

    /// Create an episode with a timestamp (Unix ms).
    pub fn with_timestamp(
        task_id: impl Into<String>,
        state_embedding: Vec<f32>,
        reward: f32,
        timestamp: i64,
    ) -> Self {
        let mut ep = Self::new(task_id, state_embedding, reward);
        ep.timestamp = Some(timestamp);
        ep
    }

    /// Create an episode with tags.
    pub fn with_tags(
        task_id: impl Into<String>,
        state_embedding: Vec<f32>,
        reward: f32,
        tags: Vec<String>,
    ) -> Self {
        let mut ep = Self::new(task_id, state_embedding, reward);
        ep.tags = Some(tags);
        ep
    }

    /// Create an episode with source.
    pub fn with_source(
        task_id: impl Into<String>,
        state_embedding: Vec<f32>,
        reward: f32,
        source: impl Into<String>,
    ) -> Self {
        let mut ep = Self::new(task_id, state_embedding, reward);
        ep.source = Some(source.into());
        ep
    }

    /// Create an episode with user_id.
    pub fn with_user_id(
        task_id: impl Into<String>,
        state_embedding: Vec<f32>,
        reward: f32,
        user_id: impl Into<String>,
    ) -> Self {
        let mut ep = Self::new(task_id, state_embedding, reward);
        ep.user_id = Some(user_id.into());
        ep
    }
}

/// Query options for similarity search with optional filters.
#[derive(Debug, Clone, Default)]
pub struct QueryOptions {
    /// Minimum episode reward to include
    pub min_reward: f32,
    /// Maximum number of episodes to return
    pub top_k: usize,
    /// Include only episodes that have any of these tags
    pub tags_any: Option<Vec<String>>,
    /// Include only episodes that have all of these tags
    pub tags_all: Option<Vec<String>>,
    /// Include only episodes with task_id starting with this prefix
    pub task_id_prefix: Option<String>,
    /// Include only episodes with timestamp >= (Unix ms)
    pub time_after: Option<i64>,
    /// Include only episodes with timestamp <= (Unix ms)
    pub time_before: Option<i64>,
    /// Include only episodes with this source (exact match)
    pub source: Option<String>,
    /// Include only episodes with this user_id (exact match)
    pub user_id: Option<String>,
}

impl QueryOptions {
    /// Simple options: min_reward and top_k only.
    pub fn new(min_reward: f32, top_k: usize) -> Self {
        Self {
            min_reward,
            top_k,
            ..Default::default()
        }
    }

    /// Add tags_any filter.
    pub fn tags_any(mut self, tags: Vec<String>) -> Self {
        self.tags_any = Some(tags);
        self
    }

    /// Add tags_all filter (episode must have all of these tags).
    pub fn tags_all(mut self, tags: Vec<String>) -> Self {
        self.tags_all = Some(tags);
        self
    }

    /// Add task_id_prefix filter.
    pub fn task_id_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.task_id_prefix = Some(prefix.into());
        self
    }

    /// Add time_after filter (timestamp >=).
    pub fn time_after(mut self, ts: i64) -> Self {
        self.time_after = Some(ts);
        self
    }

    /// Add time_before filter (timestamp <=).
    pub fn time_before(mut self, ts: i64) -> Self {
        self.time_before = Some(ts);
        self
    }

    /// Add source filter (exact match).
    pub fn source(mut self, s: impl Into<String>) -> Self {
        self.source = Some(s.into());
        self
    }

    /// Add user_id filter (exact match).
    pub fn user_id(mut self, u: impl Into<String>) -> Self {
        self.user_id = Some(u.into());
        self
    }

    pub(crate) fn matches(&self, ep: &Episode) -> bool {
        if ep.reward < self.min_reward {
            return false;
        }
        if let Some(ref tags) = self.tags_any {
            let ep_tags = ep.tags.as_deref().unwrap_or(&[]);
            if !tags.iter().any(|t| ep_tags.contains(t)) {
                return false;
            }
        }
        if let Some(ref tags) = self.tags_all {
            let ep_tags = ep.tags.as_deref().unwrap_or(&[]);
            if !tags.iter().all(|t| ep_tags.contains(t)) {
                return false;
            }
        }
        if let Some(ref prefix) = self.task_id_prefix {
            if !ep.task_id.starts_with(prefix) {
                return false;
            }
        }
        if let Some(ts) = self.time_after {
            if let Some(ep_ts) = ep.timestamp {
                if ep_ts < ts {
                    return false;
                }
            } else {
                return false;
            }
        }
        if let Some(ts) = self.time_before {
            if let Some(ep_ts) = ep.timestamp {
                if ep_ts > ts {
                    return false;
                }
            } else {
                return false;
            }
        }
        if let Some(ref s) = self.source {
            if ep.source.as_deref() != Some(s.as_str()) {
                return false;
            }
        }
        if let Some(ref u) = self.user_id {
            if ep.user_id.as_deref() != Some(u.as_str()) {
                return false;
            }
        }
        true
    }
}

/// In-memory agent memory database with HNSW approximate nearest-neighbour search.
///
/// `AgentMemDB` stores `Episode` records keyed by UUID and maintains an
/// HNSW index over `state_embedding` vectors for fast approximate nearest
/// neighbour (ANN) queries. Episodes can be filtered by reward at query time.
///
/// High-level example:
///
/// ```rust
/// use agent_mem_db::{AgentMemDB, Episode};
/// let mut db = AgentMemDB::new(16);
/// let ep = Episode::new("task1", vec![0.0f32; 16], 1.0);
/// db.store_episode(ep).unwrap();
/// let hits = db.query_similar(&vec![0.0f32;16], 0.0, 5).unwrap();
/// ```
pub struct AgentMemDB {
    dim: usize,
    episodes: HashMap<Uuid, Episode>,
    index: IndexBackend,
    key_to_uuid: HashMap<usize, Uuid>,
}

#[derive(Error, Debug)]
pub enum AgentMemError {
    #[error("Embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("HNSW or IO error: {0}")]
    HnswError(String),
    // Add bincode to dependencies
    #[error("Episode not found")]
    NotFound,
}

impl AgentMemDB {
    /// Create a new empty AgentMemDB for a given embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self::new_with_max_elements(dim, 20_000)
    }

    /// Create a new empty AgentMemDB with a custom max_elements (for scale workloads).
    pub fn new_with_max_elements(dim: usize, max_elements: usize) -> Self {
        Self {
            dim,
            episodes: HashMap::new(),
            index: IndexBackend::Hnsw(Box::new(HnswIndex::new(max_elements))),
            key_to_uuid: HashMap::new(),
        }
    }

    /// Create a new empty AgentMemDB with exact (brute-force) search. Use for small episode sets
    /// or when correctness is critical. O(n) per query.
    pub fn new_exact(dim: usize) -> Self {
        Self {
            dim,
            episodes: HashMap::new(),
            index: IndexBackend::Exact(ExactIndex::new()),
            key_to_uuid: HashMap::new(),
        }
    }

    /// Return the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Store an episode in memory and update the HNSW index.
    /// Returns an error if the embedding dimension does not match.
    ///
    /// Example:
    ///
    /// ```rust
    /// use agent_mem_db::{AgentMemDB, Episode};
    /// let mut db = AgentMemDB::new(16);
    /// let ep = Episode::new("t", vec![0.0f32; 16], 0.5);
    /// db.store_episode(ep).unwrap();
    /// ```
    pub fn store_episode(&mut self, episode: Episode) -> Result<(), AgentMemError> {
        if episode.state_embedding.len() != self.dim {
            return Err(AgentMemError::DimensionMismatch {
                expected: self.dim,
                got: episode.state_embedding.len(),
            });
        }
        let id = episode.id;
        let key = self.index.insert(&episode.state_embedding);
        self.key_to_uuid.insert(key, id);
        self.episodes.insert(id, episode);
        Ok(())
    }

    /// Query for top_k most similar episodes to the given embedding, filtered by min_reward.
    /// Returns up to top_k episodes with reward >= min_reward, ordered by similarity.
    ///
    /// Parameters:
    /// - `query_embedding`: slice with the same dimensionality as the DB.
    /// - `min_reward`: minimum episode reward to include in results.
    /// - `top_k`: maximum number of episodes to return.
    ///
    /// Example:
    ///
    /// ```rust
    /// # use agent_mem_db::{AgentMemDB, Episode};
    /// # let mut db = AgentMemDB::new(16);
    /// # let ep = Episode::new("t", vec![0.0f32; 16], 1.0);
    /// # db.store_episode(ep).unwrap();
    /// let res = db.query_similar(&vec![0.0f32;16], 0.0, 3).unwrap();
    /// ```
    pub fn query_similar(
        &self,
        query_embedding: &[f32],
        min_reward: f32,
        top_k: usize,
    ) -> Result<Vec<Episode>, AgentMemError> {
        self.query_similar_with_options(query_embedding, QueryOptions::new(min_reward, top_k))
    }

    /// Query with full filter options (tags, time range).
    pub fn query_similar_with_options(
        &self,
        query_embedding: &[f32],
        opts: QueryOptions,
    ) -> Result<Vec<Episode>, AgentMemError> {
        if query_embedding.len() != self.dim {
            return Err(AgentMemError::DimensionMismatch {
                expected: self.dim,
                got: query_embedding.len(),
            });
        }
        let candidate_mult = if opts.tags_any.is_some()
            || opts.tags_all.is_some()
            || opts.task_id_prefix.is_some()
            || opts.time_after.is_some()
            || opts.time_before.is_some()
            || opts.source.is_some()
            || opts.user_id.is_some()
        {
            4
        } else {
            2
        };
        let results = self
            .index
            .search(query_embedding, opts.top_k * candidate_mult);
        let mut candidates: Vec<(f32, Episode)> = results
            .into_iter()
            .filter_map(|(key, dist)| {
                self.key_to_uuid
                    .get(&key)
                    .and_then(|uuid| self.episodes.get(uuid))
                    .filter(|ep| opts.matches(ep))
                    .map(|ep| (dist, ep.clone()))
            })
            .collect();
        // Sort by distance asc; tie-break by recency (recent first). Episodes without timestamp sort last.
        candidates.sort_by(|a, b| {
            let dist_cmp = a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal);
            if dist_cmp != std::cmp::Ordering::Equal {
                return dist_cmp;
            }
            let ts_a = a.1.timestamp.unwrap_or(i64::MIN);
            let ts_b = b.1.timestamp.unwrap_or(i64::MIN);
            ts_b.cmp(&ts_a)
        });
        let episodes: Vec<Episode> = candidates
            .into_iter()
            .take(opts.top_k)
            .map(|(_, ep)| ep)
            .collect();
        Ok(episodes)
    }

    /// Store multiple episodes in memory and update the HNSW index for each.
    ///
    /// This is a convenience batch API that calls `store_episode` for each entry.
    /// If you need higher performance for very large batches, consider a bulk
    /// construction API (not implemented here) or increase `max_elements` in the
    /// HNSW configuration used at construction time.
    pub fn store_episodes(&mut self, episodes: Vec<Episode>) -> Result<(), AgentMemError> {
        for ep in episodes {
            self.store_episode(ep)?;
        }
        Ok(())
    }

    /// Query for similar episodes for a batch of queries.
    ///
    /// Returns a `Vec` of `Vec<Episode>`, one per query embedding. This is a
    /// thin wrapper around repeated calls to `query_similar`.
    pub fn query_similar_batch(
        &self,
        queries: &[Vec<f32>],
        min_reward: f32,
        top_k: usize,
    ) -> Result<Vec<Vec<Episode>>, AgentMemError> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            results.push(self.query_similar(q, min_reward, top_k)?);
        }
        Ok(results)
    }

    /// Save all episodes to a JSON file. On load, the HNSW index is rebuilt.
    pub fn save_to_file(&self, path: &Path) -> Result<(), AgentMemError> {
        let file = File::create(path)
            .map_err(|e| AgentMemError::HnswError(format!("File create: {e}")))?;
        let writer = BufWriter::new(file);
        let persisted = PersistedDB {
            dim: self.dim,
            episodes: self.episodes.values().cloned().collect(),
        };
        serde_json::to_writer(writer, &persisted)
            .map_err(|e| AgentMemError::HnswError(format!("Serialize: {e}")))?;
        Ok(())
    }

    /// Load episodes from a JSON file and rebuild the index. Uses HNSW backend by default.
    pub fn load_from_file(path: &Path) -> Result<Self, AgentMemError> {
        Self::load_from_file_with_index(path, false)
    }

    /// Load episodes from a JSON file, using exact (brute-force) search. Deterministic results.
    pub fn load_from_file_exact(path: &Path) -> Result<Self, AgentMemError> {
        Self::load_from_file_with_index(path, true)
    }

    /// Prune episodes with timestamp older than cutoff (Unix ms).
    /// Episodes without timestamp are kept. Returns the number of episodes removed.
    /// Rebuilds the index internally (HNSW/Exact do not support in-place removal).
    pub fn prune_older_than(&mut self, timestamp_cutoff_ms: i64) -> usize {
        let kept: Vec<Episode> = self
            .episodes
            .values()
            .filter(|ep| {
                ep.timestamp
                    .map(|t| t >= timestamp_cutoff_ms)
                    .unwrap_or(true)
            })
            .cloned()
            .collect();
        let removed = self.episodes.len() - kept.len();
        self.episodes.clear();
        self.key_to_uuid.clear();
        let was_exact = matches!(&self.index, IndexBackend::Exact(_));
        self.index = if was_exact {
            IndexBackend::Exact(ExactIndex::new())
        } else {
            IndexBackend::Hnsw(Box::new(HnswIndex::new(
                kept.len().max(20_000).max(self.dim * 2),
            )))
        };
        for ep in kept {
            let id = ep.id;
            let key = self.index.insert(&ep.state_embedding);
            self.key_to_uuid.insert(key, id);
            self.episodes.insert(id, ep);
        }
        removed
    }

    /// Prune to keep only the n most recent episodes (by timestamp).
    /// Episodes without timestamp are treated as oldest and pruned first. Returns episodes removed.
    pub fn prune_keep_newest(&mut self, n: usize) -> usize {
        if self.episodes.len() <= n {
            return 0;
        }
        let mut episodes: Vec<Episode> = self.episodes.drain().map(|(_, ep)| ep).collect();
        let original = episodes.len();
        episodes.sort_by(|a, b| {
            let ts_a = a.timestamp.unwrap_or(i64::MIN);
            let ts_b = b.timestamp.unwrap_or(i64::MIN);
            ts_b.cmp(&ts_a)
        });
        let kept: Vec<Episode> = episodes.into_iter().take(n).collect();
        let removed = original - kept.len();
        self.key_to_uuid.clear();
        let was_exact = matches!(&self.index, IndexBackend::Exact(_));
        self.index = if was_exact {
            IndexBackend::Exact(ExactIndex::new())
        } else {
            IndexBackend::Hnsw(Box::new(HnswIndex::new(
                kept.len().max(20_000).max(self.dim * 2),
            )))
        };
        for ep in kept {
            let id = ep.id;
            let key = self.index.insert(&ep.state_embedding);
            self.key_to_uuid.insert(key, id);
            self.episodes.insert(id, ep);
        }
        removed
    }

    /// Prune to keep only the n episodes with highest reward.
    /// Ties: prefer more recent (higher timestamp); episodes without timestamp sort last. Returns episodes removed.
    pub fn prune_keep_highest_reward(&mut self, n: usize) -> usize {
        if self.episodes.len() <= n {
            return 0;
        }
        let mut episodes: Vec<Episode> = self.episodes.drain().map(|(_, ep)| ep).collect();
        let original = episodes.len();
        episodes.sort_by(|a, b| {
            let reward_cmp = b
                .reward
                .partial_cmp(&a.reward)
                .unwrap_or(std::cmp::Ordering::Equal);
            if reward_cmp != std::cmp::Ordering::Equal {
                return reward_cmp;
            }
            let ts_a = a.timestamp.unwrap_or(i64::MIN);
            let ts_b = b.timestamp.unwrap_or(i64::MIN);
            ts_b.cmp(&ts_a)
        });
        let kept: Vec<Episode> = episodes.into_iter().take(n).collect();
        let removed = original - kept.len();
        self.key_to_uuid.clear();
        let was_exact = matches!(&self.index, IndexBackend::Exact(_));
        self.index = if was_exact {
            IndexBackend::Exact(ExactIndex::new())
        } else {
            IndexBackend::Hnsw(Box::new(HnswIndex::new(
                kept.len().max(20_000).max(self.dim * 2),
            )))
        };
        for ep in kept {
            let id = ep.id;
            let key = self.index.insert(&ep.state_embedding);
            self.key_to_uuid.insert(key, id);
            self.episodes.insert(id, ep);
        }
        removed
    }

    fn load_from_file_with_index(path: &Path, use_exact: bool) -> Result<Self, AgentMemError> {
        let file =
            File::open(path).map_err(|e| AgentMemError::HnswError(format!("File open: {e}")))?;
        let reader = BufReader::new(file);
        let persisted: PersistedDB = serde_json::from_reader(reader)
            .map_err(|e| AgentMemError::HnswError(format!("Deserialize: {e}")))?;
        let mut db = if use_exact {
            AgentMemDB::new_exact(persisted.dim)
        } else {
            AgentMemDB::new(persisted.dim)
        };
        for ep in persisted.episodes {
            db.store_episode(ep)
                .map_err(|e| AgentMemError::HnswError(format!("Reinsert: {e}")))?;
        }
        Ok(db)
    }
}
