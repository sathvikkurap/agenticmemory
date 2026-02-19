//! Disk-backed agent memory DB. Episodes stored in append-only JSONL log; index in RAM.

use crate::index::{ExactIndex, HnswIndex, IndexBackend};
use crate::{AgentMemError, Episode, QueryOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const EPISODES_LOG: &str = "episodes.jsonl";
const META_FILE: &str = "meta.json";
const EXACT_CHECKPOINT_FILE: &str = "exact_checkpoint.json";

/// State loaded from checkpoint or replayed from log.
type LoadedState = (HashMap<Uuid, Episode>, HashMap<usize, Uuid>, IndexBackend);

#[derive(Serialize, Deserialize)]
struct DiskMeta {
    dim: usize,
    index_type: String, // "hnsw" | "exact"
    max_elements: usize,
    #[serde(default)]
    checkpoint_line_count: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct ExactCheckpoint {
    episodes: Vec<Episode>,
}

/// Disk-backed agent memory DB. Episodes stored in append-only log; index in RAM.
///
/// Use for episode sets that exceed RAM or when durability is required.
/// On open, replays the log to rebuild the index (or loads from checkpoint when valid).
pub struct AgentMemDBDisk {
    dim: usize,
    episodes: HashMap<Uuid, Episode>,
    index: IndexBackend,
    key_to_uuid: HashMap<usize, Uuid>,
    #[allow(dead_code)] // Reserved for compaction, retention APIs
    path: PathBuf,
    log_file: File,
    use_checkpoint: bool,
}

impl AgentMemDBDisk {
    /// Open or create a disk-backed DB at the given directory.
    /// Uses HNSW with default max_elements (20_000).
    pub fn open(path: impl AsRef<Path>, dim: usize) -> Result<Self, AgentMemError> {
        Self::open_with_options(path, DiskOptions::hnsw(dim, 20_000))
    }

    /// Open with explicit options (index type, max_elements).
    pub fn open_with_options(
        path: impl AsRef<Path>,
        opts: DiskOptions,
    ) -> Result<Self, AgentMemError> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)
            .map_err(|e| AgentMemError::HnswError(format!("Create dir: {e}")))?;

        let meta_path = path.join(META_FILE);
        let log_path = path.join(EPISODES_LOG);

        let (dim, index, episodes, key_to_uuid) = if meta_path.exists() {
            // Load existing
            let meta: DiskMeta = serde_json::from_str(
                &fs::read_to_string(&meta_path)
                    .map_err(|e| AgentMemError::HnswError(format!("Read meta: {e}")))?,
            )
            .map_err(|e| AgentMemError::HnswError(format!("Parse meta: {e}")))?;

            if meta.dim != opts.dim {
                return Err(AgentMemError::HnswError(format!(
                    "Dimension mismatch: meta has {}, requested {}",
                    meta.dim, opts.dim
                )));
            }

            let index: IndexBackend = match meta.index_type.as_str() {
                "exact" => IndexBackend::Exact(ExactIndex::new()),
                _ => IndexBackend::Hnsw(Box::new(HnswIndex::new(meta.max_elements))),
            };

            let (episodes, key_to_uuid, index) = if log_path.exists() {
                let checkpoint_path = path.join(EXACT_CHECKPOINT_FILE);
                let try_checkpoint =
                    opts.use_checkpoint && meta.index_type == "exact" && checkpoint_path.exists();

                if try_checkpoint {
                    let line_count = Self::count_log_lines(&log_path)?;
                    if meta.checkpoint_line_count == Some(line_count) {
                        Self::load_from_checkpoint(&checkpoint_path, meta.dim)?
                    } else {
                        Self::replay_log(&log_path, meta.dim, meta.max_elements, &meta.index_type)?
                    }
                } else {
                    Self::replay_log(&log_path, meta.dim, meta.max_elements, &meta.index_type)?
                }
            } else {
                (HashMap::new(), HashMap::new(), index)
            };

            (meta.dim, index, episodes, key_to_uuid)
        } else {
            // Create new
            let index = match opts.index_type.as_deref() {
                Some("exact") => IndexBackend::Exact(ExactIndex::new()),
                _ => IndexBackend::Hnsw(Box::new(HnswIndex::new(opts.max_elements))),
            };

            let meta = DiskMeta {
                dim: opts.dim,
                index_type: opts.index_type.unwrap_or_else(|| "hnsw".to_string()),
                max_elements: opts.max_elements,
                checkpoint_line_count: None,
            };
            let meta_json = serde_json::to_string_pretty(&meta)
                .map_err(|e| AgentMemError::HnswError(format!("Serialize meta: {e}")))?;
            fs::write(&meta_path, meta_json)
                .map_err(|e| AgentMemError::HnswError(format!("Write meta: {e}")))?;

            (opts.dim, index, HashMap::new(), HashMap::new())
        };

        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| AgentMemError::HnswError(format!("Open log: {e}")))?;

        Ok(Self {
            dim,
            episodes,
            index,
            key_to_uuid,
            path,
            log_file,
            use_checkpoint: opts.use_checkpoint,
        })
    }

    fn count_log_lines(log_path: &Path) -> Result<usize, AgentMemError> {
        let file = File::open(log_path)
            .map_err(|e| AgentMemError::HnswError(format!("Open log for count: {e}")))?;
        let reader = BufReader::new(file);
        let count = reader
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty())
            .count();
        Ok(count)
    }

    fn load_from_checkpoint(
        checkpoint_path: &Path,
        dim: usize,
    ) -> Result<LoadedState, AgentMemError> {
        let data = fs::read_to_string(checkpoint_path)
            .map_err(|e| AgentMemError::HnswError(format!("Read checkpoint: {e}")))?;
        let cp: ExactCheckpoint = serde_json::from_str(&data)
            .map_err(|e| AgentMemError::HnswError(format!("Deserialize checkpoint: {e}")))?;

        let mut episodes = HashMap::new();
        let mut key_to_uuid = HashMap::new();
        let vectors: Vec<Vec<f32>> = cp
            .episodes
            .iter()
            .map(|ep| {
                if ep.state_embedding.len() != dim {
                    Err(AgentMemError::DimensionMismatch {
                        expected: dim,
                        got: ep.state_embedding.len(),
                    })
                } else {
                    Ok(ep.state_embedding.clone())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (i, ep) in cp.episodes.into_iter().enumerate() {
            key_to_uuid.insert(i, ep.id);
            episodes.insert(ep.id, ep);
        }

        let index = IndexBackend::Exact(ExactIndex::from_vectors(vectors));
        Ok((episodes, key_to_uuid, index))
    }

    fn replay_log(
        log_path: &Path,
        dim: usize,
        max_elements: usize,
        index_type: &str,
    ) -> Result<LoadedState, AgentMemError> {
        let file = File::open(log_path)
            .map_err(|e| AgentMemError::HnswError(format!("Open log for replay: {e}")))?;
        let reader = BufReader::new(file);
        let mut episodes = HashMap::new();
        let mut key_to_uuid = HashMap::new();

        let mut index: IndexBackend = match index_type {
            "exact" => IndexBackend::Exact(ExactIndex::new()),
            _ => IndexBackend::Hnsw(Box::new(HnswIndex::new(max_elements))),
        };

        for line in reader.lines() {
            let line = line.map_err(|e| AgentMemError::HnswError(format!("Read line: {e}")))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ep: Episode = serde_json::from_str(line)
                .map_err(|e| AgentMemError::HnswError(format!("Parse episode: {e}")))?;
            if ep.state_embedding.len() != dim {
                return Err(AgentMemError::DimensionMismatch {
                    expected: dim,
                    got: ep.state_embedding.len(),
                });
            }
            let id = ep.id;
            let key = index.insert(&ep.state_embedding);
            key_to_uuid.insert(key, id);
            episodes.insert(id, ep);
        }

        Ok((episodes, key_to_uuid, index))
    }

    /// Persist ExactIndex checkpoint for fast restart. No-op for HNSW or when checkpoint disabled.
    /// Call after storing episodes to avoid full replay on next open.
    pub fn checkpoint(&mut self) -> Result<(), AgentMemError> {
        if !self.use_checkpoint {
            return Ok(());
        }
        let IndexBackend::Exact(_) = &self.index else {
            return Ok(());
        };

        let line_count = Self::count_log_lines(&self.path.join(EPISODES_LOG))?;
        let episodes: Vec<Episode> = (0..self.index.len())
            .filter_map(|key| {
                self.key_to_uuid
                    .get(&key)
                    .and_then(|id| self.episodes.get(id))
            })
            .cloned()
            .collect();

        if episodes.len() != line_count {
            return Ok(());
        }

        let cp = ExactCheckpoint { episodes };
        let data = serde_json::to_string(&cp)
            .map_err(|e| AgentMemError::HnswError(format!("Serialize checkpoint: {e}")))?;
        let checkpoint_path = self.path.join(EXACT_CHECKPOINT_FILE);
        fs::write(&checkpoint_path, data)
            .map_err(|e| AgentMemError::HnswError(format!("Write checkpoint: {e}")))?;

        let meta_path = self.path.join(META_FILE);
        let meta: DiskMeta = serde_json::from_str(
            &fs::read_to_string(&meta_path)
                .map_err(|e| AgentMemError::HnswError(format!("Read meta: {e}")))?,
        )
        .map_err(|e| AgentMemError::HnswError(format!("Parse meta: {e}")))?;

        let updated = DiskMeta {
            checkpoint_line_count: Some(line_count),
            ..meta
        };
        let meta_json = serde_json::to_string_pretty(&updated)
            .map_err(|e| AgentMemError::HnswError(format!("Serialize meta: {e}")))?;
        fs::write(&meta_path, meta_json)
            .map_err(|e| AgentMemError::HnswError(format!("Write meta: {e}")))?;

        Ok(())
    }

    /// Store an episode: append to log and insert into index.
    pub fn store_episode(&mut self, episode: Episode) -> Result<(), AgentMemError> {
        if episode.state_embedding.len() != self.dim {
            return Err(AgentMemError::DimensionMismatch {
                expected: self.dim,
                got: episode.state_embedding.len(),
            });
        }
        let line = serde_json::to_string(&episode)
            .map_err(|e| AgentMemError::HnswError(format!("Serialize: {e}")))?;
        writeln!(self.log_file, "{}", line)
            .map_err(|e| AgentMemError::HnswError(format!("Write log: {e}")))?;
        self.log_file
            .sync_all()
            .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;

        let id = episode.id;
        let key = self.index.insert(&episode.state_embedding);
        self.key_to_uuid.insert(key, id);
        self.episodes.insert(id, episode);
        Ok(())
    }

    /// Query for top_k most similar episodes, filtered by min_reward.
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
        let candidate_mult =
            if opts.tags_any.is_some() || opts.time_after.is_some() || opts.time_before.is_some() {
                4
            } else {
                2
            };
        let results = self
            .index
            .search(query_embedding, opts.top_k * candidate_mult);
        let episodes: Vec<Episode> = results
            .into_iter()
            .filter_map(|(key, _)| {
                self.key_to_uuid
                    .get(&key)
                    .and_then(|uuid| self.episodes.get(uuid))
            })
            .filter(|ep| opts.matches(ep))
            .take(opts.top_k)
            .cloned()
            .collect();
        Ok(episodes)
    }

    /// Prune episodes with timestamp older than cutoff (Unix ms).
    /// Episodes without timestamp are kept. Compacts the log file. Returns episodes removed.
    pub fn prune_older_than(&mut self, timestamp_cutoff_ms: i64) -> Result<usize, AgentMemError> {
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
        if removed == 0 {
            return Ok(0);
        }

        self.episodes.clear();
        self.key_to_uuid.clear();
        let was_exact = matches!(&self.index, IndexBackend::Exact(_));
        self.index = if was_exact {
            IndexBackend::Exact(ExactIndex::new())
        } else {
            IndexBackend::Hnsw(Box::new(HnswIndex::new(kept.len().max(20_000).max(self.dim * 2))))
        };

        for ep in &kept {
            let id = ep.id;
            let key = self.index.insert(&ep.state_embedding);
            self.key_to_uuid.insert(key, id);
            self.episodes.insert(id, ep.clone());
        }

        let log_path = self.path.join(EPISODES_LOG);
        drop(std::mem::replace(&mut self.log_file, {
            let mut f = File::create(&log_path)
                .map_err(|e| AgentMemError::HnswError(format!("Create log for compaction: {e}")))?;
            for ep in &kept {
                let line = serde_json::to_string(ep)
                    .map_err(|e| AgentMemError::HnswError(format!("Serialize: {e}")))?;
                writeln!(f, "{}", line)
                    .map_err(|e| AgentMemError::HnswError(format!("Write log: {e}")))?;
            }
            f.sync_all()
                .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;
            drop(f);
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|e| AgentMemError::HnswError(format!("Reopen log: {e}")))?
        }));

        self.remove_checkpoint_if_exists()?;
        self.log_file
            .sync_all()
            .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;
        Ok(removed)
    }

    /// Prune to keep only the n most recent episodes (by timestamp). Compacts the log.
    /// Episodes without timestamp are treated as oldest. Returns episodes removed.
    pub fn prune_keep_newest(&mut self, n: usize) -> Result<usize, AgentMemError> {
        if self.episodes.len() <= n {
            return Ok(0);
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
            IndexBackend::Hnsw(Box::new(HnswIndex::new(kept.len().max(20_000).max(self.dim * 2))))
        };

        for ep in &kept {
            let id = ep.id;
            let key = self.index.insert(&ep.state_embedding);
            self.key_to_uuid.insert(key, id);
            self.episodes.insert(id, ep.clone());
        }

        let log_path = self.path.join(EPISODES_LOG);
        drop(std::mem::replace(&mut self.log_file, {
            let mut f = File::create(&log_path)
                .map_err(|e| AgentMemError::HnswError(format!("Create log for compaction: {e}")))?;
            for ep in &kept {
                let line = serde_json::to_string(ep)
                    .map_err(|e| AgentMemError::HnswError(format!("Serialize: {e}")))?;
                writeln!(f, "{}", line)
                    .map_err(|e| AgentMemError::HnswError(format!("Write log: {e}")))?;
            }
            f.sync_all()
                .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;
            drop(f);
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|e| AgentMemError::HnswError(format!("Reopen log: {e}")))?
        }));

        self.remove_checkpoint_if_exists()?;
        self.log_file
            .sync_all()
            .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;
        Ok(removed)
    }

    /// Prune to keep only the n episodes with highest reward. Compacts the log.
    pub fn prune_keep_highest_reward(&mut self, n: usize) -> Result<usize, AgentMemError> {
        if self.episodes.len() <= n {
            return Ok(0);
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
            IndexBackend::Hnsw(Box::new(HnswIndex::new(kept.len().max(20_000).max(self.dim * 2))))
        };

        for ep in &kept {
            let id = ep.id;
            let key = self.index.insert(&ep.state_embedding);
            self.key_to_uuid.insert(key, id);
            self.episodes.insert(id, ep.clone());
        }

        let log_path = self.path.join(EPISODES_LOG);
        drop(std::mem::replace(&mut self.log_file, {
            let mut f = File::create(&log_path)
                .map_err(|e| AgentMemError::HnswError(format!("Create log for compaction: {e}")))?;
            for ep in &kept {
                let line = serde_json::to_string(ep)
                    .map_err(|e| AgentMemError::HnswError(format!("Serialize: {e}")))?;
                writeln!(f, "{}", line)
                    .map_err(|e| AgentMemError::HnswError(format!("Write log: {e}")))?;
            }
            f.sync_all()
                .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;
            drop(f);
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|e| AgentMemError::HnswError(format!("Reopen log: {e}")))?
        }));

        self.remove_checkpoint_if_exists()?;
        self.log_file
            .sync_all()
            .map_err(|e| AgentMemError::HnswError(format!("Sync log: {e}")))?;
        Ok(removed)
    }

    fn remove_checkpoint_if_exists(&self) -> Result<(), AgentMemError> {
        let p = self.path.join(EXACT_CHECKPOINT_FILE);
        if p.exists() {
            fs::remove_file(&p)
                .map_err(|e| AgentMemError::HnswError(format!("Remove checkpoint: {e}")))?;
        }
        Ok(())
    }
}

/// Options for opening a disk-backed DB.
pub struct DiskOptions {
    pub dim: usize,
    pub index_type: Option<String>,
    pub max_elements: usize,
    /// If true and index is ExactIndex, enables checkpoint for fast restart.
    /// Call `checkpoint()` to persist; on next open, replay is skipped when checkpoint is valid.
    pub use_checkpoint: bool,
}

impl DiskOptions {
    pub fn hnsw(dim: usize, max_elements: usize) -> Self {
        Self {
            dim,
            index_type: Some("hnsw".to_string()),
            max_elements,
            use_checkpoint: false,
        }
    }

    pub fn exact(dim: usize) -> Self {
        Self {
            dim,
            index_type: Some("exact".to_string()),
            max_elements: 0, // unused for exact
            use_checkpoint: false,
        }
    }

    /// Exact index with checkpoint enabled for fast restart.
    pub fn exact_with_checkpoint(dim: usize) -> Self {
        Self {
            dim,
            index_type: Some("exact".to_string()),
            max_elements: 0,
            use_checkpoint: true,
        }
    }
}
