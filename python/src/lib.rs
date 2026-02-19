use agent_mem_db::{
    AgentMemDB as RustAgentMemDB, AgentMemDBDisk as RustAgentMemDBDisk, DiskOptions,
    Episode as RustEpisode, QueryOptions,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyType;
use serde_json::Value as JsonValue;
use std::path::Path;

#[pyclass]
pub struct Episode {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get, set)]
    pub task_id: String,
    #[pyo3(get, set)]
    pub state_embedding: Vec<f32>,
    #[pyo3(get, set)]
    pub reward: f32,
    #[pyo3(get, set)]
    pub metadata: Option<PyObject>,
    #[pyo3(get, set)]
    pub timestamp: Option<i64>,
    #[pyo3(get, set)]
    pub tags: Option<Vec<String>>,
    #[pyo3(get, set)]
    pub source: Option<String>,
    #[pyo3(get, set)]
    pub user_id: Option<String>,
}

#[pymethods]
impl Episode {
    #[new]
    #[pyo3(signature = (task_id, state_embedding, reward, metadata=None, timestamp=None, tags=None, source=None, user_id=None))]
    fn new(
        task_id: String,
        state_embedding: Vec<f32>,
        reward: f32,
        metadata: Option<PyObject>,
        timestamp: Option<i64>,
        tags: Option<Vec<String>>,
        source: Option<String>,
        user_id: Option<String>,
    ) -> Self {
        let rust_ep = RustEpisode::new(task_id.clone(), state_embedding.clone(), reward);
        let mut ep = Episode {
            id: rust_ep.id.to_string(),
            task_id,
            state_embedding,
            reward,
            metadata,
            timestamp: None,
            tags: None,
            source: None,
            user_id: None,
        };
        ep.timestamp = timestamp;
        ep.tags = tags;
        ep.source = source;
        ep.user_id = user_id;
        ep
    }
}

fn pyobj_to_json(py: Python, obj: &PyAny) -> PyResult<JsonValue> {
    // Use Python's json.dumps to serialize, then parse with serde_json
    let json = py
        .import("json")?
        .call_method1("dumps", (obj,))?
        .extract::<String>()?;
    serde_json::from_str(&json)
        .map_err(|e| PyValueError::new_err(format!("metadata serialize error: {e}")))
}

fn json_to_pyobject(py: Python, v: &JsonValue) -> PyResult<PyObject> {
    let s = serde_json::to_string(v)
        .map_err(|e| PyValueError::new_err(format!("metadata to string: {e}")))?;
    let obj = py.import("json")?.call_method1("loads", (s,))?;
    Ok(obj.into())
}

fn rust_episode_to_py(py: Python, ep: &agent_mem_db::Episode) -> PyResult<Episode> {
    let meta = if ep.metadata == JsonValue::Null {
        None
    } else {
        Some(json_to_pyobject(py, &ep.metadata)?)
    };
    Ok(Episode {
        id: ep.id.to_string(),
        task_id: ep.task_id.clone(),
        state_embedding: ep.state_embedding.clone(),
        reward: ep.reward,
        metadata: meta,
        timestamp: ep.timestamp,
        tags: ep.tags.clone(),
        source: ep.source.clone(),
        user_id: ep.user_id.clone(),
    })
}

fn results_to_py(py: Python, results: Vec<agent_mem_db::Episode>) -> PyResult<Vec<Episode>> {
    let mut out = Vec::with_capacity(results.len());
    for ep in results {
        out.push(rust_episode_to_py(py, &ep)?);
    }
    Ok(out)
}

#[pyclass]
pub struct AgentMemDB {
    db: RustAgentMemDB,
}

#[pymethods]
impl AgentMemDB {
    #[new]
    fn new(dim: usize) -> Self {
        AgentMemDB {
            db: RustAgentMemDB::new(dim),
        }
    }

    #[classmethod]
    fn with_max_elements(_cls: &PyType, dim: usize, max_elements: usize) -> Self {
        AgentMemDB {
            db: RustAgentMemDB::new_with_max_elements(dim, max_elements),
        }
    }

    /// Create a DB with exact (brute-force) search. O(n) per query; use for small episode sets.
    #[classmethod]
    fn exact(_cls: &PyType, dim: usize) -> Self {
        AgentMemDB {
            db: RustAgentMemDB::new_exact(dim),
        }
    }

    fn store_episode(&mut self, py: Python, episode: &Episode) -> PyResult<()> {
        let mut rust_ep = RustEpisode::new(
            episode.task_id.clone(),
            episode.state_embedding.clone(),
            episode.reward,
        );
        if let Some(meta) = &episode.metadata {
            let meta_val = pyobj_to_json(py, meta.as_ref(py))?;
            rust_ep.metadata = meta_val;
        }
        rust_ep.timestamp = episode.timestamp;
        rust_ep.tags = episode.tags.clone();
        rust_ep.source = episode.source.clone();
        rust_ep.user_id = episode.user_id.clone();
        self.db
            .store_episode(rust_ep)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (state_embedding, min_reward, top_k, tags_any=None, tags_all=None, task_id_prefix=None, time_after=None, time_before=None, source=None, user_id=None))]
    fn query_similar(
        &self,
        py: Python,
        state_embedding: Vec<f32>,
        min_reward: f32,
        top_k: usize,
        tags_any: Option<Vec<String>>,
        tags_all: Option<Vec<String>>,
        task_id_prefix: Option<String>,
        time_after: Option<i64>,
        time_before: Option<i64>,
        source: Option<String>,
        user_id: Option<String>,
    ) -> PyResult<Vec<Episode>> {
        let mut opts = QueryOptions::new(min_reward, top_k);
        if let Some(tags) = tags_any {
            opts.tags_any = Some(tags);
        }
        if let Some(tags) = tags_all {
            opts.tags_all = Some(tags);
        }
        if let Some(prefix) = task_id_prefix {
            opts.task_id_prefix = Some(prefix);
        }
        if let Some(ts) = time_after {
            opts.time_after = Some(ts);
        }
        if let Some(ts) = time_before {
            opts.time_before = Some(ts);
        }
        if let Some(s) = source {
            opts.source = Some(s);
        }
        if let Some(u) = user_id {
            opts.user_id = Some(u);
        }
        let results = self
            .db
            .query_similar_with_options(&state_embedding, opts)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        results_to_py(py, results)
    }

    fn save_to_file(&self, path: &str) -> PyResult<()> {
        self.db
            .save_to_file(Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    #[classmethod]
    fn load_from_file(_cls: &PyType, path: &str) -> PyResult<Self> {
        let db = RustAgentMemDB::load_from_file(Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(AgentMemDB { db })
    }

    /// Prune episodes with timestamp older than cutoff (Unix ms). Episodes without timestamp are kept.
    fn prune_older_than(&mut self, timestamp_cutoff_ms: i64) -> usize {
        self.db.prune_older_than(timestamp_cutoff_ms)
    }

    /// Prune to keep only the n most recent episodes (by timestamp). Episodes without timestamp are treated as oldest.
    fn prune_keep_newest(&mut self, n: usize) -> usize {
        self.db.prune_keep_newest(n)
    }

    /// Prune to keep only the n episodes with highest reward.
    fn prune_keep_highest_reward(&mut self, n: usize) -> usize {
        self.db.prune_keep_highest_reward(n)
    }
}

/// Disk-backed agent memory DB. Episodes stored in append-only log; index in RAM.
#[pyclass]
pub struct AgentMemDBDisk {
    db: RustAgentMemDBDisk,
}

#[pymethods]
impl AgentMemDBDisk {
    /// Open or create a disk-backed DB at the given directory.
    #[classmethod]
    fn open(_cls: &PyType, path: &str, dim: usize) -> PyResult<Self> {
        let db = RustAgentMemDBDisk::open(Path::new(path), dim)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(AgentMemDBDisk { db })
    }

    /// Open with exact index and checkpoint enabled for fast restart. Call checkpoint() after stores.
    #[classmethod]
    fn open_exact_with_checkpoint(_cls: &PyType, path: &str, dim: usize) -> PyResult<Self> {
        let db = RustAgentMemDBDisk::open_with_options(
            Path::new(path),
            DiskOptions::exact_with_checkpoint(dim),
        )
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(AgentMemDBDisk { db })
    }

    /// Persist checkpoint for fast restart (ExactIndex only). No-op for HNSW.
    fn checkpoint(&mut self) -> PyResult<()> {
        self.db
            .checkpoint()
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    fn store_episode(&mut self, py: Python, episode: &Episode) -> PyResult<()> {
        let mut rust_ep = RustEpisode::new(
            episode.task_id.clone(),
            episode.state_embedding.clone(),
            episode.reward,
        );
        if let Some(meta) = &episode.metadata {
            let meta_val = pyobj_to_json(py, meta.as_ref(py))?;
            rust_ep.metadata = meta_val;
        }
        rust_ep.timestamp = episode.timestamp;
        rust_ep.tags = episode.tags.clone();
        rust_ep.source = episode.source.clone();
        rust_ep.user_id = episode.user_id.clone();
        self.db
            .store_episode(rust_ep)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (state_embedding, min_reward, top_k, tags_any=None, tags_all=None, task_id_prefix=None, time_after=None, time_before=None, source=None, user_id=None))]
    fn query_similar(
        &self,
        py: Python,
        state_embedding: Vec<f32>,
        min_reward: f32,
        top_k: usize,
        tags_any: Option<Vec<String>>,
        tags_all: Option<Vec<String>>,
        task_id_prefix: Option<String>,
        time_after: Option<i64>,
        time_before: Option<i64>,
        source: Option<String>,
        user_id: Option<String>,
    ) -> PyResult<Vec<Episode>> {
        let mut opts = QueryOptions::new(min_reward, top_k);
        if let Some(tags) = tags_any {
            opts.tags_any = Some(tags);
        }
        if let Some(tags) = tags_all {
            opts.tags_all = Some(tags);
        }
        if let Some(prefix) = task_id_prefix {
            opts.task_id_prefix = Some(prefix);
        }
        if let Some(ts) = time_after {
            opts.time_after = Some(ts);
        }
        if let Some(ts) = time_before {
            opts.time_before = Some(ts);
        }
        if let Some(s) = source {
            opts.source = Some(s);
        }
        if let Some(u) = user_id {
            opts.user_id = Some(u);
        }
        let results = self
            .db
            .query_similar_with_options(&state_embedding, opts)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        results_to_py(py, results)
    }

    /// Prune episodes with timestamp older than cutoff (Unix ms). Episodes without timestamp are kept. Compacts the log.
    fn prune_older_than(&mut self, timestamp_cutoff_ms: i64) -> PyResult<usize> {
        self.db
            .prune_older_than(timestamp_cutoff_ms)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Prune to keep only the n most recent episodes (by timestamp). Compacts the log.
    fn prune_keep_newest(&mut self, n: usize) -> PyResult<usize> {
        self.db
            .prune_keep_newest(n)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }

    /// Prune to keep only the n episodes with highest reward. Compacts the log.
    fn prune_keep_highest_reward(&mut self, n: usize) -> PyResult<usize> {
        self.db
            .prune_keep_highest_reward(n)
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }
}

#[pymodule]
fn agent_mem_db_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AgentMemDB>()?;
    m.add_class::<AgentMemDBDisk>()?;
    m.add_class::<Episode>()?;
    Ok(())
}
