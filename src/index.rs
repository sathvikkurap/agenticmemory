//! Pluggable vector index backends for episode similarity search.

use hnswx::{EuclideanDistance, HnswConfig, HNSW};

/// Euclidean L2 distance between two vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Exact (brute-force) vector index. O(n) per query; use for small episode sets or correctness-critical use.
#[derive(Default)]
pub struct ExactIndex {
    vectors: Vec<Vec<f32>>,
}

impl ExactIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }

    /// Create an ExactIndex from pre-existing vectors (e.g. loaded from checkpoint).
    /// Keys are 0..vectors.len().
    pub fn from_vectors(vectors: Vec<Vec<f32>>) -> Self {
        Self { vectors }
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Insert a vector; returns the internal key (index).
    pub fn insert(&mut self, vec: Vec<f32>) -> usize {
        let key = self.vectors.len();
        self.vectors.push(vec);
        key
    }

    /// Search for top-k nearest neighbors by L2 distance. Returns (key, distance) pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_distance(query, v)))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

/// HNSW approximate nearest-neighbor index. Fast for large episode sets.
pub struct HnswIndex {
    hnsw: HNSW<EuclideanDistance>,
}

impl HnswIndex {
    pub fn new(max_elements: usize) -> Self {
        let config = HnswConfig {
            max_elements,
            m: 16,
            m_max: 16,
            m_max_0: 16,
            ef_construction: 200,
            level_multiplier: 1.0 / (16.0f64.ln()),
            allow_replace_deleted: false,
            batch_size: 64,
            ef_search: 32,
            num_threads: 1,
        };
        Self {
            hnsw: HNSW::new(config, EuclideanDistance::new()),
        }
    }

    /// Insert a vector; returns the internal key.
    pub fn insert(&mut self, vec: &[f32]) -> usize {
        self.hnsw.insert(vec.to_vec())
    }

    /// Search for top-k nearest neighbors. Returns (key, distance) pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.hnsw
            .search_knn(query, k)
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect()
    }
}

/// Pluggable index backend. AgentMemDB uses this internally.
pub enum IndexBackend {
    Hnsw(Box<HnswIndex>),
    Exact(ExactIndex),
}

impl IndexBackend {
    pub fn insert(&mut self, vec: &[f32]) -> usize {
        match self {
            IndexBackend::Hnsw(idx) => idx.insert(vec),
            IndexBackend::Exact(idx) => idx.insert(vec.to_vec()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            IndexBackend::Hnsw(idx) => idx.hnsw.len(),
            IndexBackend::Exact(idx) => idx.len(),
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        match self {
            IndexBackend::Hnsw(idx) => idx.search(query, k),
            IndexBackend::Exact(idx) => idx.search(query, k),
        }
    }
}
