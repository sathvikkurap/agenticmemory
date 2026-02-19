//! Concurrent read/write stress tests for AgentMemDB.
//!
//! Verifies that AgentMemDB behaves correctly when wrapped in Arc<RwLock<>> and
//! accessed concurrently by multiple writer and reader threads. No panics,
//! no data corruption, queries return valid results.

use agent_mem_db::{AgentMemDB, Episode};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

const DIM: usize = 16;
const WRITERS: usize = 4;
const READERS: usize = 8;
const OPS_PER_WRITER: usize = 200;
const OPS_PER_READER: usize = 500;

fn make_episode(seed: u64, reward: f32) -> Episode {
    let mut emb = vec![0.0f32; DIM];
    for (i, v) in emb.iter_mut().enumerate() {
        *v = ((seed as f32 * 0.1 + i as f32 * 0.01) % 1.0) - 0.5;
    }
    Episode {
        id: Uuid::new_v4(),
        task_id: format!("task_{}", seed),
        state_embedding: emb,
        reward,
        metadata: serde_json::Value::Null,
        steps: None,
        timestamp: None,
        tags: None,
        source: None,
        user_id: None,
    }
}

#[test]
fn test_concurrent_store_and_query() {
    // Use exact index for deterministic full-result verification; HNSW is approximate.
    let db = Arc::new(std::sync::RwLock::new(AgentMemDB::new_exact(DIM)));
    let write_count = Arc::new(AtomicU64::new(0));
    let read_count = Arc::new(AtomicU64::new(0));

    let mut writers = Vec::new();
    for w in 0..WRITERS {
        let db = Arc::clone(&db);
        let write_count = Arc::clone(&write_count);
        writers.push(thread::spawn(move || {
            for i in 0..OPS_PER_WRITER {
                let ep = make_episode(
                    (w * OPS_PER_WRITER + i) as u64,
                    0.5 + (i % 10) as f32 * 0.05,
                );
                let mut guard = db.write().unwrap();
                guard.store_episode(ep).unwrap();
                write_count.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }

    let mut readers = Vec::new();
    for r in 0..READERS {
        let db = Arc::clone(&db);
        let read_count = Arc::clone(&read_count);
        readers.push(thread::spawn(move || {
            let query: Vec<f32> = (0..DIM)
                .map(|i| (r as f32 * 0.02 + i as f32 * 0.01) % 1.0 - 0.5)
                .collect();
            for _ in 0..OPS_PER_READER {
                let guard = db.read().unwrap();
                let results = guard.query_similar(&query, 0.0, 5).unwrap();
                drop(guard);
                for ep in &results {
                    assert_eq!(ep.state_embedding.len(), DIM);
                    assert!(ep.reward >= 0.0);
                }
                read_count.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }

    for h in writers.into_iter().chain(readers) {
        h.join().unwrap();
    }

    assert_eq!(
        write_count.load(Ordering::SeqCst),
        (WRITERS * OPS_PER_WRITER) as u64
    );
    assert_eq!(
        read_count.load(Ordering::SeqCst),
        (READERS * OPS_PER_READER) as u64
    );

    let guard = db.read().unwrap();
    let total = guard
        .query_similar(&[0.0; DIM], -1.0, WRITERS * OPS_PER_WRITER + 1)
        .unwrap();
    assert_eq!(
        total.len(),
        WRITERS * OPS_PER_WRITER,
        "all stored episodes should be queryable (exact index)"
    );
}

#[test]
fn test_concurrent_mixed_ops() {
    let db = Arc::new(std::sync::RwLock::new(AgentMemDB::new_exact(DIM)));
    let n_ops = 100;

    let mut handles = Vec::new();
    for t in 0..8 {
        let db = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..n_ops {
                if t % 2 == 0 {
                    let ep = make_episode((t * n_ops + i) as u64, 0.7);
                    let mut guard = db.write().unwrap();
                    guard.store_episode(ep).unwrap();
                } else {
                    let query = vec![0.1f32; DIM];
                    let guard = db.read().unwrap();
                    let _ = guard.query_similar(&query, 0.0, 3).unwrap();
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let guard = db.read().unwrap();
    let results = guard.query_similar(&[0.0; DIM], 0.0, 1000).unwrap();
    assert_eq!(results.len(), 4 * n_ops, "4 writer threads * n_ops each");
}

/// Stress test with HNSW backend. Verifies no panics under concurrent load;
/// HNSW approximate search may return fewer than total episodes.
#[test]
fn test_concurrent_hnsw_no_panic() {
    let db = Arc::new(std::sync::RwLock::new(AgentMemDB::new(DIM)));
    let n_writes = 100;
    let n_reads = 200;

    let mut writers: Vec<_> = (0..4)
        .map(|w| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                for i in 0..n_writes {
                    let ep = make_episode((w * n_writes + i) as u64, 0.8);
                    let mut guard = db.write().unwrap();
                    guard.store_episode(ep).unwrap();
                }
            })
        })
        .collect();

    let readers: Vec<_> = (0..4)
        .map(|r| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                let query: Vec<f32> = (0..DIM)
                    .map(|i| (r as f32 * 0.1 + i as f32) * 0.01 % 1.0 - 0.5)
                    .collect();
                for _ in 0..n_reads {
                    let guard = db.read().unwrap();
                    let _ = guard.query_similar(&query, 0.0, 10).unwrap();
                }
            })
        })
        .collect();

    for h in writers.drain(..).chain(readers) {
        h.join().unwrap();
    }

    let guard = db.read().unwrap();
    let results = guard.query_similar(&[0.0; DIM], 0.0, 500).unwrap();
    assert!(!results.is_empty(), "should have some episodes");
    assert!(results.len() <= 4 * n_writes, "cannot exceed total stored");
}
