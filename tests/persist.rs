use agent_mem_db::{AgentMemDB, Episode};
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

fn make_episode(dim: usize, reward: f32) -> Episode {
    Episode {
        id: Uuid::new_v4(),
        task_id: "persist_test".to_string(),
        state_embedding: vec![0.2; dim],
        reward,
        metadata: json!({"test": true}),
        steps: None,
        timestamp: None,
        tags: None,
        source: None,
        user_id: None,
    }
}

#[test]
fn test_save_and_load_roundtrip() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    let ep1 = make_episode(dim, 0.6);
    let ep2 = make_episode(dim, 0.9);
    db.store_episode(ep1.clone()).unwrap();
    db.store_episode(ep2.clone()).unwrap();
    let query = vec![0.2; dim];
    let orig_results = db.query_similar(&query, 0.5, 2).unwrap();
    let path = PathBuf::from("/tmp/agent_mem_db_test.bin");
    db.save_to_file(&path).unwrap();
    let db2 = AgentMemDB::load_from_file(&path).unwrap();
    let loaded_results = db2.query_similar(&query, 0.5, 2).unwrap();
    assert_eq!(orig_results.len(), loaded_results.len());
    let orig_ids: HashSet<_> = orig_results.iter().map(|ep| ep.id).collect();
    let loaded_ids: HashSet<_> = loaded_results.iter().map(|ep| ep.id).collect();
    assert_eq!(orig_ids, loaded_ids);
    fs::remove_file(&path).unwrap();
}

#[test]
fn test_load_missing_file() {
    let path = PathBuf::from("/tmp/agent_mem_db_missing.bin");
    let res = AgentMemDB::load_from_file(&path);
    assert!(res.is_err());
}

#[test]
fn test_load_corrupted_file() {
    let path = PathBuf::from("/tmp/agent_mem_db_corrupt.bin");
    std::fs::write(&path, b"not a valid bincode").unwrap();
    let res = AgentMemDB::load_from_file(&path);
    assert!(res.is_err());
    fs::remove_file(&path).unwrap();
}
