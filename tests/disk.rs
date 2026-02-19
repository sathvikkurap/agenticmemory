use agent_mem_db::{AgentMemDBDisk, DiskOptions, Episode};
use serde_json::json;
use std::fs;
use uuid::Uuid;

fn make_episode(dim: usize, reward: f32) -> Episode {
    Episode {
        id: Uuid::new_v4(),
        task_id: "test_task".to_string(),
        state_embedding: vec![0.1; dim],
        reward,
        metadata: json!({}),
        steps: None,
        timestamp: None,
        tags: None,
        source: None,
        user_id: None,
    }
}

#[test]
fn test_disk_open_create_store_query() {
    let dir = std::env::temp_dir().join("agent_mem_db_disk_test");
    let _ = fs::remove_dir_all(&dir);
    let dim = 8;

    let mut db = AgentMemDBDisk::open(&dir, dim).unwrap();
    let ep = make_episode(dim, 0.7);
    db.store_episode(ep.clone()).unwrap();

    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.5, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, ep.id);
}

#[test]
fn test_disk_reload_persists() {
    let dir = std::env::temp_dir().join("agent_mem_db_disk_reload_test");
    let _ = fs::remove_dir_all(&dir);
    let dim = 8;

    {
        let mut db = AgentMemDBDisk::open(&dir, dim).unwrap();
        let ep = make_episode(dim, 0.9);
        db.store_episode(ep).unwrap();
    }

    let db2 = AgentMemDBDisk::open(&dir, dim).unwrap();
    let results = db2.query_similar(&vec![0.1; dim], 0.5, 5).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].reward >= 0.5);
}

#[test]
fn test_disk_prune_older_than() {
    let dir = std::env::temp_dir().join("agent_mem_db_disk_prune_test");
    let _ = fs::remove_dir_all(&dir);
    let dim = 8;

    {
        let mut db = AgentMemDBDisk::open(&dir, dim).unwrap();
        db.store_episode(Episode::with_timestamp("old", vec![0.1; dim], 0.9, 1000))
            .unwrap();
        db.store_episode(Episode::with_timestamp("new", vec![0.1; dim], 0.8, 3000))
            .unwrap();
        let removed = db.prune_older_than(2000).unwrap();
        assert_eq!(removed, 1);
    }

    let db2 = AgentMemDBDisk::open(&dir, dim).unwrap();
    let results = db2.query_similar(&vec![0.1; dim], 0.0, 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].task_id, "new");
}

#[test]
fn test_disk_prune_keep_newest() {
    let dir = std::env::temp_dir().join("agent_mem_db_disk_prune_newest_test");
    let _ = fs::remove_dir_all(&dir);
    let dim = 8;

    {
        let mut db = AgentMemDBDisk::open(&dir, dim).unwrap();
        db.store_episode(Episode::with_timestamp("a", vec![0.1; dim], 0.9, 1000))
            .unwrap();
        db.store_episode(Episode::with_timestamp("b", vec![0.1; dim], 0.8, 2000))
            .unwrap();
        db.store_episode(Episode::with_timestamp("c", vec![0.1; dim], 0.7, 3000))
            .unwrap();
        let removed = db.prune_keep_newest(2).unwrap();
        assert_eq!(removed, 1);
    }

    let db2 = AgentMemDBDisk::open(&dir, dim).unwrap();
    let results = db2.query_similar(&vec![0.1; dim], 0.0, 5).unwrap();
    assert_eq!(results.len(), 2);
    let task_ids: Vec<&str> = results.iter().map(|e| e.task_id.as_str()).collect();
    assert!(task_ids.contains(&"b"));
    assert!(task_ids.contains(&"c"));
    assert!(!task_ids.contains(&"a"));
}

#[test]
fn test_disk_prune_keep_highest_reward() {
    let dir = std::env::temp_dir().join("agent_mem_db_disk_prune_reward_test");
    let _ = fs::remove_dir_all(&dir);
    let dim = 8;

    {
        let mut db = AgentMemDBDisk::open(&dir, dim).unwrap();
        db.store_episode(Episode::new("a", vec![0.1; dim], 0.3))
            .unwrap();
        db.store_episode(Episode::new("b", vec![0.1; dim], 0.9))
            .unwrap();
        db.store_episode(Episode::new("c", vec![0.1; dim], 0.5))
            .unwrap();
        let removed = db.prune_keep_highest_reward(2).unwrap();
        assert_eq!(removed, 1);
    }

    let db2 = AgentMemDBDisk::open(&dir, dim).unwrap();
    let results = db2.query_similar(&vec![0.1; dim], 0.0, 5).unwrap();
    assert_eq!(results.len(), 2);
    let rewards: Vec<f32> = results.iter().map(|e| e.reward).collect();
    assert!(rewards.contains(&0.9));
    assert!(rewards.contains(&0.5));
    assert!(!rewards.contains(&0.3));
}

#[test]
fn test_disk_checkpoint_fast_restart() {
    let dir = std::env::temp_dir().join("agent_mem_db_disk_checkpoint_test");
    let _ = fs::remove_dir_all(&dir);
    let dim = 8;

    {
        let mut db =
            AgentMemDBDisk::open_with_options(&dir, DiskOptions::exact_with_checkpoint(dim))
                .unwrap();
        db.store_episode(make_episode(dim, 0.7)).unwrap();
        db.store_episode(make_episode(dim, 0.8)).unwrap();
        db.checkpoint().unwrap();
    }

    let db2 =
        AgentMemDBDisk::open_with_options(&dir, DiskOptions::exact_with_checkpoint(dim)).unwrap();
    let results = db2.query_similar(&vec![0.1; dim], 0.5, 5).unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|e| e.reward >= 0.5));

    assert!(dir.join("exact_checkpoint.json").exists());
    assert!(dir.join("meta.json").exists());
}
