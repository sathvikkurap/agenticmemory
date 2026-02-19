use agent_mem_db::{AgentMemDB, AgentMemError, Episode, QueryOptions};
use serde_json::json;
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
fn test_store_and_query() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    let ep = make_episode(dim, 0.7);
    db.store_episode(ep.clone()).unwrap();
    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.5, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, ep.id);
}

#[test]
fn test_min_reward_filter() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    let ep1 = make_episode(dim, 0.2);
    let ep2 = make_episode(dim, 0.8);
    db.store_episode(ep1.clone()).unwrap();
    db.store_episode(ep2.clone()).unwrap();
    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.5, 2).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, ep2.id);
}

#[test]
fn test_exact_backend() {
    let dim = 8;
    let mut db = AgentMemDB::new_exact(dim);
    let ep1 = make_episode(dim, 0.9);
    let ep2 = make_episode(dim, 0.3);
    db.store_episode(ep1.clone()).unwrap();
    db.store_episode(ep2.clone()).unwrap();
    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.5, 2).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, ep1.id);
}

#[test]
fn test_query_filters_tags_and_time() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    let ep1 = Episode::with_tags("t1", vec![0.1; dim], 0.9, vec!["coding".into()]);
    let ep2 = Episode::with_tags("t2", vec![0.1; dim], 0.8, vec!["support".into()]);
    let ep3 = Episode::with_timestamp("t3", vec![0.1; dim], 0.7, 1000);
    let ep4 = Episode::with_timestamp("t4", vec![0.1; dim], 0.6, 2000);
    db.store_episode(ep1).unwrap();
    db.store_episode(ep2).unwrap();
    db.store_episode(ep3).unwrap();
    db.store_episode(ep4).unwrap();

    let query = vec![0.1; dim];
    let opts = QueryOptions::new(0.0, 5).tags_any(vec!["coding".into()]);
    let results = db.query_similar_with_options(&query, opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].tags.as_deref(), Some(&["coding".to_string()][..]));

    let opts = QueryOptions::new(0.0, 5).time_after(1500).time_before(2500);
    let results = db.query_similar_with_options(&query, opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].timestamp, Some(2000));

    // tags_all: episode must have all tags
    let ep5 = Episode::with_tags("t5", vec![0.1; dim], 0.85, vec!["coding".into(), "python".into()]);
    db.store_episode(ep5).unwrap();
    let opts = QueryOptions::new(0.0, 5).tags_all(vec!["coding".into(), "python".into()]);
    let results = db.query_similar_with_options(&query, opts).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].tags.as_ref().unwrap().contains(&"coding".to_string()));
    assert!(results[0].tags.as_ref().unwrap().contains(&"python".to_string()));

    // task_id_prefix
    let opts = QueryOptions::new(0.0, 5).task_id_prefix("t".to_string());
    let results = db.query_similar_with_options(&query, opts).unwrap();
    assert_eq!(results.len(), 5);
    let opts = QueryOptions::new(0.0, 5).task_id_prefix("t4".to_string());
    let results = db.query_similar_with_options(&query, opts).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].task_id, "t4");
}

#[test]
fn test_recency_tie_breaker() {
    // When two episodes have identical embeddings (same distance), prefer the more recent one.
    let dim = 8;
    let emb = vec![0.1; dim];
    let mut db = AgentMemDB::new_exact(dim);
    let ep_old = Episode::with_timestamp("old", emb.clone(), 0.9, 1000);
    let ep_new = Episode::with_timestamp("new", emb.clone(), 0.9, 2000);
    db.store_episode(ep_old).unwrap();
    db.store_episode(ep_new).unwrap();
    let results = db.query_similar_with_options(&emb, QueryOptions::new(0.0, 2)).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].task_id, "new");
    assert_eq!(results[1].task_id, "old");
}

#[test]
fn test_prune_older_than() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    let ep_old = Episode::with_timestamp("old", vec![0.1; dim], 0.9, 1000);
    let ep_new = Episode::with_timestamp("new", vec![0.1; dim], 0.8, 3000);
    let ep_no_ts = make_episode(dim, 0.7);
    db.store_episode(ep_old).unwrap();
    db.store_episode(ep_new).unwrap();
    db.store_episode(ep_no_ts).unwrap();

    let removed = db.prune_older_than(2000);
    assert_eq!(removed, 1);
    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.0, 5).unwrap();
    assert_eq!(results.len(), 2);
    let task_ids: Vec<&str> = results.iter().map(|e| e.task_id.as_str()).collect();
    assert!(task_ids.contains(&"new"));
    assert!(task_ids.contains(&"test_task"));
    assert!(!task_ids.contains(&"old"));
}

#[test]
fn test_prune_keep_newest() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    db.store_episode(Episode::with_timestamp("a", vec![0.1; dim], 0.9, 1000))
        .unwrap();
    db.store_episode(Episode::with_timestamp("b", vec![0.1; dim], 0.8, 2000))
        .unwrap();
    db.store_episode(Episode::with_timestamp("c", vec![0.1; dim], 0.7, 3000))
        .unwrap();
    db.store_episode(Episode::with_timestamp("d", vec![0.1; dim], 0.6, 4000))
        .unwrap();

    let removed = db.prune_keep_newest(2);
    assert_eq!(removed, 2);
    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.0, 5).unwrap();
    assert_eq!(results.len(), 2);
    let task_ids: Vec<&str> = results.iter().map(|e| e.task_id.as_str()).collect();
    assert!(task_ids.contains(&"c"));
    assert!(task_ids.contains(&"d"));
    assert!(!task_ids.contains(&"a"));
    assert!(!task_ids.contains(&"b"));
}

#[test]
fn test_prune_keep_highest_reward() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    db.store_episode(make_episode(dim, 0.3)).unwrap();
    db.store_episode(make_episode(dim, 0.9)).unwrap();
    db.store_episode(make_episode(dim, 0.5)).unwrap();

    let removed = db.prune_keep_highest_reward(2);
    assert_eq!(removed, 1);
    let query = vec![0.1; dim];
    let results = db.query_similar(&query, 0.0, 5).unwrap();
    assert_eq!(results.len(), 2);
    let rewards: Vec<f32> = results.iter().map(|e| e.reward).collect();
    assert!(rewards.contains(&0.9));
    assert!(rewards.contains(&0.5));
    assert!(!rewards.contains(&0.3));
}

#[test]
fn test_dimension_mismatch() {
    let dim = 8;
    let mut db = AgentMemDB::new(dim);
    let ep = make_episode(dim + 1, 0.5);
    let err = db.store_episode(ep).unwrap_err();
    match err {
        AgentMemError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, dim);
            assert_eq!(got, dim + 1);
        }
        _ => panic!("Expected DimensionMismatch error"),
    }
    let query = vec![0.0; dim + 1];
    let err = db.query_similar(&query, 0.0, 1).unwrap_err();
    match err {
        AgentMemError::DimensionMismatch { expected, got } => {
            assert_eq!(expected, dim);
            assert_eq!(got, dim + 1);
        }
        _ => panic!("Expected DimensionMismatch error"),
    }
}
