//! Property-based tests for save/load invariants.
//!
//! Invariant: For any DB state, after save and load, query_similar returns
//! the same set of episode IDs (order may differ due to HNSW approximation).

use agent_mem_db::{AgentMemDB, Episode};
use proptest::prelude::*;
use serde_json::Value;
use std::collections::HashSet;
use uuid::Uuid;

type ProptestInput = (usize, Vec<(Vec<f32>, f32)>, Vec<f32>, usize);

fn test_input_strategy() -> impl Strategy<Value = ProptestInput> {
    (2usize..=16, 1usize..=25).prop_flat_map(|(dim, num_episodes)| {
        let ep_strat = prop::collection::vec(
            (
                prop::collection::vec(-1.0f32..=1.0f32, dim),
                -1.0f32..=1.0f32,
            ),
            num_episodes,
        );
        let query_strat = prop::collection::vec(-1.0f32..=1.0f32, dim);
        let top_k_strat = 1usize..=(num_episodes.min(10));
        (ep_strat, Just(dim), query_strat, top_k_strat)
            .prop_map(|(episodes, dim, query, top_k)| (dim, episodes, query, top_k))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn save_load_preserves_query_results(input in test_input_strategy()) {
        let (dim, episodes, query, top_k) = input;

        let mut db = AgentMemDB::new_exact(dim);
        for (i, (emb, reward)) in episodes.into_iter().enumerate() {
            let ep = Episode {
                id: Uuid::new_v4(),
                task_id: format!("task_{}", i),
                state_embedding: emb,
                reward,
                metadata: Value::Null,
                steps: None,
                timestamp: None,
                tags: None,
                source: None,
                user_id: None,
            };
            db.store_episode(ep).unwrap();
        }

        let orig_results = db.query_similar(&query, -1.0, top_k).unwrap();
        let orig_ids: HashSet<_> = orig_results.iter().map(|ep| ep.id).collect();

        let path = std::env::temp_dir().join("agent_mem_db_proptest.json");
        db.save_to_file(&path).unwrap();
        let db2 = AgentMemDB::load_from_file_exact(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let loaded_results = db2.query_similar(&query, -1.0, top_k).unwrap();
        let loaded_ids: HashSet<_> = loaded_results.iter().map(|ep| ep.id).collect();

        assert_eq!(orig_ids, loaded_ids, "save/load must preserve query result set");
    }
}
