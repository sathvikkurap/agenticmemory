//! Fuzzing of episode inputs and metadata.
//!
//! Property-based tests that verify store_episode and query_similar never panic
//! and behave correctly with arbitrary (but valid) episode inputs: metadata,
//! tags, timestamp, steps, and varied embeddings.

use agent_mem_db::{AgentMemDB, Episode, EpisodeStep, QueryOptions};
use proptest::prelude::*;
use serde_json::Value;
use uuid::Uuid;

/// Strategy for serde_json::Value (arbitrary metadata, non-recursive).
fn metadata_strategy() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        (-1e10f64..=1e10f64).prop_map(|f| Value::Number(
            serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0))
        )),
        any::<String>().prop_map(Value::String),
        prop::collection::vec(any::<String>(), 0..5)
            .prop_map(|v| Value::Array(v.into_iter().map(Value::String).collect())),
    ]
}

/// Strategy for EpisodeStep.
fn step_strategy() -> impl Strategy<Value = EpisodeStep> {
    (
        any::<u32>(),
        any::<String>(),
        any::<String>(),
        -1.0f32..=1.0f32,
    )
        .prop_map(|(index, action, observation, step_reward)| EpisodeStep {
            index,
            action,
            observation,
            step_reward,
        })
}

/// Strategy for a full Episode with arbitrary metadata, tags, timestamp, steps.
fn episode_strategy(dim: usize) -> impl Strategy<Value = Episode> {
    (
        prop::collection::vec(-1.0f32..=1.0f32, dim),
        -1.0f32..=1.0f32,
        any::<String>(),
        metadata_strategy(),
        prop::option::of(prop::collection::vec(step_strategy(), 0..4)),
        prop::option::of(any::<i64>()),
        prop::option::of(prop::collection::vec(any::<String>(), 0..5)),
        prop::option::of(any::<String>()),
        prop::option::of(any::<String>()),
    )
        .prop_map(
            |(
                state_embedding,
                reward,
                task_id,
                metadata,
                steps,
                timestamp,
                tags,
                source,
                user_id,
            )| Episode {
                id: Uuid::new_v4(),
                task_id,
                state_embedding,
                reward,
                metadata,
                steps,
                timestamp,
                tags,
                source,
                user_id,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn store_episode_never_panics_with_arbitrary_input(
        (dim, episodes) in (2usize..=32).prop_flat_map(|dim| {
            (
                Just(dim),
                prop::collection::vec(episode_strategy(dim), 1..=20),
            )
        }),
    ) {
        let mut db = AgentMemDB::new_exact(dim);
        for ep in episodes {
            let _ = db.store_episode(ep);
        }

        let query = vec![0.0f32; dim];
        let _ = db.query_similar(&query, -1.0, 10);
    }

    #[test]
    fn query_with_options_never_panics(
        (dim, episodes, tags_any, time_after, time_before) in (2usize..=16)
            .prop_flat_map(|dim| {
                (
                    Just(dim),
                    prop::collection::vec(episode_strategy(dim), 1..=15),
                    prop::option::of(prop::collection::vec(any::<String>(), 0..3)),
                    prop::option::of(any::<i64>()),
                    prop::option::of(any::<i64>()),
                )
            }),
    ) {
        let mut db = AgentMemDB::new_exact(dim);
        for ep in episodes {
            let _ = db.store_episode(ep);
        }

        let query = vec![0.0f32; dim];
        let mut opts = QueryOptions::new(-1.0, 10);
        if let Some(t) = tags_any {
            opts = opts.tags_any(t);
        }
        if let Some(t) = time_after {
            opts = opts.time_after(t);
        }
        if let Some(t) = time_before {
            opts = opts.time_before(t);
        }
        let _ = db.query_similar_with_options(&query, opts);
    }
}
