//! Simulated agent loop using agent_mem_db
//!
//! This minimal example demonstrates the core read/write flow against
//! `AgentMemDB` from Rust:
//! 1. Embed the current state.
//! 2. Query memory for similar past episodes to inform action selection.
//! 3. Take an action (simulated reward here).
//! 4. Store the resulting episode (optionally with `EpisodeStep`s).
//!
//! For a Python usage example (PyO3 bindings), see `python/examples/basic.py`.

use agent_mem_db::{AgentMemDB, Episode, EpisodeStep};
use rand::Rng;

fn main() {
    let dim = 16;
    // Construct an in-memory DB for 16-dim embeddings.
    // See AgentMemDB docs for constructor options (max size/parameters).
    let mut db = AgentMemDB::new(dim);
    let mut rng = rand::thread_rng();
    let num_tasks = 100;
    println!("Simulating {} agent tasks...", num_tasks);
    for i in 0..num_tasks {
        // --- Step 1: Embed the current state (synthetic for demo) ---
        let task_id = format!("task_{}", i);
        let embedding: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

        // --- Step 2: Query memory for similar past episodes (top K = 5) ---
        let similar = db.query_similar(&embedding, -1.0, 5).unwrap();
        let n_sim = similar.len();
        let (best_reward, best_steps) = if let Some(best) = similar
            .iter()
            .max_by(|a, b| a.reward.partial_cmp(&b.reward).unwrap())
        {
            (
                best.reward,
                best.steps.as_ref().map(|s| s.len()).unwrap_or(0),
            )
        } else {
            (f32::NAN, 0)
        };
        println!(
            "Task {}: found {} similar, best_reward={:.2}, best_steps={}",
            task_id, n_sim, best_reward, best_steps
        );

        // --- Step 3: Decide action and sample reward (simulate agent) ---
        let reward: f32 = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };

        // --- Step 4: Optionally log a trajectory (steps) for the first episode ---
        let mut episode = Episode::new(task_id.clone(), embedding, reward);
        if i == 0 {
            episode.steps = Some(vec![
                EpisodeStep {
                    index: 0,
                    action: "move_a".into(),
                    observation: "obs1".into(),
                    step_reward: 0.2,
                },
                EpisodeStep {
                    index: 1,
                    action: "move_b".into(),
                    observation: "obs2".into(),
                    step_reward: 0.3,
                },
                EpisodeStep {
                    index: 2,
                    action: "move_c".into(),
                    observation: "obs3".into(),
                    step_reward: 0.5,
                },
            ]);
            println!("  (First episode logs a trajectory of 3 steps)");
        }
        db.store_episode(episode).unwrap();
    }
    println!("\nDone. DB now contains {} episodes.", num_tasks);
}
