use agent_mem_db::{AgentMemDB, Episode};
use rand::Rng;

fn main() {
    let dim = 768;
    let mut db = AgentMemDB::new(dim);
    let mut rng = rand::thread_rng();
    // Generate and store 1000 random episodes
    for _ in 0..1000 {
        let embedding: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
        let reward: f32 = rng.gen_range(-1.0..=1.0);
        let episode = Episode::new(
            format!("task_{}", rng.gen_range(0..10)),
            embedding,
            reward,
        );
        db.store_episode(episode).unwrap();
    }
    // Query with a random embedding and min_reward = 0.5
    let query_embedding: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
    let results = db.query_similar(&query_embedding, 0.5, 5).unwrap();
    println!("Top {} episodes with reward >= 0.5:", results.len());
    for ep in results {
        println!("id: {}, reward: {}", ep.id, ep.reward);
    }
}
