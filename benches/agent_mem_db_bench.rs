use agent_mem_db::{AgentMemDB, AgentMemDBDisk, DiskOptions, Episode};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use std::path::PathBuf;

pub fn bench_insert(c: &mut Criterion) {
    let dims = [256, 768];
    let sizes = [1_000, 10_000]; // 50k exceeds default HNSW max_elements (20k)
    for &dim in &dims {
        for &n in &sizes {
            let name = format!("insert_{}d_{}eps", dim, n);
            c.bench_function(&name, |b| {
                b.iter_batched(
                    || (AgentMemDB::new(dim), make_episodes(n, dim)),
                    |(mut db, episodes)| {
                        for ep in episodes {
                            db.store_episode(ep).unwrap();
                        }
                    },
                    BatchSize::PerIteration,
                )
            });
        }
    }
}

pub fn bench_query(c: &mut Criterion) {
    let dims = [256, 768];
    let sizes = [1_000, 10_000];
    let top_ks = [5, 10];
    for &dim in &dims {
        for &n in &sizes {
            let mut db = AgentMemDB::new(dim);
            let episodes = make_episodes(n, dim);
            db.store_episodes(episodes).unwrap();
            let queries: Vec<Vec<f32>> = (0..10).map(|_| random_embedding(dim)).collect();
            for &top_k in &top_ks {
                let name = format!("query_{}d_{}eps_topk{}", dim, n, top_k);
                c.bench_function(&name, |b| {
                    b.iter(|| {
                        for q in &queries {
                            let _ = db.query_similar(q, -1.0, top_k).unwrap();
                        }
                    })
                });
            }
        }
    }
}

pub fn bench_save(c: &mut Criterion) {
    let dim = 768;
    let n = 10_000;
    let mut db = AgentMemDB::new(dim);
    let episodes = make_episodes(n, dim);
    db.store_episodes(episodes).unwrap();
    let path: PathBuf = std::env::temp_dir().join("agent_mem_db_bench_save.json");
    c.bench_function(&format!("save_{}d_{}eps", dim, n), |b| {
        b.iter(|| {
            db.save_to_file(&path).unwrap();
        })
    });
}

pub fn bench_scale_insert(c: &mut Criterion) {
    let dim = 768;
    for &n in &[50_000, 100_000] {
        let episodes = make_episodes(n, dim);
        let name = format!("scale_insert_{}d_{}eps", dim, n);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let mut d = AgentMemDB::new_with_max_elements(dim, n + 1000);
                for ep in &episodes {
                    d.store_episode(ep.clone()).unwrap();
                }
            })
        });
    }
}

pub fn bench_exact_query(c: &mut Criterion) {
    let dim = 256;
    for &n in &[100, 1_000, 5_000] {
        let mut db = AgentMemDB::new_exact(dim);
        let episodes = make_episodes(n, dim);
        db.store_episodes(episodes).unwrap();
        let queries: Vec<Vec<f32>> = (0..5).map(|_| random_embedding(dim)).collect();
        let name = format!("exact_query_{}d_{}eps_topk10", dim, n);
        c.bench_function(&name, |b| {
            b.iter(|| {
                for q in &queries {
                    let _ = db.query_similar(q, -1.0, 10).unwrap();
                }
            })
        });
    }
}

pub fn bench_scale_query(c: &mut Criterion) {
    let dim = 768;
    for &n in &[50_000, 100_000] {
        let mut db = AgentMemDB::new_with_max_elements(dim, n + 1000);
        let episodes = make_episodes(n, dim);
        db.store_episodes(episodes).unwrap();
        let queries: Vec<Vec<f32>> = (0..5).map(|_| random_embedding(dim)).collect();
        let name = format!("scale_query_{}d_{}eps_topk10", dim, n);
        c.bench_function(&name, |b| {
            b.iter(|| {
                for q in &queries {
                    let _ = db.query_similar(q, -1.0, 10).unwrap();
                }
            })
        });
    }
}

pub fn bench_disk_open_replay_vs_checkpoint(c: &mut Criterion) {
    let dim = 64;
    let n = 5_000;
    let dir_replay: PathBuf = std::env::temp_dir().join("agent_mem_bench_replay");
    let dir_checkpoint: PathBuf = std::env::temp_dir().join("agent_mem_bench_checkpoint");

    // Prepare: create DB with episodes, write to disk
    let episodes = make_episodes(n, dim);
    let _ = std::fs::remove_dir_all(&dir_replay);
    let _ = std::fs::remove_dir_all(&dir_checkpoint);

    {
        let mut db =
            AgentMemDBDisk::open_with_options(&dir_replay, DiskOptions::exact_with_checkpoint(dim))
                .unwrap();
        for ep in &episodes {
            db.store_episode(ep.clone()).unwrap();
        }
    }
    {
        let mut db = AgentMemDBDisk::open_with_options(
            &dir_checkpoint,
            DiskOptions::exact_with_checkpoint(dim),
        )
        .unwrap();
        for ep in &episodes {
            db.store_episode(ep.clone()).unwrap();
        }
        db.checkpoint().unwrap();
    }

    let mut g = c.benchmark_group("disk_open");
    g.bench_function("open_replay_5k_eps", |b| {
        b.iter(|| {
            let _ = AgentMemDBDisk::open_with_options(
                &dir_replay,
                DiskOptions::exact_with_checkpoint(dim),
            );
        })
    });
    g.bench_function("open_from_checkpoint_5k_eps", |b| {
        b.iter(|| {
            let _ = AgentMemDBDisk::open_with_options(
                &dir_checkpoint,
                DiskOptions::exact_with_checkpoint(dim),
            );
        })
    });
    g.finish();

    let _ = std::fs::remove_dir_all(&dir_replay);
    let _ = std::fs::remove_dir_all(&dir_checkpoint);
}

pub fn bench_load(c: &mut Criterion) {
    let dim = 768;
    let n = 10_000;
    let mut db = AgentMemDB::new(dim);
    let episodes = make_episodes(n, dim);
    db.store_episodes(episodes).unwrap();
    let path: PathBuf = std::env::temp_dir().join("agent_mem_db_bench_load.json");
    db.save_to_file(&path).unwrap();
    drop(db);
    c.bench_function(&format!("load_{}d_{}eps", dim, n), |b| {
        b.iter(|| {
            let _ = AgentMemDB::load_from_file(&path).unwrap();
        })
    });
}

fn make_episodes(n: usize, dim: usize) -> Vec<Episode> {
    (0..n)
        .map(|i| {
            let embedding = random_embedding(dim);
            let reward = rand::thread_rng().gen_range(-1.0..=1.0);
            Episode::new(format!("task_{}", i), embedding, reward)
        })
        .collect()
}

fn random_embedding(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::random()).collect()
}

criterion_group!(
    benches,
    bench_insert,
    bench_query,
    bench_save,
    bench_load,
    bench_exact_query,
    bench_scale_insert,
    bench_scale_query,
    bench_disk_open_replay_vs_checkpoint
);
criterion_main!(benches);
