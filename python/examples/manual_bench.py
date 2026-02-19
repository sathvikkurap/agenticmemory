import time
import random
import statistics
import agent_mem_db_py as agent_mem_db

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def random_embedding(dim):
    return [random.random() for _ in range(dim)]

def bench_inserts(dim, n):
    db = agent_mem_db.AgentMemDB(dim)
    start = time.perf_counter()
    for i in range(n):
        ep = agent_mem_db.Episode(task_id=f"t{i}", state_embedding=random_embedding(dim), reward=random.random(), metadata={})
        db.store_episode(ep)
    elapsed = time.perf_counter() - start
    eps = n / elapsed if elapsed>0 else float('inf')
    print(f"insert,{dim},{n},{eps:.2f},{elapsed:.4f}")
    return eps

def bench_queries(dim, n, queries=100):
    db = agent_mem_db.AgentMemDB(dim)
    for i in range(n):
        ep = agent_mem_db.Episode(task_id=f"t{i}", state_embedding=random_embedding(dim), reward=random.random(), metadata={})
        db.store_episode(ep)
    qvecs = [random_embedding(dim) for _ in range(queries)]
    latencies = []
    for q in qvecs:
        t0 = time.perf_counter()
        _ = db.query_similar(q, -1.0, 5)
        latencies.append((time.perf_counter()-t0)*1000)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95*len(latencies))-1]
    avg = sum(latencies)/len(latencies)
    print(f"query,{dim},{n},{avg:.4f},{p50:.4f},{p95:.4f}")
    return avg, p50, p95

if __name__ == '__main__':
    dims = [256, 768]
    sizes = [1000, 10000]

    print("# Inserts throughput (eps/sec) and elapsed (s):")
    for dim in dims:
        for n in sizes:
            bench_inserts(dim, n)

    print("# Query latency (ms): avg, p50, p95")
    for dim in dims:
        for n in [1000, 10000]:
            bench_queries(dim, n, queries=50)
