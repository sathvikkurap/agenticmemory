#!/usr/bin/env python3
"""
Run A/B evaluation: no_memory, naive, agent_mem_db.
Run from agent_mem_db/python/: python -m examples.agents.run_eval
Optional: python -m examples.agents.run_eval --config eval_config.json

Produces stdout summary and optionally writes JSON report to eval_results.json.
"""

import argparse
import json
import sys
from pathlib import Path

from .base import make_tasks, EvalResult
from .no_memory import run_task as run_no_memory
from .naive_memory import run_task as run_naive
from .agent_mem_db_agent import run_task as run_agent_mem_db

VARIANT_REGISTRY = {
    "no_memory": run_no_memory,
    "naive": run_naive,
    "agent_mem_db": run_agent_mem_db,
}

DEFAULT_CONFIG = {
    "variants": ["no_memory", "naive", "agent_mem_db"],
    "num_tasks": 40,
}


def load_config(path: Path) -> dict:
    """Load config from JSON or YAML file."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError("YAML config requires PyYAML: pip install pyyaml")
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Run agent memory evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to eval config (JSON or YAML). Default: built-in config.",
    )
    args = parser.parse_args()

    if args.config and args.config.exists():
        config = load_config(args.config)
    else:
        config = DEFAULT_CONFIG

    variants_names = config.get("variants", DEFAULT_CONFIG["variants"])
    num_tasks = config.get("num_tasks", DEFAULT_CONFIG["num_tasks"])

    for name in variants_names:
        if name not in VARIANT_REGISTRY:
            print(f"Unknown variant: {name}. Available: {list(VARIANT_REGISTRY)}")
            return 1

    variants = [(name, VARIANT_REGISTRY[name]) for name in variants_names]
    tasks = make_tasks(num_tasks)
    results: list[EvalResult] = []
    for _name, run_fn in variants:
        for task in tasks:
            r = run_fn(task)
            results.append(r)

    by_variant: dict[str, list[EvalResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant, []).append(r)

    def _percentile(sorted_vals: list[float], p: float) -> float:
        """Compute percentile (0-100). Uses linear interpolation."""
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_vals) else f
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    # Overall stats
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print()
    print("Setup          | Success | Tokens/turn | Latency/turn (ms)")
    print("-" * 60)
    report = {}
    for name, run_fn in variants:
        rs = by_variant[name]
        success_rate = sum(1 for r in rs if r.success) / len(rs) * 100
        avg_tokens = sum(r.total_tokens for r in rs) / len(rs)
        avg_latency = sum(r.total_latency_ms for r in rs) / len(rs)
        avg_turns = sum(r.num_turns for r in rs) / len(rs)
        tokens_per_turn = avg_tokens / avg_turns if avg_turns else 0
        latency_per_turn = avg_latency / avg_turns if avg_turns else 0
        print(f"{name:14} | {success_rate:6.1f}% | {tokens_per_turn:11.0f} | {latency_per_turn:17.2f}")
        # Latency percentiles (per-task total latency)
        latencies = sorted(r.total_latency_ms for r in rs)
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        report[name] = {
            "success_rate": round(success_rate, 1),
            "avg_tokens": round(avg_tokens, 0),
            "avg_latency_ms": round(avg_latency, 2),
            "tokens_per_turn": round(tokens_per_turn, 0),
            "latency_per_turn_ms": round(latency_per_turn, 2),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
        }

    # Per task-type stats
    short_results = {v: [r for r in rs if r.task_type == "short"] for v, rs in by_variant.items()}
    long_results = {v: [r for r in rs if r.task_type == "long"] for v, rs in by_variant.items()}

    # Latency percentiles
    print()
    print("Latency percentiles (per-task total ms):")
    print("Setup          | p50     | p95     | p99")
    print("-" * 50)
    for name in [v[0] for v in variants]:
        r = report[name]
        print(f"{name:14} | {r['latency_p50_ms']:7.2f} | {r['latency_p95_ms']:7.2f} | {r['latency_p99_ms']:7.2f}")

    print()
    print("By task type:")
    print("Setup          | Short tasks success | Long tasks success")
    print("-" * 60)
    for name in [v[0] for v in variants]:
        short_rs = short_results[name]
        long_rs = long_results[name]
        short_sr = (sum(1 for r in short_rs if r.success) / len(short_rs) * 100) if short_rs else 0
        long_sr = (sum(1 for r in long_rs if r.success) / len(long_rs) * 100) if long_rs else 0
        print(f"{name:14} | {short_sr:18.1f}% | {long_sr:18.1f}%")
        report[name]["short_success"] = round(short_sr, 1)
        report[name]["long_success"] = round(long_sr, 1)

    # Failure cases: naive succeeds, agent_mem_db fails (if both variants run)
    if "naive" in by_variant and "agent_mem_db" in by_variant:
        naive_by_task = {r.task_id: r for r in by_variant["naive"]}
        amdb_by_task = {r.task_id: r for r in by_variant["agent_mem_db"]}
        failures = []
        for tid in naive_by_task:
            n = naive_by_task[tid]
            a = amdb_by_task.get(tid)
            if a and n.success and not a.success:
                failures.append({"task_id": tid, "task_type": n.task_type})
        if failures:
            print()
            print(f"Cases where naive succeeds but agent_mem_db fails: {len(failures)}")
            for f in failures[:5]:
                print(f"  - {f['task_id']} ({f['task_type']})")
        report["failure_cases_naive_wins"] = len(failures)

    # Write JSON to agent_mem_db root
    out_path = Path(__file__).resolve().parent.parent.parent.parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(f"Report written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
