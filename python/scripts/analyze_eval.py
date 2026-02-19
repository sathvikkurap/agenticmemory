#!/usr/bin/env python3
"""
Analyze eval_results.json and produce summary tables and optional charts.

Usage:
  python scripts/analyze_eval.py [--input eval_results.json] [--output report.md] [--plot]

Run from agent_mem_db/python/ or agent_mem_db/.
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    """Load eval_results.json."""
    with open(path) as f:
        return json.load(f)


def format_table(headers: list[str], rows: list[list]) -> str:
    """Format a markdown table."""
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    header = "|" + "|".join(headers) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("|" + "|".join(str(c) for c in row) + "|")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze eval results")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to eval_results.json (default: repo root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib charts (requires matplotlib)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    in_path = args.input or root / "eval_results.json"
    if not in_path.exists():
        print(f"Error: {in_path} not found. Run 'make eval' first.", file=sys.stderr)
        return 1

    data = load_results(in_path)

    # Exclude non-variant keys
    variants = [k for k in data if k not in ("failure_cases_naive_wins",) and isinstance(data[k], dict)]
    if not variants:
        print("No variant data found in eval_results.json", file=sys.stderr)
        return 1

    lines = []
    lines.append("# Evaluation Analysis")
    lines.append("")
    lines.append(f"**Source:** {in_path.resolve()}")
    lines.append("")

    # Main results table
    lines.append("## Results")
    lines.append("")
    headers = ["Setup", "Success %", "Tokens/turn", "Latency/turn (ms)"]
    rows = []
    for v in variants:
        r = data[v]
        rows.append([
            v,
            f"{r.get('success_rate', 0):.1f}",
            int(r.get("tokens_per_turn", 0)),
            f"{r.get('latency_per_turn_ms', 0):.2f}",
        ])
    lines.append(format_table(headers, rows))
    lines.append("")

    # Latency percentiles
    lines.append("## Latency Percentiles (per-task ms)")
    lines.append("")
    headers = ["Setup", "p50", "p95", "p99"]
    rows = []
    for v in variants:
        r = data[v]
        rows.append([
            v,
            f"{r.get('latency_p50_ms', 0):.2f}",
            f"{r.get('latency_p95_ms', 0):.2f}",
            f"{r.get('latency_p99_ms', 0):.2f}",
        ])
    lines.append(format_table(headers, rows))
    lines.append("")

    # By task type
    if any("short_success" in data.get(v, {}) for v in variants):
        lines.append("## By Task Type")
        lines.append("")
        headers = ["Setup", "Short success %", "Long success %"]
        rows = []
        for v in variants:
            r = data[v]
            rows.append([
                v,
                f"{r.get('short_success', 0):.1f}",
                f"{r.get('long_success', 0):.1f}",
            ])
        lines.append(format_table(headers, rows))
        lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.output:
        args.output.write_text(report)
        print(f"\nReport written to {args.output}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
            return 1

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Success rate
        ax = axes[0]
        ax.bar(variants, [data[v].get("success_rate", 0) for v in variants], color=["#e74c3c", "#3498db", "#2ecc71"])
        ax.set_ylabel("Success %")
        ax.set_title("Success Rate")
        ax.set_ylim(0, 105)

        # Latency p95
        ax = axes[1]
        ax.bar(variants, [data[v].get("latency_p95_ms", 0) for v in variants], color=["#e74c3c", "#3498db", "#2ecc71"])
        ax.set_ylabel("ms")
        ax.set_title("Latency p95")

        # By task type (grouped)
        ax = axes[2]
        x = range(len(variants))
        w = 0.35
        short = [data[v].get("short_success", 0) for v in variants]
        long = [data[v].get("long_success", 0) for v in variants]
        ax.bar([i - w/2 for i in x], short, w, label="Short", color="#3498db")
        ax.bar([i + w/2 for i in x], long, w, label="Long", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylabel("Success %")
        ax.set_title("By Task Type")
        ax.legend()
        ax.set_ylim(0, 105)

        plt.tight_layout()
        out_plot = args.output.with_suffix(".png") if args.output else root / "eval_analysis.png"
        plt.savefig(out_plot, dpi=100)
        print(f"Chart saved to {out_plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
