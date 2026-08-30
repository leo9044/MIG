#!/usr/bin/env python3
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_latency(rows, output_dir):
    labels = [r["condition"] for r in rows]
    values = [float(r["latency_ms"]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.title("Experiment latency comparison")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Condition")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"), dpi=200)
    plt.close()


def plot_throughput(rows, output_dir):
    labels = [r["condition"] for r in rows]
    values = [float(r["throughput"]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["#8172B3", "#CCB974", "#64B5CD"])
    plt.title("Experiment throughput comparison")
    plt.ylabel("Throughput")
    plt.xlabel("Condition")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"), dpi=200)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_experiment_summary.py <csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    output_dir = csv_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    rows = load_rows(str(csv_path))
    if not rows:
        print("No rows found in CSV.")
        sys.exit(1)

    plot_latency(rows, str(output_dir))
    plot_throughput(rows, str(output_dir))
    print(f"Created figures in: {output_dir}")


if __name__ == "__main__":
    main()
