#!/usr/bin/env python3
import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("result_dir")
parser.add_argument("--out", type=Path)
args = parser.parse_args()
root = Path(args.result_dir)
out = args.out or root / "analysis"
out.mkdir(exist_ok=True)
groups = defaultdict(list)
for csv_path in root.glob("workers*_*/worker_*.csv"):
    run = csv_path.parent.name
    workers, mode, _ = run.split("_")
    data = list(csv.DictReader(csv_path.open()))
    timings = np.array([float(row["cuda_batch_ms"]) for row in data])
    operations = sum(int(row["operations"]) for row in data)
    stamps = [float(row["start_unix_s"]) for row in data]
    elapsed = stamps[-1] - stamps[0] + float(data[-1]["wall_ms"]) / 1000
    groups[(int(workers.removeprefix("workers")), mode, run)].append((operations / elapsed, timings))

summary = []
for (workers, mode, run), values in sorted(groups.items()):
    batch_times = np.concatenate([value[1] for value in values])
    summary.append({"workers": workers, "mode": mode, "run": run,
                    "throughput_ops_s": sum(value[0] for value in values),
                    "mean_batch_ms": float(batch_times.mean()),
                    "p95_batch_ms": float(np.percentile(batch_times, 95))})
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for workers in sorted({row["workers"] for row in summary}):
    labels, values = [], []
    for mode in ("off", "on"):
        measures = [row["throughput_ops_s"] for row in summary if row["workers"] == workers and row["mode"] == mode]
        labels.append(f"MPS {mode}")
        values.append(float(np.median(measures)))
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#777777", "#4e8f62"])
    plt.title(f"MPS lightweight: {workers} workers (median of 3)")
    plt.ylabel("aggregate matmul operations / second")
    plt.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(out / f"throughput_workers{workers}.png", dpi=200)
    plt.close()
print(out)
