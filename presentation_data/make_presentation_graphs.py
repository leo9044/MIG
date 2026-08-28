#!/usr/bin/env python3
"""Create restrained presentation plots from the verified experiment metrics."""
from pathlib import Path
import csv

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "presentation_metrics.csv").open(encoding="utf-8")))

COLORS = {"1g": "#2563eb", "2g": "#0f766e", "MPS on": "#15803d", "MPS off": "#b45309", "stress": "#b91c1c", "nostress": "#2563eb"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.labelsize": 10, "figure.dpi": 150})


def save_bar(path, labels, values, colors, ylabel, title, note):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.2f}", ha="center", va="bottom", fontsize=9)
    fig.text(0.01, 0.01, note, fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ROOT / path, bbox_inches="tight")
    plt.close(fig)

mig = [row for row in rows if row["experiment"] == "MIG"]
save_bar(
    "01_mig_comparison.png",
    ["MIG 1g", "MIG 2g"],
    [float(row["avg_kernel_ms"]) for row in mig],
    [COLORS["1g"], COLORS["2g"]],
    "Average GEMM kernel time (ms)",
    "MIG instance size changes GEMM latency",
    "20 s capture; one run; same 2000x2000 GEMM; lower is better.",
)

fig, ax = plt.subplots(figsize=(8, 4.6))
mps = [row for row in rows if row["experiment"] == "MPS"]
labels = ["MPS on\nclient A", "MPS on\nclient B", "MPS off\naggregate"]
values = [float(row["throughput_instances_per_s"]) for row in mps]
bars = ax.bar(labels, values, color=[COLORS["MPS on"], COLORS["MPS on"], COLORS["MPS off"]], width=0.62)
ax.set_ylabel("GEMM kernels per second")
ax.set_title("MPS keeps the two 1g client results balanced", loc="left", weight="bold")
ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
ax.set_axisbelow(True)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.1f}", ha="center", va="bottom", fontsize=9)
fig.text(0.01, 0.01, "20 s capture; MPS on values are per-client reports, MPS off is one aggregate report; do not compare as identical units.", fontsize=8, color="#4b5563")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(ROOT / "02_mps_comparison.png", bbox_inches="tight")
plt.close(fig)

uma = [row for row in rows if row["experiment"] == "UMA"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
labels = ["No CPU\nstress", "CPU memory\nstress"]
colors = [COLORS["nostress"], COLORS["stress"]]
for ax, key, ylabel, title, fmt in [
    (axes[0], "throughput_instances_per_s", "GEMM kernels per second", "GPU throughput", "{:.1f}"),
    (axes[1], "dram_read_pct", "DRAM read throughput (%)", "Shared UMA traffic", "{:.2f}"),
]:
    values = [float(row[key]) for row in uma]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value), ha="center", va="bottom", fontsize=9)
fig.suptitle("CPU memory traffic affects a MIG 1g workload", x=0.06, ha="left", weight="bold", fontsize=13)
fig.text(0.01, 0.01, "20 s capture; one run; SoC Metrics t264; CPU stress uses four NumPy memory workers.", fontsize=8, color="#4b5563")
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
fig.savefig(ROOT / "03_uma_contention.png", bbox_inches="tight")
plt.close(fig)
