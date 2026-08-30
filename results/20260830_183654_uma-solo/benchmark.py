#!/usr/bin/env python3
"""Write one CSV row per CUDA operation; do not use profiler API-time as a metric."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--size", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--fault-oom", action="store_true")
    p.add_argument("--fault-delay", type=float, default=0.0,
                   help="seconds to wait before the intentional OOM")
    return p.parse_args()


def write_summary(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    a = args()
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    metadata = {
        "label": a.label, "started_unix_s": time.time(), "duration_s": a.duration,
        "matrix_size": a.size, "cuda_device": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pid": os.getpid(),
    }
    if a.fault_oom:
        # Intentional, recoverable CUDA out-of-memory test.  No driver reset or illegal kernel.
        if a.fault_delay:
            time.sleep(a.fault_delay)
        blocks = []
        try:
            while True:
                blocks.append(torch.empty((4096, 4096), dtype=torch.float32, device=device))
        except torch.OutOfMemoryError as e:
            metadata.update({"result": "expected_cuda_oom", "allocated_blocks": len(blocks), "error": str(e)})
            write_summary(a.summary, metadata)
            print(json.dumps(metadata), flush=True)
            return
        raise RuntimeError("OOM was not reached")

    x = torch.randn((a.size, a.size), device=device)
    y = torch.randn((a.size, a.size), device=device)
    for _ in range(a.warmup):
        torch.matmul(x, y)
    torch.cuda.synchronize()

    rows = []
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    deadline = time.monotonic() + a.duration
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=["label", "iteration", "start_unix_s", "wall_ms", "cuda_ms"])
        out.writeheader()
        i = 0
        while time.monotonic() < deadline:
            stamp = time.time()
            wall = time.perf_counter()
            start_evt.record()
            torch.matmul(x, y)
            end_evt.record()
            torch.cuda.synchronize()
            row = {"label": a.label, "iteration": i, "start_unix_s": f"{stamp:.6f}",
                   "wall_ms": f"{(time.perf_counter() - wall) * 1000:.6f}",
                   "cuda_ms": f"{start_evt.elapsed_time(end_evt):.6f}"}
            out.writerow(row)
            rows.append(row)
            i += 1
    gpu = [float(r["cuda_ms"]) for r in rows]
    metadata.update({"result": "completed", "iterations": len(rows), "ended_unix_s": time.time(),
                     "mean_cuda_ms": sum(gpu) / len(gpu) if gpu else None})
    write_summary(a.summary, metadata)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
