#!/usr/bin/env python3
"""Aggregate raw per-operation measurements and make presentation-ready figures."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
rows = []
for path in sorted(root.glob('*/*.csv')):
    with path.open() as f:
        data = list(csv.DictReader(f))
    if not data: continue
    lat = np.array([float(x['cuda_ms']) for x in data])
    wall = np.array([float(x['wall_ms']) for x in data])
    duration = (float(data[-1]['start_unix_s']) - float(data[0]['start_unix_s'])) + wall[-1]/1000
    rows.append({'run': path.parent.name, 'workload': path.stem, 'n':len(lat), 'duration_s':duration,
                 'throughput_ops_s':len(lat)/duration, 'mean_cuda_ms':lat.mean(), 'p95_cuda_ms':np.percentile(lat,95), 'mean_wall_ms':wall.mean()})
out = root/'summary.csv'; out.parent.mkdir(exist_ok=True)
with out.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ['run']); w.writeheader(); w.writerows(rows)
if not rows: raise SystemExit('No completed CSV result was found.')
figdir=root/'figures'; figdir.mkdir(exist_ok=True)
labels=[f"{x['run']}\n{x['workload']}" for x in rows]
for field,title,ylabel,name in [('throughput_ops_s','Measured throughput','Operations / second','throughput'),('mean_cuda_ms','Mean GPU latency','CUDA event time (ms)','mean_latency'),('p95_cuda_ms','Tail GPU latency (p95)','CUDA event time (ms)','p95_latency')]:
    plt.figure(figsize=(max(7,len(rows)*1.15),4.5)); plt.bar(labels,[x[field] for x in rows],color='#376b9b'); plt.title(title); plt.ylabel(ylabel); plt.xticks(rotation=25,ha='right'); plt.grid(axis='y',alpha=.25); plt.tight_layout(); plt.savefig(figdir/f'{name}.png',dpi=200); plt.close()
print(f'Wrote {out} and {figdir}')
