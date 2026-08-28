# Presentation Data

This folder contains derived presentation material. The original `.nsys-rep` files remain in `../nsys_results` and `../nsys_mps_results`.

## Generate graphs

```bash
cd /home/leo/MIG
python3 presentation_data/make_presentation_graphs.py
```

Outputs:

- `01_mig_comparison.png`: average GEMM kernel time for 1g vs 2g
- `02_mps_comparison.png`: MPS on per-client throughput and MPS off aggregate throughput
- `03_uma_contention.png`: GPU throughput and DRAM read throughput with/without CPU memory stress

## Interpretation

- Lower kernel time is better.
- Higher GEMM kernels/s is better.
- The MPS off result is an aggregate report, while MPS on has one report per client. The MPS chart is useful for showing balance and scale, but it is not a perfectly matched per-client comparison.
- All values are from one 20-second run. Use the plots as clear visual summaries, not statistical proof. Repeat each condition at least three times before making a strong performance claim.

## Data source

`presentation_metrics.csv` contains the verified values from the latest Toolkit 1.19.1 and raw MIG device-node experiments. UMA values come from the latest clean 20-second SoC Metrics `t264` collection.
