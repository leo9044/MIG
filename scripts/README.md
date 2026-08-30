# MIG experiment execution

All reported performance numbers are derived from `benchmark.py` CSVs: one CUDA-event latency and one synchronized wall-clock time per matmul. Nsight traces are optional qualitative evidence only; CUDA API timing is never used as an application-performance result.

After a reboot, run `./setup_mig.sh` once. Each experiment now has its own entry script (the common runner is an implementation detail):

```bash
./exp_1_1_mig.sh 30         # 1-1: 1g first, then 2g
./exp_1_3_fault.sh 30       # 1-3: controlled 1g OOM / 2g continuity
./exp_1_2_shared_mem.sh 30    # 1-2: 2g alone, then 1g+2g contention
./exp_3_1_mps.sh 30         # 3-1: MPS off, then MPS on in 1g
./exp_1_1_nomig.sh 30       # 1-1 Non-MIG: run last; disables MIG
./analyze_results.py ../results
```

For 1-1's Non-MIG baseline, execute `./exp_1_1_nomig.sh 30` **after** the MIG runs; it disables MIG. Re-run `setup_mig.sh` before subsequent MIG tests. Each invocation produces a timestamped raw-result directory containing CSV, JSON metadata, `tegrastats.log`, and the observed MIG topology.
