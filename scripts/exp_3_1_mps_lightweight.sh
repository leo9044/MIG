#!/usr/bin/env bash
set -Eeuo pipefail
RUN_SECONDS="${RUN_SECONDS:-20}"
MATRIX_SIZE="${MATRIX_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-50}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$DIR/.." && pwd)"
OUT="$ROOT/results/$(date +%Y%m%d_%H%M%S)_mps_lightweight"
mkdir -p "$OUT"
UUID_1G="$(nvidia-smi -L | awk -F'UUID: |)' '/MIG 1g\.0gb/ {print $2; exit}')"
[[ -n "$UUID_1G" ]] || { echo 'A 1g MIG instance is required.' >&2; exit 1; }

run_condition() {
  local workers="$1" mode="$2" index="$3" run="$OUT/workers${1}_${2}_rep${3}"
  mkdir -p "$run"
  echo "[MPS follow-up] workers=$workers MPS=$mode repetition=$index"
  sudo tegrastats --interval 100 --logfile "$run/tegrastats.log" --start
  local start_epoch="$(python3 -c 'import time; print(time.time()+6)')"
  sudo docker run --rm --runtime nvidia --ipc=host --cpuset-cpus=2-11 \
    -v "$DIR:/work:ro" -v "$run:/results" -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="$UUID_1G" \
    --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm "$IMAGE" bash -lc '
      set -Eeuo pipefail
      if [ "'$mode'" = on ]; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 2; fi
      pids=(); for i in $(seq 1 '"$workers"'); do
        python3 -u /work/mps_microbatch_worker.py --label worker_$i --csv /results/worker_$i.csv --summary /results/worker_$i.json --start-epoch '"$start_epoch"' --run-seconds '"$RUN_SECONDS"' --matrix-size '"$MATRIX_SIZE"' --batch-size '"$BATCH_SIZE"' & pids+=("$!")
      done
      for p in "${pids[@]}"; do wait "$p"; done
      if [ "'$mode'" = on ]; then echo quit | nvidia-cuda-mps-control; fi'
  sudo tegrastats --stop >/dev/null 2>&1 || true
  sleep 10
}

for workers in 4 8; do
  run_condition "$workers" off 1; run_condition "$workers" on 1
  run_condition "$workers" on 2; run_condition "$workers" off 2
  run_condition "$workers" off 3; run_condition "$workers" on 3
done
printf 'Raw results written to %s\n' "$OUT"
