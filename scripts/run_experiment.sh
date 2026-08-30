#!/usr/bin/env bash
# Usage: ./run_experiment.sh {nomig|mig|fault|uma-solo|uma|mps-off|mps-on} [seconds]
# The 1g workload is intentionally launched before every 2g workload.
set -Eeuo pipefail

MODE=${1:?"mode: nomig|mig|fault|uma-solo|uma|mps-off|mps-on"}
RUN_SECONDS=${2:-30}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/results/${STAMP}_${MODE}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
mkdir -p "$OUT"
cp "$ROOT/scripts/benchmark.py" "$OUT/"

require_mig() {
  UUID_1G="$(nvidia-smi -L | awk -F'UUID: |)' '/MIG 1g\.0gb/ {print $2; exit}')"
  UUID_2G="$(nvidia-smi -L | awk -F'UUID: |)' '/MIG 2g\.0gb/ {print $2; exit}')"
  [[ -n "$UUID_1G" && -n "$UUID_2G" ]] || { echo 'Missing 1g/2g instances. Run scripts/setup_mig.sh after reboot.' >&2; exit 1; }
}

start_tegra() { sudo tegrastats --interval 100 --logfile "$OUT/tegrastats.log" --start; }
stop_tegra() { sudo tegrastats --stop >/dev/null 2>&1 || true; }
trap stop_tegra EXIT INT TERM

run_1g() {
  sudo docker run --rm --runtime nvidia --ipc=host \
    -v "$ROOT/scripts:/work:ro" -v "$OUT:/results" \
    -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="$UUID_1G" \
    --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    "$IMAGE" python3 -u /work/benchmark.py --label "$1" --csv "/results/$1.csv" --summary "/results/$1.json" --duration "$RUN_SECONDS" --size 2000 "${@:2}"
}
run_2g() {
  sudo docker run --rm --runtime nvidia --ipc=host \
    -v "$ROOT/scripts:/work:ro" -v "$OUT:/results" \
    -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="$UUID_2G" \
    --device /dev/nvidia-caps/nvidia-cap12 --device /dev/nvidia-caps/nvidia-cap13 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    "$IMAGE" python3 -u /work/benchmark.py --label "$1" --csv "/results/$1.csv" --summary "/results/$1.json" --duration "$RUN_SECONDS" --size 2000 "${@:2}"
}

case "$MODE" in
  nomig)
    sudo nvidia-smi mig -dci || true; sudo nvidia-smi mig -dgi || true; sudo nvidia-smi -mig 0
    sleep 10
    start_tegra
    # Same GPU, two independent CUDA processes: shared-GPU comparison baseline.
    for label in nomig_a nomig_b; do
      sudo docker run --rm --runtime nvidia --ipc=host -v "$ROOT/scripts:/work:ro" -v "$OUT:/results" \
        -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0 --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
        "$IMAGE" python3 -u /work/benchmark.py --label "$label" --csv "/results/$label.csv" --summary "/results/$label.json" --duration "$RUN_SECONDS" --size 2000 &
    done; wait ;;
  mig)
    require_mig; start_tegra; run_1g mig_1g & p1=$!; sleep 2; run_2g mig_2g & p2=$!; wait "$p1" "$p2" ;;
  fault)
    # The delay ensures 2g is actively measured when the 1g allocation fails.
    require_mig; start_tegra; run_1g fault_oom --fault-oom --fault-delay 8 & p1=$!; sleep 2; run_2g fault_target & p2=$!; wait "$p1" "$p2" ;;
  uma-solo)
    require_mig; start_tegra; run_2g uma_2g_solo ;;
  uma)
    require_mig; start_tegra; run_1g uma_1g & p1=$!; sleep 2; run_2g uma_2g & p2=$!; wait "$p1" "$p2" ;;
  mps-off|mps-on)
    require_mig; start_tegra
    MPS="$([[ "$MODE" == mps-on ]] && echo on || echo off)"
    sudo docker run --rm --runtime nvidia --ipc=host --cpuset-cpus=2-11 \
      -v "$ROOT/scripts:/work:ro" -v "$OUT:/results" \
      -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="$UUID_1G" \
      --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22 \
      --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm "$IMAGE" bash -lc '
        set -e; if [ "'$MPS'" = on ]; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 2; fi
        python3 -u /work/benchmark.py --label worker_a --csv /results/worker_a.csv --summary /results/worker_a.json --duration "'$RUN_SECONDS'" --size 2000 & a=$!
        python3 -u /work/benchmark.py --label worker_b --csv /results/worker_b.csv --summary /results/worker_b.json --duration "'$RUN_SECONDS'" --size 2000 & b=$!
        wait "$a" "$b"; if [ "'$MPS'" = on ]; then echo quit | nvidia-cuda-mps-control; fi' ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac

stop_tegra; trap - EXIT INT TERM
nvidia-smi -L | tee "$OUT/nvidia-smi-L.txt"
printf 'Raw result directory: %s\n' "$OUT"
