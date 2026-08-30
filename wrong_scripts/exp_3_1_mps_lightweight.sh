#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_DIR="${RESULT_DIR:-/home/leo/MIG/experiments/fresh_20260830/exp_3_1_light}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
mkdir -p "$RESULT_DIR"

echo "=== Fresh experiment 3-1 (Lightweight): MPS spatial sharing ==="

nvidia-smi -L | grep -q 'MIG 1g\.0gb' || {
  echo "MIG 1g instance missing; recreating it."
  sudo nvidia-smi -mig 1
  sleep 3
  sudo nvidia-smi mig -cgi 83,78 -C || true
  sleep 3
}

uuid_1g="$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')"
[[ -n "$uuid_1g" ]] || { echo "A 1g MIG UUID is required." >&2; exit 1; }

start_mps() {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  killall -9 nvidia-cuda-mps-control || true
  nvidia-cuda-mps-control -d || true
  sleep 2
}

stop_mps() {
  echo quit | nvidia-cuda-mps-control || true
  sleep 1
}

run_four_workers() {
  local label="$1"
  echo "[mps] ${label}"
  sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
    -v "${RESULT_DIR}:/results" \
    -v /tmp/nvidia-mps:/tmp/nvidia-mps \
    -v /tmp/nvidia-log:/tmp/nvidia-log \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$uuid_1g" \
    -e RUN_SECONDS="$RUN_SECONDS" \
    -e LABEL="$label" \
    --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    "$IMAGE" \
    bash -lc '
set -Eeuo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p /results
cat > /tmp/worker.py <<"PY"
import os, time, torch

torch.set_float32_matmul_precision("high")
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
for _ in range(10):
    torch.matmul(a, b)
    torch.cuda.synchronize()
end = time.monotonic() + float(os.environ["RUN_SECONDS"])
iterations = 0
while time.monotonic() < end:
    torch.matmul(a, b)
    torch.cuda.synchronize()
    iterations += 1
print(f"iterations={iterations}")
PY
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --output="/results/${LABEL}_a" python3 -u /tmp/worker.py &
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --output="/results/${LABEL}_b" python3 -u /tmp/worker.py &
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --output="/results/${LABEL}_c" python3 -u /tmp/worker.py &
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --output="/results/${LABEL}_d" python3 -u /tmp/worker.py &
wait
' || true
}

start_mps
run_four_workers "mps_on"
stop_mps
run_four_workers "mps_off"

ls -lh "$RESULT_DIR"/*.nsys-rep
