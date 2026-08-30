#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_DIR="${RESULT_DIR:-/home/leo/MIG/experiments/fresh_20260830/exp_2}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
mkdir -p "$RESULT_DIR"

echo "=== Fresh experiment 2: shared-memory-path hypothesis ==="

nvidia-smi -L | grep -q 'MIG 1g\.0gb' || {
  echo "MIG partitions missing; recreating them."
  sudo nvidia-smi -mig 1
  sleep 3
  sudo nvidia-smi mig -cgi 83,78 -C || true
  sleep 3
}

echo "[cpu] start memory stress on host"
python3 - <<'PY' &
import time, numpy as np
# Keep CPU memory traffic active for the full 20-second GPU workload window.
end_time = time.monotonic() + 30
buf = np.ones(128 * 1024 * 1024, dtype=np.float64)
while time.monotonic() < end_time:
    buf = np.add(buf, 1.0)
PY
stress_pid=$!

echo "Waiting 3 seconds for CPU stress to reach peak memory bandwidth..."
sleep 3

uuid_1g="$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')"
[[ -n "$uuid_1g" ]] || { echo "A 1g MIG UUID is required." >&2; exit 1; }

echo "[nsys] mig_1g_shared_mem_stress"
sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
  -v "${RESULT_DIR}:/results" \
  -e NVIDIA_VISIBLE_DEVICES="all" \
  -e CUDA_VISIBLE_DEVICES="$uuid_1g" \
  -e RUN_SECONDS="$RUN_SECONDS" \
  --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  "$IMAGE" \
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output="/results/shared_mem_1g_stress" \
    python3 -u -c '
import os, time, torch

torch.set_float32_matmul_precision("high")
a = torch.randn(2000, 2000, device="cuda")
b = torch.randn(2000, 2000, device="cuda")
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
' || true

kill "$stress_pid" || true
ls -lh "$RESULT_DIR"/*.nsys-rep
