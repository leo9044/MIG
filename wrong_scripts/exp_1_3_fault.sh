#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_DIR="${RESULT_DIR:-/home/leo/MIG/experiments/fresh_20260830/exp_1_3}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
mkdir -p "$RESULT_DIR"

echo "=== Fresh experiment 1-3: fault isolation ==="

nvidia-smi -L | grep -q 'MIG 1g\.0gb' || {
  echo "1g MIG instance is missing. Recreating MIG partitions..."
  sudo nvidia-smi -mig 1
  sleep 3
  sudo nvidia-smi mig -cgi 83,78 -C || true
  sleep 3
}

uuid_1g="$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')"
uuid_2g="$(nvidia-smi -L | grep "MIG 2g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')"
[[ -n "$uuid_1g" && -n "$uuid_2g" ]] || {
  echo "Expected 1g and 2g MIG partitions are not available." >&2
  exit 1
}

# Fault injection must happen while the 2g slice is already under active load.
# This matches the intended MIG fault-isolation scenario: 2g keeps running while 1g faults.

echo "[nsys] mig_2g_fault_isolation (background)"
sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
  -v "${RESULT_DIR}:/results" \
  -e NVIDIA_VISIBLE_DEVICES="all" \
  -e CUDA_VISIBLE_DEVICES="$uuid_2g" \
  -e RUN_SECONDS="$RUN_SECONDS" \
  --device /dev/nvidia-caps/nvidia-cap12 --device /dev/nvidia-caps/nvidia-cap13 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  "$IMAGE" \
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output="/results/mig_2g_fault_isolation" \
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
' &

# Let the 2g workload stabilize before injecting the fault into the other slice.
echo "Waiting 5 seconds for 2g workload to warm up..."
sleep 5

echo "[fault] trigger 1g OOM condition while 2g is active"
sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES="all" \
  -e CUDA_VISIBLE_DEVICES="$uuid_1g" \
  -e RUN_SECONDS="$RUN_SECONDS" \
  --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  "$IMAGE" \
  python3 -u -c '
import torch
try:
    print("Allocating a large tensor to trigger OOM on 1g...")
    x = torch.empty(200000, 200000, device="cuda", dtype=torch.float16)
    print("OOM_TRIGGERED")
except Exception as e:
    print(f"FAULT CAUGHT on 1g: {type(e).__name__} - {e}")
' || true

echo "Waiting for background 2g workload to finish..."
wait

ls -lh "$RESULT_DIR"/*.nsys-rep
