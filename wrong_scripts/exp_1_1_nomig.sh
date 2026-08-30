#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_DIR="${RESULT_DIR:-/home/leo/MIG/experiments/fresh_20260830/exp_1_1/nomig}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
mkdir -p "$RESULT_DIR"

echo "=== Fresh experiment 1-1: non-MIG baseline ==="

# Keep the same hardware device access pattern as the legacy scripts.
sudo nvidia-smi mig -dci || true
sudo nvidia-smi mig -dgi || true
sudo nvidia-smi -mig 0
sleep 30

nvidia-smi -L | grep -q 'MIG ' && {
  echo "MIG cleanup did not complete; aborting baseline run." >&2
  exit 1
}

run_case() {
  local label="$1"
  echo "[nsys] ${label}: ${RUN_SECONDS}s capture"
  sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
    -v "${RESULT_DIR}:/results" \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="0" \
    -e RUN_SECONDS="$RUN_SECONDS" \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    "$IMAGE" \
    nsys profile \
      --force-overwrite=true \
      --trace=cuda,nvtx \
      --sample=none \
      --cpuctxsw=none \
      --output="/results/${label}" \
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
' || return $?

  [[ -f "${RESULT_DIR}/${label}.nsys-rep" ]] || {
    echo "Nsight report missing: ${RESULT_DIR}/${label}.nsys-rep" >&2
    return 1
  }
  sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/${label}.nsys-rep"
}

run_case nomig_workload_a
run_case nomig_workload_b

echo "=== restoring MIG mode after non-MIG baseline ==="
sudo nvidia-smi -mig 1
sleep 3
sudo nvidia-smi mig -cgi 83,78 -C || true
sleep 3
nvidia-smi -L

echo "Reports written to: $RESULT_DIR"
ls -lh "$RESULT_DIR"/*.nsys-rep
