#!/usr/bin/env bash
set -Eeuo pipefail

RESULT_DIR="${RESULT_DIR:-/home/leo/MIG/experiments/fresh_20260830/exp_1_1/mig}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
mkdir -p "$RESULT_DIR"

echo "=== Fresh experiment 1-1: MIG baseline ==="

nvidia-smi -L | grep -q 'MIG 2g\.0gb' || {
  echo "MIG mode or 2g instance is not available. Recreating MIG partitions..."
  sudo nvidia-smi -mig 1
  sleep 3
  sudo nvidia-smi mig -cgi 83,78 -C || true
  sleep 3
}

uuid_1g="$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')"
uuid_2g="$(nvidia-smi -L | grep "MIG 2g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')"
[[ -n "$uuid_1g" && -n "$uuid_2g" ]] || {
  echo "Could not find both 1g and 2g MIG UUIDs." >&2
  exit 1
}

run_case() {
  local label="$1"
  local uuid="$2"
  local device_map=("--device" "/dev/nvidia-caps/nvidia-cap21" "--device" "/dev/nvidia-caps/nvidia-cap22" "--device" "/dev/nvidia0" "--device" "/dev/nvidiactl" "--device" "/dev/nvidia-uvm")

  echo "[nsys] ${label}: ${RUN_SECONDS}s capture"
  sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
    -v "${RESULT_DIR}:/results" \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$uuid" \
    -e RUN_SECONDS="$RUN_SECONDS" \
    "${device_map[@]}" \
    "$IMAGE" \
    nsys profile \
      --force-overwrite=true \
      --trace=cuda,nvtx \
      --sample=none \
      --cpuctxsw=none \
      --duration="$RUN_SECONDS" \
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
    iterations += 1
torch.cuda.synchronize()
print(f"iterations={iterations}")
' || return $?

  [[ -f "${RESULT_DIR}/${label}.nsys-rep" ]] || {
    echo "Nsight report missing: ${RESULT_DIR}/${label}.nsys-rep" >&2
    return 1
  }
  sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/${label}.nsys-rep"
}

# Use the actual MIG UUIDs to bind the session to each partition.
# The same workload is captured on both slices to compare the isolated case.
for label in mig_1g mig_2g; do
  case "$label" in
    mig_1g) run_case "$label" "$uuid_1g" ;;
    mig_2g) run_case "$label" "$uuid_2g" ;;
  esac
done

echo "Reports written to: $RESULT_DIR"
ls -lh "$RESULT_DIR"/*.nsys-rep
