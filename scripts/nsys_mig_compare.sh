#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/../nsys_results}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
MODE="${MODE:-all}"

mkdir -p "$RESULT_DIR"

run_profile() {
    local label="$1"
    local cuda_visible="$2"
    local report_path="/results/${label}"
    shift 2

    echo "[nsys] ${label}: ${RUN_SECONDS}s capture starting"
    local profile_status=0
    sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
        -v "${RESULT_DIR}:/results" \
        "$@" \
        "$IMAGE" \
        nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --duration="$RUN_SECONDS" \
        --output="$report_path" \
        python3 -u -c '
import time
import torch

torch.set_float32_matmul_precision("high")
a = torch.randn(2000, 2000, device="cuda")
b = torch.randn(2000, 2000, device="cuda")
for _ in range(10):
    torch.matmul(a, b)
torch.cuda.synchronize()

deadline = time.monotonic() + float("'"$RUN_SECONDS"'")
iterations = 0
torch.cuda.nvtx.range_push("steady_state_matmul")
while time.monotonic() < deadline:
    torch.matmul(a, b)
    iterations += 1
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print(f"iterations={iterations}")
' || profile_status=$?
    [[ "$profile_status" -eq 0 || "$profile_status" -eq 143 ]] || return "$profile_status"
    [[ -f "${RESULT_DIR}/${label}.nsys-rep" ]] || {
        echo "Nsight report was not created: ${RESULT_DIR}/${label}.nsys-rep" >&2
        return 1
    }
    sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/${label}.nsys-rep"
}

start_mig_capture() {
    echo "=== MIG capture: using the already-configured MIG instances ==="

    local mig_uuid_1g
    local mig_uuid_2g
    mig_uuid_1g="$(nvidia-smi -L | awk -F'UUID: |)' '/MIG 1g\.0gb/ {print $2; exit}')"
    mig_uuid_2g="$(nvidia-smi -L | awk -F'UUID: |)' '/MIG 2g\.0gb/ {print $2; exit}')"
    [[ -n "$mig_uuid_1g" && -n "$mig_uuid_2g" ]] || {
        echo "Could not find both 1g and 2g MIG UUIDs." >&2
        exit 1
    }

    run_profile mig_1g "$mig_uuid_1g" \
        --device /dev/nvidia-caps/nvidia-cap21 \
        --device /dev/nvidia-caps/nvidia-cap22 \
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm &
    local pid_1g=$!
    sleep 2
    run_profile mig_2g "$mig_uuid_2g" \
        --device /dev/nvidia-caps/nvidia-cap12 \
        --device /dev/nvidia-caps/nvidia-cap13 \
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm &
    local pid_2g=$!
    wait "$pid_1g" "$pid_2g"
}

start_nomig_capture() {
    echo "=== Disabling MIG for shared-GPU baseline ==="
    sudo nvidia-smi mig -dci
    sudo nvidia-smi mig -dgi
    sudo nvidia-smi -mig 0
    sleep 30

    nvidia-smi -L | grep -q 'MIG ' && {
        echo "MIG instances are still present; refusing to run baseline." >&2
        exit 1
    }

    run_profile nomig_workload_a 0 \
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm &
    local pid_a=$!
    sleep 2
    run_profile nomig_workload_b 0 \
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm &
    local pid_b=$!
    wait "$pid_a" "$pid_b"
}

case "$MODE" in
    mig)
        start_mig_capture
        ;;
    nomig)
        start_nomig_capture
        ;;
    all)
        start_mig_capture
        start_nomig_capture
        ;;
    *)
        echo "MODE must be one of: mig, nomig, all" >&2
        exit 2
        ;;
esac

echo "Reports written to: $RESULT_DIR"
ls -lh "$RESULT_DIR"/*.nsys-rep