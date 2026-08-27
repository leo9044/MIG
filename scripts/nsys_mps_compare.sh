#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/../nsys_mps_results}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
MODE="${MODE:-all}"

mkdir -p "$RESULT_DIR"

get_uuid() {
    local profile="$1"
    nvidia-smi -L | awk -F'UUID: |)' -v profile="$profile" '$0 ~ profile {print $2; exit}'
}

workload_1g='import os, time, torch; torch.set_float32_matmul_precision("high"); a = torch.randn(2000, 2000, device="cuda"); deadline = time.monotonic() + float(os.environ["RUN_SECONDS"]); iterations = 0; torch.cuda.nvtx.range_push("steady_state_matmul"); exec("while time.monotonic() < deadline:\n    torch.matmul(a, a)\n    iterations += 1"); torch.cuda.nvtx.range_pop(); torch.cuda.synchronize(); print(f"iterations={iterations}")'
workload_2g='import os, time, torch; torch.set_float32_matmul_precision("high"); a = torch.randn(2000, 2000, device="cuda"); b = torch.randn(2000, 2000, device="cuda"); [torch.matmul(a, b) for _ in range(10)]; torch.cuda.synchronize(); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); deadline = time.monotonic() + float(os.environ["RUN_SECONDS"]); total_ms = 0.0; count = 0; torch.cuda.nvtx.range_push("steady_state_matmul"); exec("while time.monotonic() < deadline:\n    start.record(); torch.matmul(a, b); end.record(); torch.cuda.synchronize(); total_ms += start.elapsed_time(end); count += 1"); torch.cuda.nvtx.range_pop(); print(f"iterations={count} total_gpu_ms={total_ms:.3f} avg_gpu_ms={total_ms / count:.3f}")'

profile_1g() {
    local label="$1"
    local uuid="$2"
    local use_mps="$3"
    local report_path="/results/${label}"
    local profile_status=0
    local docker_args=(run --rm --runtime nvidia --pid=host --ipc=host
        -v "${RESULT_DIR}:/results"
        -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="$uuid"
        -e RUN_SECONDS="$RUN_SECONDS"
        --device /dev/nvidia-caps/nvidia-cap21 --device /dev/nvidia-caps/nvidia-cap22
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm
        --cpuset-cpus=2-11)

    echo "[nsys] ${label}: ${RUN_SECONDS}s capture, MPS=${use_mps}"
    if [[ "$use_mps" == "on" ]]; then
        local command="export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p \"\$CUDA_MPS_PIPE_DIRECTORY\" \"\$CUDA_MPS_LOG_DIRECTORY\"
nvidia-cuda-mps-control -d
sleep 2
cleanup() { echo quit | nvidia-cuda-mps-control || true; }
trap cleanup EXIT TERM INT
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --duration=\"\$RUN_SECONDS\" --output=/results/${label}_a python3 -u -c '${workload_1g}' &
first=\$!
nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none --duration=\"\$RUN_SECONDS\" --output=/results/${label}_b python3 -u -c '${workload_1g}' &
second=\$!
wait \"\$first\" \"\$second\""
        sudo docker "${docker_args[@]}" "$IMAGE" bash -lc "$command" || profile_status=$?
    else
        local command="python3 -u -c '${workload_1g}' &
python3 -u -c '${workload_1g}' &
wait"
        sudo docker "${docker_args[@]}" "$IMAGE" nsys profile \
            --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
            --duration="$RUN_SECONDS" --output="$report_path" bash -lc "$command" || profile_status=$?
    fi
    [[ "$profile_status" -eq 0 || "$profile_status" -eq 143 ]] || return "$profile_status"
    if [[ "$use_mps" == "on" ]]; then
        [[ -f "${RESULT_DIR}/${label}_a.nsys-rep" && -f "${RESULT_DIR}/${label}_b.nsys-rep" ]] || return 1
        sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/${label}"_*.nsys-rep
    else
        [[ -f "${RESULT_DIR}/${label}.nsys-rep" ]] || return 1
        sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/${label}.nsys-rep"
    fi
}

profile_2g() {
    local label="$1"
    local uuid="$2"
    local report_path="/results/${label}"
    local profile_status=0
    echo "[nsys] ${label}: ${RUN_SECONDS}s capture, MPS=off on 2g"
    sudo docker run --rm --runtime nvidia --pid=host --ipc=host \
        -v "${RESULT_DIR}:/results" \
        -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES="$uuid" -e RUN_SECONDS="$RUN_SECONDS" \
        --device /dev/nvidia-caps/nvidia-cap12 --device /dev/nvidia-caps/nvidia-cap13 \
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --cpuset-cpus=0-1 \
        "$IMAGE" nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --duration="$RUN_SECONDS" --output="$report_path" python3 -u -c "$workload_2g" || profile_status=$?
    [[ "$profile_status" -eq 0 || "$profile_status" -eq 143 ]] || return "$profile_status"
    [[ -f "${RESULT_DIR}/${label}.nsys-rep" ]] || return 1
    sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/${label}.nsys-rep"
}

run_condition() {
    local condition="$1"
    local use_mps="off"
    [[ "$condition" == "mps" ]] && use_mps="on"
    local uuid_1g="$(get_uuid 'MIG 1g\.0gb')"
    local uuid_2g="$(get_uuid 'MIG 2g\.0gb')"
    [[ -n "$uuid_1g" && -n "$uuid_2g" ]] || {
        echo "Both 1g and 2g MIG UUIDs are required." >&2
        return 1
    }

    profile_1g "${condition}_1g" "$uuid_1g" "$use_mps" &
    local pid_1g=$!
    sleep 2
    profile_2g "${condition}_2g" "$uuid_2g" &
    local pid_2g=$!
    wait "$pid_1g" "$pid_2g"
}

case "$MODE" in
    mps) run_condition mps ;;
    nomps) run_condition nomps ;;
    all) run_condition mps; run_condition nomps ;;
    *) echo "MODE must be one of: mps, nomps, all" >&2; exit 2 ;;
esac

echo "Reports written to: $RESULT_DIR"
ls -lh "$RESULT_DIR"/*.nsys-rep
