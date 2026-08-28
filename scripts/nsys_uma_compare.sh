#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/../nsys_uma_results}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
RUN_SECONDS="${RUN_SECONDS:-20}"
mkdir -p "$RESULT_DIR"

uuid_1g="$(nvidia-smi -L | awk -F'UUID: |)' '/MIG 1g\.0gb/ {print $2; exit}')"
[[ -n "$uuid_1g" ]] || { echo "A 1g MIG UUID is required." >&2; exit 1; }

run_condition() {
    local condition="$1"
    local stress="$2"
    local report="/results/${condition}_1g"
    local soc_report="${RESULT_DIR}/${condition}_soc"
    local tegra_log="${RESULT_DIR}/${condition}_tegrastats.log"
    local status=0

    echo "[UMA] ${condition}: CPU memory stress=${stress}, ${RUN_SECONDS}s"
    sudo tegrastats --interval 100 --logfile "$tegra_log" --start
    trap 'sudo tegrastats --stop >/dev/null 2>&1 || true' RETURN

    sudo nsys profile \
        --run-as=root \
        --force-overwrite=true \
        --soc-metrics=true \
        --soc-metrics-set=t264 \
        --soc-metrics-frequency=1000 \
        --trace=none \
        --sample=none \
        --cpuctxsw=system-wide \
        --kill=none \
        --duration="$RUN_SECONDS" \
        --output="$soc_report" \
        docker run --rm --runtime nvidia --pid=host --ipc=host \
            -v "${RESULT_DIR}:/results" \
            --cpuset-cpus=2-11 \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e CUDA_VISIBLE_DEVICES="$uuid_1g" \
            -e RUN_SECONDS="$RUN_SECONDS" \
            -e CPU_STRESS="$stress" \
            -e RESULT_FILE="/results/${condition}_workload.txt" \
            --device /dev/nvidia-caps/nvidia-cap21 \
            --device /dev/nvidia-caps/nvidia-cap22 \
            --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
            "$IMAGE" \
            python3 -u -c '
import multiprocessing as mp
import os
import time
import torch


def memory_stress():
    import numpy as np

    buffer = np.ones(32 * 1024 * 1024, dtype=np.float64)
    while True:
        np.add(buffer, 1.0, out=buffer)


def gpu_workload():
    torch.set_float32_matmul_precision("high")
    matrix = torch.randn(2000, 2000, device="cuda")
    for _ in range(10):
        torch.matmul(matrix, matrix)
    torch.cuda.synchronize()
    deadline = time.monotonic() + float(os.environ["RUN_SECONDS"])
    iterations = 0
    total_ms = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push("uma_steady_state_matmul")
    while time.monotonic() < deadline:
        start.record()
        torch.matmul(matrix, matrix)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
        iterations += 1
    torch.cuda.nvtx.range_pop()
    result = f"iterations={iterations} total_gpu_ms={total_ms:.3f} avg_gpu_ms={total_ms / iterations:.3f}\n"
    print(result, end="", flush=True)
    with open(os.environ["RESULT_FILE"], "w", encoding="ascii") as output:
        output.write(result)


stress_processes = []
if os.environ["CPU_STRESS"] == "on":
    for _ in range(4):
        process = mp.Process(target=memory_stress)
        process.start()
        stress_processes.append(process)
gpu_workload()
for process in stress_processes:
    process.terminate()
for process in stress_processes:
    process.join()
' || status=$?

    sudo tegrastats --stop >/dev/null 2>&1 || true
    trap - RETURN
    [[ "$status" -eq 0 || "$status" -eq 143 ]] || return "$status"
    [[ -f "${soc_report}.nsys-rep" ]] || {
        echo "SoC report was not created: ${soc_report}.nsys-rep" >&2
        return 1
    }
}

case "${MODE:-all}" in
    stress)
        run_condition stress on
        ;;
    nostress)
        run_condition nostress off
        ;;
    all)
        run_condition nostress off
        run_condition stress on
        ;;
    *)
        echo "MODE must be one of: stress, nostress, all" >&2
        exit 2
        ;;
esac

printf 'Reports written to: %s\n' "$RESULT_DIR"
ls -lh "$RESULT_DIR"/*
