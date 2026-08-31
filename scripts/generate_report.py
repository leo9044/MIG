#!/usr/bin/env python3
"""Generate figures and a Korean report from completed benchmark CSV files only."""
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "actual_20260830"
FIG = OUT / "figures"


def read_csv(run, workload):
    path = RESULTS / run / f"{workload}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cuda = np.array([float(row["cuda_ms"]) for row in rows])
    wall = np.array([float(row["wall_ms"]) for row in rows])
    ts = np.array([float(row["start_unix_s"]) for row in rows])
    observed_s = ts[-1] - ts[0] + wall[-1] / 1000
    return {"run": run, "workload": workload, "path": str(path.relative_to(ROOT)), "rows": rows,
            "cuda": cuda, "wall": wall, "throughput": len(rows) / observed_s,
            "mean": float(cuda.mean()), "p95": float(np.percentile(cuda, 95)),
            "p99": float(np.percentile(cuda, 99)), "max": float(cuda.max()), "observed_s": float(observed_s)}


def combined(*items):
    return {"throughput": sum(x["throughput"] for x in items),
            "mean": float(np.concatenate([x["cuda"] for x in items]).mean()),
            "p95": float(np.percentile(np.concatenate([x["cuda"] for x in items]), 95)),
            "n": sum(len(x["cuda"]) for x in items)}


def bar(path, labels, values, title, ylabel, colors=None):
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, values, color=colors or "#34699a")
    for b, value in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.title(title); plt.ylabel(ylabel); plt.grid(axis="y", alpha=.25); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def pct(a, b):
    return (b / a - 1) * 100


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    mig1 = read_csv("20260830_175806_mig", "mig_1g")
    mig2 = read_csv("20260830_175806_mig", "mig_2g")
    nomig_a = read_csv("20260830_183524_nomig", "nomig_a")
    nomig_b = read_csv("20260830_183524_nomig", "nomig_b")
    solo = read_csv("20260830_183654_uma-solo", "uma_2g_solo")
    cont1 = read_csv("20260830_183727_uma", "uma_1g")
    cont2 = read_csv("20260830_183727_uma", "uma_2g")
    mps_off_a = read_csv("20260830_183859_mps-off", "worker_a")
    mps_off_b = read_csv("20260830_183859_mps-off", "worker_b")
    mps_on_a = read_csv("20260830_183932_mps-on", "worker_a")
    mps_on_b = read_csv("20260830_183932_mps-on", "worker_b")
    fault = read_csv("20260830_181838_fault", "fault_target")
    mig, nomig = combined(mig1, mig2), combined(nomig_a, nomig_b)
    mps_off, mps_on = combined(mps_off_a, mps_off_b), combined(mps_on_a, mps_on_b)

    bar(FIG / "exp_1_1_throughput.png", ["Non-MIG\n(two processes)", "MIG\n(1g + 2g)"],
        [nomig["throughput"], mig["throughput"]], "Experiment 1-1: aggregate throughput", "matmul operations / second", ["#777777", "#34699a"])
    bar(FIG / "exp_1_2_2g_latency.png", ["2g alone", "2g + 1g concurrent"], [solo["p95"], cont2["p95"]],
        "Experiment 1-2: 2g p95 latency", "ms", ["#777777", "#be6b45"])
    bar(FIG / "exp_1_2_2g_throughput.png", ["2g alone", "2g + 1g concurrent"], [solo["throughput"], cont2["throughput"]],
        "Experiment 1-2: 2g throughput", "matmul operations / second", ["#777777", "#be6b45"])
    bar(FIG / "exp_3_1_mps.png", ["MPS off", "MPS on"], [mps_off["throughput"], mps_on["throughput"]],
        "Experiment 3-1: aggregate throughput in 1g", "matmul operations / second", ["#777777", "#4e8f62"])

    # Plot a deterministic subsample; raw CSV remains the source of truth.
    take = max(1, len(fault["cuda"]) // 6000)
    elapsed = fault["cuda"][::take]
    time_s = (np.array([float(x["start_unix_s"]) for x in fault["rows"]])[::take] - float(fault["rows"][0]["start_unix_s"]))
    plt.figure(figsize=(7.2, 4.2)); plt.scatter(time_s, elapsed, s=2, alpha=.45, color="#9b3e3e")
    plt.yscale("log"); plt.ylim(max(float(fault["cuda"].min()) * .8, .001), fault["max"] * 1.2)
    plt.title("Experiment 1-3: 2g CUDA-event latency during 1g fault attempt")
    plt.xlabel("time after first measured operation (s)"); plt.ylabel("CUDA-event latency (ms)"); plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIG / "exp_1_3_fault_2g_timeline.png", dpi=200); plt.close()

    metrics = {"exp_1_1": {"mig": mig, "nomig": nomig}, "exp_1_2": {"solo_2g": {k: solo[k] for k in ("throughput", "mean", "p95", "p99")}, "concurrent_2g": {k: cont2[k] for k in ("throughput", "mean", "p95", "p99")}}, "exp_3_1": {"mps_off": mps_off, "mps_on": mps_on}, "exp_1_3": {k: fault[k] for k in ("throughput", "mean", "p95", "p99", "max", "path")}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = f"""# Jetson AGX Thor MIG/MPS 실험 보고서 — 실제 CSV 기반

## 데이터 범위와 측정 방법

- 장비: NVIDIA Thor, Driver 595.78, CUDA 13.2 (실행 당시 `nvidia-smi` 기록).
- 워크로드: 각 프로세스가 2000×2000 FP32 `torch.matmul`을 반복했다.
- latency는 각 matmul 앞뒤의 CUDA event로 측정했고 매 반복마다 `torch.cuda.synchronize()` 했다. 따라서 Nsight CUDA API 호출 시간이 아니라 GPU event elapsed time이다.
- throughput은 각 CSV의 반복 수 / 관측 구간으로 계산했다. 두 워커 조건의 aggregate throughput은 각 워커 throughput의 합이다.
- 조건마다 1회 실행만 존재한다. 평균·p95는 **그 실행 안의 반복 분포**이지 반복 실험의 신뢰구간이 아니다.

## 실험 1-1 — MIG vs Non-MIG 동시 실행

| 조건 | Aggregate throughput (ops/s) | 반복 수 |
|---|---:|---:|
| Non-MIG, 2 processes | {nomig['throughput']:.2f} | {nomig['n']} |
| MIG, 1g + 2g | {mig['throughput']:.2f} | {mig['n']} |

MIG 조건의 aggregate throughput은 Non-MIG 대비 {pct(nomig['throughput'], mig['throughput']):+.1f}%이다. 1g·2g·전체 GPU의 latency 분포를 섞은 aggregate p95는 직접 비교 가능한 지표가 아니므로 보고하지 않는다. 이 결과는 시스템 수준 처리량 비교다.

![Experiment 1-1 throughput](figures/exp_1_1_throughput.png)

## 실험 1-2 — 공유 메모리 경합 가설

2g 단독과 1g+2g 동시 실행에서 2g만 비교했다.

| 2g 조건 | Throughput (ops/s) | 평균 CUDA latency (ms) | p95 CUDA latency (ms) |
|---|---:|---:|---:|
| 단독 | {solo['throughput']:.2f} | {solo['mean']:.3f} | {solo['p95']:.3f} |
| 1g 동시 부하 | {cont2['throughput']:.2f} | {cont2['mean']:.3f} | {cont2['p95']:.3f} |

동시 부하에서 2g throughput은 {pct(solo['throughput'], cont2['throughput']):+.1f}%, p95 latency는 {pct(solo['p95'], cont2['p95']):+.1f}% 변했다. 이는 동시 실행이 2g 작업에 영향을 줄 수 있다는 **관측 결과**다. `tegrastats.log`는 각 run directory에 원시 로그로 보존했다. 이 한 번의 matmul 실험만으로 DRAM/메모리 컨트롤러 병목을 증명할 수는 없고, 원인 후보를 뒷받침하는 수준으로 해석해야 한다.

![Experiment 1-2 p95 latency](figures/exp_1_2_2g_latency.png)

![Experiment 1-2 throughput](figures/exp_1_2_2g_throughput.png)

## 실험 3-1 — 같은 1g slice에서 MPS on/off

| 조건 | Aggregate throughput (ops/s) | 평균 CUDA latency (ms) | p95 CUDA latency (ms) | 반복 수 |
|---|---:|---:|---:|---:|
| MPS off | {mps_off['throughput']:.2f} | {mps_off['mean']:.3f} | {mps_off['p95']:.3f} | {mps_off['n']} |
| MPS on | {mps_on['throughput']:.2f} | {mps_on['mean']:.3f} | {mps_on['p95']:.3f} | {mps_on['n']} |

MPS on의 aggregate throughput 변화는 {pct(mps_off['throughput'], mps_on['throughput']):+.1f}%이다. 이 수치는 MPS 서버가 시작된 조건에서 실제 두 CUDA 프로세스가 완료한 작업량 비교다. 커널 overlap 자체는 Nsight Systems 타임라인으로 별도 확인해야 하므로, 현재 데이터만으로 overlap을 단정하지 않는다.

![Experiment 3-1 MPS throughput](figures/exp_3_1_mps.png)

## 실험 1-3 — Fault isolation: 확인된 사실과 결론

보존된 fault run(`20260830_181838_fault`)에서 2g 대상 작업은 {len(fault['cuda'])}회를 완료했다. 평균 CUDA latency는 {fault['mean']:.3f} ms, p95는 {fault['p95']:.3f} ms, p99는 {fault['p99']:.3f} ms, 최대값은 {fault['max']:.3f} ms이다. 즉 2g CSV가 생성되고 정상 종료된 사실은 확인된다. 평균과 p95가 작더라도, 타임라인의 희소한 큰 이상치는 별도로 확인해야 한다.

그러나 1g 컨테이너에는 기대한 `expected_cuda_oom` JSON이 생성되지 않았다. 호스트 journal에는 같은 fault 실행 뒤 사용자 서비스가 OOM killer에 의해 종료된 기록이 있다. 따라서 이 실험은 “1g의 CUDA OOM이 2g에 격리됐다”를 입증하지 못한다. 더 정확한 결론은 **현재의 무한 CUDA 할당 방식이 Jetson UMA 시스템 메모리 부족을 유발했고, fault-isolation 실험으로는 안전하지 않았다**이다. 이 결과는 Jetson에서의 실험 설계 제약을 보여 준다.

![Experiment 1-3 2g latency timeline](figures/exp_1_3_fault_2g_timeline.png)

"""
    (OUT / "experiment_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
