# 실험 3-1 추가 — 경량 다중 worker에서 MPS on/off

## 목적

기존의 무거운 2000×2000 matmul 두 개 대신, 짧은 512×512 matmul을 여러 프로세스가 동시에 제출하는 조건에서 MPS의 효용을 확인했다. 이는 “여러 작은 CUDA 작업의 동시 제출”이라는 MPS의 사용 목적에 더 가까운 설정이다.

## 실험 방법

1. 1g MIG instance 하나에 512×512 FP32 matmul worker를 4개 또는 8개 연결했다.
2. 각 worker는 matmul 50개를 연속 enqueue한 뒤 한 번 `torch.cuda.synchronize()` 했다. 이 batch를 20초 동안 반복했다.
3. worker 수별로 MPS off와 MPS on을 각각 3회 수행했다. 순서는 `off → on → on → off → off → on`으로 교차해, 항상 한 조건이 먼저 실행되는 편향을 줄였다.
4. batch마다 CUDA event elapsed time, wall time, operation 수를 CSV에 기록했다. aggregate throughput은 worker별 throughput의 합이다.
5. 각 run에 100 ms 간격 tegrastats를 원시 로그로 저장했다.

## 결과

| Worker 수 | 조건 | Aggregate throughput 중앙값 (ops/s) | 3회 범위 (ops/s) | p95 batch latency 중앙값 (ms) |
|---:|---|---:|---:|---:|
| 4 | MPS off | 41,762.93 | 41,712.35–41,765.19 | 1.102 |
| 4 | MPS on | 25,508.32 | 25,435.85–25,752.45 | 7.912 |
| 8 | MPS off | 41,743.27 | 41,726.51–41,759.84 | 1.101 |
| 8 | MPS on | 25,152.84 | 25,137.53–25,623.16 | 15.829 |

- 4 worker: MPS on throughput은 중앙값 기준 **38.9% 감소**했고 p95 batch latency는 약 **7.2배**가 됐다.
- 8 worker: MPS on throughput은 **39.7% 감소**했고 p95 batch latency는 약 **14.4배**가 됐다.
- 각 조건 내 3회 범위는 작다. 특히 MPS off 처리량은 4·8 worker 모두 약 41.7k ops/s로 포화됐고, MPS on에서는 worker 수를 4에서 8로 늘려도 약 25k ops/s 수준에 머물렀다.

![4 worker throughput](exp_3_1_lightweight_analysis/throughput_workers4.png)

![8 worker throughput](exp_3_1_lightweight_analysis/throughput_workers8.png)

## 결론

이 장비·1g MIG·512×512 batch-50·20초 조건에서는 가설 H1, 즉 “MPS on이 경량 다중 worker aggregate throughput을 높인다”는 결과가 관찰되지 않았다. 반대로 MPS on은 처리량을 약 39% 낮추고 batch latency를 크게 높였다.

이것은 MPS 일반론의 반증은 아니다. 가능한 설명은 다음과 같다.

1. 1g slice의 실제 자원이 이 worker 수에서 이미 포화돼 MPS가 새 병렬성을 만들지 못했다.
2. 현재 MPS server 경로의 scheduling/IPC 비용 또는 이 Jetson MIG 구성의 제약이 짧은 kernel workload에서 크게 나타났다.
3. MPS server가 시작됐다는 사실과 kernel이 시간축에서 실제 overlap했다는 사실은 다르다. 현재 CSV는 성능만 보여 주므로 overlap 여부는 확인하지 못한다.

따라서 결론은 **이 경량 다중-worker 조건에서 반복 측정한 결과, MPS on의 성능상 이득은 관찰되지 않았고 오히려 일관된 손실이 관찰됐다**이다.

## 다음 검증

1. 4-worker에서 MPS on/off를 각각 5초씩 Nsight Systems CUDA/NVTX trace로 캡처해 kernel overlap을 직접 확인한다. profiler 결과는 throughput 수치에 섞지 않는다.
2. MPS on 실행 중 `echo get_server_list | nvidia-cuda-mps-control`로 server가 활성인지 확인한다.
3. 256·1024 matrix size 및 batch size 1·10·100을 바꿔, MPS에 유리한 커널 길이/제출 빈도 구간이 존재하는지 탐색한다.
4. tegrastats 로그의 온도·클럭을 run별로 표에 추가해 thermal/power 원인을 분리한다.
