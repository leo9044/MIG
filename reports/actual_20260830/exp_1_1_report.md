# 실험 1-1 — MIG vs Non-MIG 동시 실행 성능

## 목적

GPU 전체를 두 프로세스가 공유하는 경우와, GPU를 1g·2g MIG 인스턴스로 분할해 각각 한 프로세스를 배정한 경우의 실제 처리량을 비교한다.

## 배경지식과 가설

MIG(Multi-Instance GPU)는 한 GPU 안의 일부 연산 자원을 논리적 GPU 인스턴스로 나누는 기능이다. Non-MIG에서는 두 CUDA 프로세스가 한 GPU의 스케줄러와 자원을 함께 사용한다.

가설은 “MIG 자원 분할은 두 동시 작업의 aggregate throughput 특성을 바꾼다”이다. 1g과 2g의 크기가 다르므로, 이 실험은 개별 slice의 절대 성능 공정 비교가 아니라 시스템 수준 처리량 비교다.

## 방법

1. 각 프로세스는 Docker 안에서 2000×2000 FP32 torch.matmul을 반복했다.
2. 각 반복에서 CUDA event를 matmul 직전·직후에 기록하고 torch.cuda.synchronize() 후 elapsed time을 CSV에 기록했다.
3. MIG 조건은 1g 컨테이너를 먼저 시작하고 2초 후 2g 컨테이너를 시작했다. 기존 검증 스크립트와 동일하게 MIG UUID, cap21/22(1g), cap12/13(2g), nvidia 장치 옵션을 사용했다.
4. Non-MIG 조건은 MIG를 끈 뒤 같은 크기의 두 워크로드를 GPU 0에서 동시에 실행했다.
5. throughput은 각 CSV의 반복 수 / 실제 관측 시간이며 aggregate throughput은 두 프로세스 throughput의 합이다.

## 실제 결과

| 조건 | Aggregate throughput (ops/s) |
|---|---:|
| Non-MIG, 2 processes | 2648.13 |
| MIG, 1g + 2g | 2854.06 |

MIG aggregate throughput은 Non-MIG 대비 **+7.8%**였다.

worker별 p95 CUDA-event latency는 Non-MIG A/B가 0.286/0.288 ms, MIG 1g/2g가 1.231/0.484 ms였다. 그러나 이 네 분포는 서로 다른 크기의 GPU 자원(전체 GPU, 1g, 2g)에서 얻은 것이다. 이를 합쳐 만든 aggregate p95 1.175 ms는 1g의 느린 분포가 지배한 값이므로, **“MIG p95 latency가 +309.1% 증가했다”는 직접 비교나 결론으로 사용하지 않는다.**

![Aggregate throughput](figures/exp_1_1_throughput.png)

## 해석과 결론

이 한 번의 실행에서는 MIG가 aggregate throughput을 소폭 높였다. latency는 동일한 자원 크기·동일 workload의 조건이 아니므로 이 실험에서 MIG/Non-MIG latency 우열을 주장할 수 없다.

## 발표 문장

“두 동시 작업의 총 처리량은 MIG에서 7.8% 높았습니다. 다만 1g·2g와 전체 GPU의 latency 분포는 직접 비교 가능한 동일 자원 조건이 아니므로, latency 우열은 이 실험에서 주장하지 않았습니다.”
