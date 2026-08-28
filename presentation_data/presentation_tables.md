# 발표용 표

## 1. MIG 인스턴스 비교

| 조건 | 대표 GEMM kernel 실행 횟수 | 총 GPU kernel 시간 | 평균 kernel 시간 |
|---|---:|---:|---:|
| MIG 1g | 12,675 | 18.744 s | 1.479 ms |
| MIG 2g | 37,176 | 18.624 s | 0.501 ms |

같은 20초 동안 2g가 1g보다 더 많은 kernel을 실행했고 평균 kernel 시간도 짧았다. 이 결과는 인스턴스 크기에 따른 compute 자원 차이가 GEMM 처리량에 반영된 것으로 해석할 수 있다. 단일 실행 결과이므로 반복 측정이 필요하다.

## 2. MPS on/off, 1g

| 조건 | 대표 GEMM kernel 실행 횟수 | 평균 kernel 시간 | 해석 |
|---|---:|---:|---|
| MPS on, client A | 10,913 | 0.860 ms | client별 report |
| MPS on, client B | 10,928 | 0.859 ms | client별 report |
| MPS off, 두 process | 18,151 | 2.045 ms | 두 process aggregate report |

MPS on에서는 두 client의 실행 횟수와 평균 시간이 거의 같아 process 간 균형을 보여준다. MPS off 값은 두 process를 합친 하나의 report이므로 MPS on 개별 client와 평균을 직접 비교하지 않는다. MPS의 총 처리량 우위는 동일한 report 단위와 3회 이상 반복 실험으로 확인해야 한다.

## 3. UMA CPU memory stress

| 지표 | CPU stress 없음 | CPU stress 있음 | 변화 |
|---|---:|---:|---:|
| GPU matmul iterations | 22,418 | 21,030 | -6.2% |
| 평균 GPU kernel 시간 | 0.849 ms | 0.897 ms | +5.7% |
| CPU Read Throughput | 0.76% | 19.34% | 증가 |
| GPU Read Throughput | 33.48% | 30.16% | 감소 |
| DRAM Read Throughput | 33.62% | 49.43% | 증가 |
| DRAM Write Throughput | 6.04% | 24.83% | 증가 |

CPU memory stress에서 CPU와 DRAM traffic이 증가하고 GPU workload 처리량이 6.2% 감소했다. 따라서 MIG가 GPU 내부 compute 자원을 분할하더라도 Thor UMA의 공유 memory path 경합까지 모두 제거한다고 볼 수 없다. 다만 현재 workload에서는 slowdown이 약 6%이므로 치명적인 병목으로 단정하지 않는다.

## 발표 주의사항

- MIG/MPS 표는 최신 Toolkit 1.19.1과 raw `--device` 방식의 20초 raw report에서 추출했다. UMA 표는 Toolkit downgrade 전의 SoC Metrics `t264`와 `tegrastats` 수집 결과이므로, Toolkit 1.19.1 기준으로 엄밀히 비교하려면 UMA 실험을 다시 수행해야 한다.
- `.nsys-rep`는 단일 실행 결과다. 평균과 표준편차를 주장하려면 각 조건을 최소 3회 반복해야 한다.
- Nsight Systems는 실행 timeline과 counter의 상관관계를 보여주지만 단독으로 인과관계를 증명하지 않는다.
