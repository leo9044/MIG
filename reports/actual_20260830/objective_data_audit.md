# 실제 실험 데이터 객관성 감사

## 판정 기준

1. **측정값의 사실성:** 원시 CSV/JSON/journal이 존재하고 계산을 재현할 수 있는가.
2. **비교의 공정성:** 대상 workload·자원·실행 기간이 비교 가능한가.
3. **반복성:** 조건별 독립 반복과 실행 순서 통제가 있는가.
4. **결론 범위:** 데이터가 실제로 뒷받침하는 주장인가.

## 요약

| 항목 | 원시 데이터 | 데이터가 뒷받침하는 주장 | 신뢰도 | 데이터가 뒷받침하지 않는 주장 |
|---|---|---|---|---|
| 1-1 MIG/Non-MIG | 완결 CSV 4개 | 이 1회 실행에서 aggregate throughput이 +7.8% | 낮음~보통 | MIG latency가 309% 증가했다 / 일반적으로 더 빠르다 |
| 1-2 간섭 | 완결 CSV·tegrastats | 이 1회 실행에서 1g 동시 부하 시 2g throughput -15.6%, p95 +23.3% | 보통 이하 | DRAM 경합이 원인으로 증명됐다 |
| 1-3 fault | 2g CSV·journal OOM | 호스트 OOM killer 발생 | 높음 | 1g CUDA OOM이 2g에 격리됐다 |
| 3-1 기존 heavy MPS | CSV는 존재 | pilot에서 MPS on 수치가 낮게 관찰됨 | 낮음, 핵심 결과 제외 | MPS가 성능을 저하시킨다 |
| 3-1 추가 lightweight MPS | 4/8 worker × on/off × 3회 CSV | 이 설정에서 MPS on throughput 이득 미관찰, 약 39% 낮음 | 보통~높음 | 모든 MPS workload에서 느리다 / kernel overlap이 없다 |

## 실험 1-1

CUDA-event CSV와 throughput 계산은 재현 가능하다. 다만 MIG run은 1g·2g, Non-MIG run은 전체 GPU 두 worker이므로 latency의 자원 크기가 동일하지 않다. 각 조건은 1회이며 Non-MIG run은 과거 Bash `SECONDS` 특수 변수 문제로 46초, MIG worker는 30/32초였다. throughput은 실제 관측 시간으로 정규화했기 때문에 숫자 자체는 맞지만, 열·클럭·시간 경과를 통제한 반복 비교는 아니다.

따라서 +7.8%는 **이 실행에서의 관측값**으로만 사용한다. aggregate p95 latency +309.1%는 서로 다른 slice 분포를 합친 통계량이라 비교 지표에서 제외했다.

## 실험 1-2

비교 대상은 같은 2g worker이므로 latency·throughput 지표의 정의는 적절하다. -15.6% throughput과 +23.3% p95는 원시 CSV에서 재현된다. 그러나 단독 조건은 30초, 동시 조건은 과거 `SECONDS` 문제로 65초였고, 단독→동시 순서의 단일 반복이다. 온도·클럭 변화가 원인 일부일 수 있다.

결론은 “동시 부하와 성능 저하가 함께 관찰됐다”로 제한한다. 공유 DRAM 경합은 가설이며, 확정에는 수정된 `RUN_SECONDS` 스크립트로 조건별 최소 3회 반복과 tegrastats 시간축 비교가 필요하다.

## 실험 1-3

fault_oom Docker 명령 뒤 journal에 OOM killer가 사용자 서비스를 종료한 기록이 존재한다. 호스트 OOM이 일어났다는 증거는 강하다. 반면 1g 컨테이너의 `expected_cuda_oom` JSON은 없고, 2g CSV만 정상 종료됐다. 따라서 MIG fault isolation의 성공/실패를 판정할 데이터는 없다.

## 실험 3-1 기존 heavy MPS

MPS off run은 30초, on run은 63초였고 둘 다 한 번만 실행됐다. 이는 Bash `SECONDS` 특수 변수 사용으로 생긴 기간 오류이며 이후 `RUN_SECONDS`로 수정했다. throughput은 관측 시간으로 정규화됐지만 순차 실행의 thermal/clock 상태를 통제하지 못했다. 이 수치는 pilot/설계 변경 근거로만 남기고 성능 결론 표에서는 제외한다.

## 실험 3-1 추가 lightweight MPS

512×512 matmul, batch 50, 1g MIG slice에서 4/8 worker 각각 MPS off/on 3회가 모두 완결됐다. 실행 순서도 off→on→on→off→off→on으로 교차했고 각 조건 내 throughput 범위가 작다.

- 4 worker 중앙값: off 41,762.93 ops/s, on 25,508.32 ops/s (-38.9%)
- 8 worker 중앙값: off 41,743.27 ops/s, on 25,152.84 ops/s (-39.7%)

따라서 **이 정확한 workload·batch·1g 조건에서 MPS on이 throughput 이득을 주지 않았다는 관측**은 신뢰할 수 있다. 그러나 MPS server의 상태를 `get_server_list`로 보존하지 않았고 Nsight Systems kernel-overlap trace도 없으므로, 원인을 MPS 비동작·특정 scheduler 정책·일반 MPS 특성 중 하나로 단정할 수는 없다.

## 해석 및 활용 권고

주 결과는 1-2의 간섭 관측과 lightweight 3-1의 반복 MPS 결과로 구성한다. 1-1은 처리량 관측, 1-3은 안전하지 않은 fault injection의 교훈으로 보조 설명한다. 모든 수치에 대해 “이 장비와 이 workload에서”라는 범위를 유지한다.


## 실험 방식에 근거한 신뢰성 설명

### 공통 측정 방식

모든 성능 CSV는 PyTorch CUDA event를 GPU stream의 matmul 직전·직후에 기록하고 torch.cuda.synchronize() 후 읽었다. CPU에서 CUDA 호출이 반환되는 시간이나 Nsight CUDA API duration은 실제 GPU 연산 시간이 아닐 수 있지만, CUDA-event elapsed time은 해당 stream의 GPU 작업 구간을 측정한다. CSV에는 반복/batch별 wall time·시작 시각·완료 operation 수도 있어 throughput 계산을 재현할 수 있다.

따라서 **각 CSV run의 latency·throughput 계산값 자체는 신뢰할 수 있다.** 조건 간 차이가 원인 효과인지 여부는 비교 설계가 결정한다.

### 1-1

**믿을 수 있는 이유:** 동일한 2000×2000 matmul 두 개를 실제 CUDA process로 실행했고, 각 worker의 operation/실제 관측 시간으로 throughput을 계산했다. 따라서 그 실행의 +7.8% aggregate throughput은 재현 가능한 관측값이다.

**제한되는 이유:** MIG는 1g+2g, Non-MIG는 전체 GPU 두 worker여서 worker 자원 크기가 다르다. 조건별 1회뿐이고, 과거 SECONDS 오류로 30/32/46초의 실행 길이 차이와 열·clock·시간 경과를 통제하지 못했다. 따라서 일반적인 MIG 우위나 latency 우열은 주장할 수 없다.

### 1-2

**믿을 수 있는 이유:** 기준선과 비교 조건이 같은 2g MIG UUID, 같은 matrix size, 같은 CUDA-event 코드다. 의도적으로 달라진 것은 1g 동시 부하이므로 2g CSV의 -15.6% throughput, +23.3% p95는 실제 관측값이다. tegrastats 원시 로그도 존재한다.

**제한되는 이유:** 단독→동시의 고정 순서 단일 반복이고 run 기간도 30초/65초로 달랐다. 동시 부하의 영향이 DRAM bandwidth, memory-controller queue, 전력, thermal 중 무엇인지는 분리하지 못했다. 따라서 “성능 저하 동반 관측”까지만 말할 수 있다.

### 1-3

**믿을 수 있는 이유:** journal에는 1g fault_oom Docker 실행 뒤 systemd 서비스가 OOM killer로 종료된 시각이 있다. 이는 호스트 OOM의 운영체제 수준 증거다.

**제한되는 이유:** 1g CUDA OOM을 정상 catch했다는 JSON이 없고, host service가 종료됐다. OOM이 slice 내부에 격리됐는지와 2g fault isolation을 판단할 기준선이 없다. 따라서 host OOM만 결론이다.

### 3-1 기존 heavy MPS

**제한되는 이유:** off 30초/on 63초, 조건별 1회, 고정 순서다. 이는 Bash 특수 변수 SECONDS 구현 오류와 thermal/clock drift 가능성을 포함한다. CSV는 pilot의 실제 기록이지만 성능 비교 결론에는 사용하지 않는다.

### 3-1 추가 lightweight MPS

**믿을 수 있는 이유:** 같은 1g UUID·512×512·batch 50으로 4/8 worker 각각 off/on 3회를 반복했다. off→on→on→off→off→on으로 순서를 교차했고 조건 내 범위가 작다. worker가 동일 start epoch에 시작하며 모든 operation/time을 합산해 aggregate throughput 정의도 명확하다. 따라서 이 설정에서 MPS on이 약 39% 낮았다는 관측은 신뢰할 수 있다.

**제한되는 이유:** 512×512, batch 50, 4/8 worker, 1g slice라는 특정 설정에만 일반화된다. MPS server get_server_list 출력과 Nsight Systems kernel-overlap trace가 없으므로 성능 저하의 원인을 MPS 비동작·스케줄러 정책·일반 MPS 특성 중 하나로 특정할 수 없다. “MPS는 본질적으로 느리다” 또는 “kernel overlap이 없다”는 결론은 불가하다.
