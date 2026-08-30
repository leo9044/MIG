# 📊 [실험 설계 캔버스] Jetson Thor 환경에서의 MIG/MPS 성능 및 아키텍처 병목 분석

## 1. MIG 기본 성능 및 하드웨어 격리 검증 (MIG Tests)

**목적:** MIG 적용 유무에 따른 성능 차이를 확인하고, MIG의 핵심 목적인 하드웨어 수준의 자원/오류 격리(Isolation)가 실제로 어느 정도 보장되는지를 검증한다.

* **실험 1-1. MIG vs Non-MIG 동시 실행 성능 비교**
* **Non-MIG:** GPU 전체 자원(단일 공간)에 워크로드 A, B를 동시에 할당하여 실행한다.
* **MIG 적용:** GPU를 1g, 2g 파티션으로 나눈 뒤, 각각 워크로드 A, B를 할당하여 동시에 실행한다.
* **지표:** 처리량(Throughput), 총 실행 시간, 평균 latency를 비교한다.


* **실험 1-2. 레이턴시 간섭(Interference) 측정**
* **상태:** 이미 수행 완료된 실험으로 간주한다. 관련 스크립트는 [MIG/scripts/test_1g.sh](MIG/scripts/test_1g.sh) 와 [MIG/scripts/test.sh](MIG/scripts/test.sh) 에서 확인 가능하다.
* **내용:** 2g 파티션 단독 실행 시의 레이턴시를 기준선으로 삼고, 1g 파티션에 지속적인 부하를 주었을 때 2g 워크로드의 레이턴시 변화가 얼마나 발생하는지를 관찰한다.
* **지표:** baseline 대비 latency 증가율과 연산 완료 시간 변화.


* **실험 1-3. 오류 전파 차단(Fault Isolation) 검증**
* **내용:** 1g 파티션에서 OOM 또는 커널 충돌을 유발하는 워크로드를 실행한 뒤, 2g 파티션의 작업이 영향을 받는지 확인한다.
* **지표:** 2g 파티션의 처리량, latency, 실행 지속 여부를 기준으로 격리 수준을 평가한다.


---

## 2. Jetson SoC 아키텍처 병목 원인 규명 (Additional Tests)

**목적:** MIG 실험에서 레이턴시 증가가 관찰될 경우, 서버형 GPU와 다른 Jetson Thor만의 SoC 구조적 제약이 병목으로 작용하는지 확인한다. 이 실험은 원인 확정이 아니라, 가장 가능성이 높은 후보를 좁혀 검증하는 데 초점을 둔다.

공개적으로 알려진 구조를 근거로 한 가설로는, Jetson Thor는 전체 GPU/CPU/기타 IP가 유사한 메모리 경로를 공유하는 구조를 갖고 있어, MIG 파티션 간 메모리 대역폭 경합과 shared memory access queue가 병목 후보가 될 가능성이 높다. 이는 NVIDIA MIG 문서와 Jetson 문서에서 설명하는 MIG 자원 분할 개념과 MPS 문서에서 설명하는 GPU scheduling/resource sharing 개념과 연결된다.

- MIG 사용자 가이드: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/
- MPS 문서: https://docs.nvidia.com/deploy/mps/latest/index.html
- Jetson 문서: https://docs.nvidia.com/jetson/

> 💡 **가설: 가장 가능성이 높은 원인 = 공유 메모리 경로의 대역폭 경합(UMA/DRAM bandwidth contention)**
> MIG가 각 파티션에 독립적인 compute slice를 제공하더라도, 최종적으로는 공유된 DRAM/메모리 컨트롤러 경로를 통과해야 하므로, 동시 실행 시 경합이 발생할 수 있다.

**[검증할 핵심 가설]**

- GPU 인스턴스 내부의 L2 캐시나 SM 자원은 분리될 수 있어도, 최종적인 DRAM 접근은 공유된 메모리 경로를 따라야 한다.
- 따라서 두 워크로드가 동시에 대규모 행렬 연산을 수행할 때, 메모리-bandwidth 사용률이 높아지고 latency가 상승하는 패턴이 관찰될 가능성이 높다.
- 이 가설을 검증하려면 Nsight Systems와 tegrastats를 함께 활용해 메모리/전력/latency 변화를 확인한다.

**실험 절차(보완 버전):**
1. 기준 상태: 1g 또는 2g 파티션 단독 실행, 20~30초 동안 steady-state matmul 수행.
2. 비교 상태: 같은 워크로드를 동시에 실행하거나, 1g 파티션에 CPU 메모리 stress를 함께 부여한다.
3. 데이터 수집:
   - `nsys profile`로 CUDA trace 기록
   - `tegrastats`로 전력/클럭/메모리 사용량 기록
   - `nvidia-smi` 또는 `dmon`으로 GPU utilization 확인
4. 결과 해석:
   - 단독 실행 대비 동시 실행 시 latency가 유의하게 증가하는가?
   - 메모리 접근이 병목으로 보이는가?
   - 전력/클럭이 함께 제한(throttling)되는가?

이 실험은 “원인 증명”이 아니라 “가설 검증” 수준으로 다룬다. 즉, 공개 구조와 실제 프로파일링 데이터가 서로 보완되는지 확인하는 것이 목표다.

---

## 3. MPS 도입 타당성 및 하이브리드 검증 (MPS Tests)

* **MPS 도입 타당성 (논리적 배경):**
* MIG는 물리적으로 GPU를 분할하는 방식이지만, 1g 파티션 안에 여러 프로세스를 넣으면 시분할(Time-slicing)로 교차 실행되며 비효율이 생길 수 있다.
* MPS는 이와 달리 하나의 GPU scheduling resource를 공유하면서 여러 프로세스의 CUDA 작업을 더 자연스럽게 겹쳐 실행하려는 목적을 가진다.
* MPS의 기본 개념은 NVIDIA 공식 MPS 문서에서 설명하고 있으며, 이를 비교 실험의 배경으로 활용한다.


* **실험 3-1. MIG 환경 내 MPS 동작성 검증**
**목적:** MIG 파티션(특히 1g slice) 안에서 MPS가 실제로 동작하는지 확인한다. 핵심 질문은 “MIG 환경에서도 MPS가 의미 있는 병렬성을 제공하는가?”이며, MPS 그 자체가 목적이 아니라 MIG 제약 조건 하에서 MPS의 유효성을 검증하는 것이 목적이다.
* 같은 1g MIG slice에서 MPS on/off 상태를 비교하고, 두 CUDA 프로세스를 동시에 실행할 때 overlap, 처리량, latency, 전체 완료 시간을 관찰한다.
* **기준:** 평균 완료 시간, throughput, p95 latency, 작업 간 overlap 여부, MPS 적용 시 동시 실행 안정성을 확인한다.


* **실험 3-2. MPS 정상 동작 검증 기준**
* > 💡 “MPS가 정상인지의 기준”은 다음 3가지로 설정한다.


* **기준 1. 커널 동시 실행 (Concurrency):** Nsight Systems 타임라인에서 프로세스 A와 B의 CUDA 커널이 겹쳐서 실행되는지 확인한다.
* **기준 2. 문맥 교환 비용 감소:** MPS 미적용 상태 대비 전체 실행 시간이 단축되는지 확인한다.
* **기준 3. 격리성 한계 확인:** 한 프로세스에 장애가 발생했을 때 MPS 데몬을 공유하는 다른 프로세스에도 영향을 주는지 확인한다.

이 실험은 MPS가 단순히 “동작했다”가 아니라, 실제로 성능을 높이고 실험 조건에서 병렬 실행을 안정적으로 보여주는지 확인하는 데 목적이 있다.

### 수행 범위 정리
- 수행 대상: 1-1, 1-3, 2, 3-1
- 완료/기존: 1-2
- 이후 확장: 3-2

**현재 기준으로 가장 현실적인 실험 범위는 위와 같다.**

---

## 3-1 추가 실험 — 다수의 경량 CUDA 작업에서 MPS 유효성 재검증

### 추가하는 이유

기존 3-1은 2000×2000 matmul을 수행하는 무거운 프로세스 2개를 비교했다. 이 workload는 1g slice를 이미 포화시킬 수 있어 MPS가 겹쳐 실행할 여유가 없는 조건일 수 있다. MPS의 목적에 더 맞는 조건은 여러 개의 짧은 CUDA 작업이 동시에 제출되는 경우다.

### 목적과 가설

**목적:** 같은 1g MIG slice에서 다수의 경량 CUDA worker가 작업을 제출할 때 MPS가 aggregate throughput 또는 kernel overlap을 높이는지 확인한다.

- H1: 작은 matmul을 여러 worker가 연속 제출하면 MPS on에서 aggregate throughput이 증가할 수 있다.
- H2: 차이가 없거나 MPS on이 낮으면 workload 포화, MPS 관리 비용, 공유 자원 경합이 이득보다 크다.
- H3: Nsight Systems timeline에 kernel overlap이 없으면 MPS 설정/동작 검증이 먼저 필요하다.

### 실험 설계

1. 동일한 1g MIG UUID에서 512×512 FP32 matmul worker를 4개, 8개 실행한다.
2. 각 worker는 50개 matmul을 연속 enqueue한 뒤 한 번 synchronize한다. batch 반복으로 짧은 커널의 동시 제출 기회를 만든다.
3. worker 수별로 MPS off/on을 각각 3회 실행하고 순서는 off → on → on → off → off → on으로 교차한다.
4. 매 batch의 CUDA-event 시간, wall time, 완료 operation 수를 CSV에 저장한다. aggregate throughput은 모든 worker throughput의 합이다.
5. 매 run에 tegrastats(100 ms)를 수집한다.
6. 정량 측정과 별도로 4-worker MPS on/off에서 Nsight Systems CUDA/NVTX trace를 짧게 캡처해 kernel overlap을 확인한다. profiler trace는 성능 수치 계산에 사용하지 않는다.

### 판정 기준과 한계

- 3회 반복 중앙값 aggregate throughput, p95 batch latency, worker 간 편차를 비교한다.
- MPS on 수치가 높고 timeline overlap도 확인되면 경량 다중 프로세스 조건의 MPS 효용을 뒷받침한다.
- 한 번의 실험이나 API timing만으로 일반 성능 우위를 주장하지 않는다.
