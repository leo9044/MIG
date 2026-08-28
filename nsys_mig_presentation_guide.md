# Jetson AGX Thor MIG 연구: Nsight Systems 실험 및 발표 가이드

이 문서는 현재 연구의 Nsight Systems 결과를 확인하고, 같은 실험을 혼자 반복하며, 발표에서 MIG의 자원 분할 효과를 설명하기 위한 안내서다.

## 1. 연구에서 Nsight Systems를 쓰는 이유

Nsight Systems는 CUDA API 호출과 GPU kernel 실행을 시간축으로 보여준다. 따라서 다음 질문에 답할 수 있다.

- 두 workload가 실제로 동시에 GPU를 사용했는가?
- MIG를 사용했을 때 각 workload가 서로의 kernel 실행을 방해하는가?
- 1g와 2g 인스턴스의 자원 크기 차이가 kernel 처리량과 실행 시간에 반영되는가?
- MIG를 사용하지 않고 하나의 GPU를 공유했을 때 kernel이 한 GPU 자원을 두고 경쟁하는가?

Nsight Systems만으로 모든 하드웨어 자원 사용률을 증명할 수 있는 것은 아니다. MIG 인스턴스의 실제 SM 수, 메모리 용량, 프로파일별 정적 자원 정보는 `nvidia-smi`와 MIG 프로파일 문서를 함께 사용해야 한다. Nsight Systems는 주로 시간적 실행 행태와 간섭 여부를 보여주는 자료다.

### 실험 환경 기준

핵심 MIG/MPS 재실험은 다음 환경에서 수행했다.

```text
Jetson AGX Thor
Ubuntu 24.04 / L4T 39.2.1
NVIDIA driver 595.78
Docker Engine 29.7.2
NVIDIA Container Toolkit 1.19.1
PyTorch container nvcr.io/nvidia/pytorch:25.08-py3
```

JetPack 전체 meta-package는 설치하지 않았지만 L4T CUDA와 driver는 설치되어 있고, PyTorch/CUDA runtime은 컨테이너 내부에서 사용했다.

## 2. 실행 순서

### MIG 설정

재부팅 직후 최초 1회만 MIG 설정을 한다.

```bash
cd /home/leo/MIG/scripts
./setup_mig.sh
sleep 30
nvidia-smi -L
```

`setup_mig.sh`는 다음 두 인스턴스를 만든다.

- `83`: `2g.0gb`
- `78`: `1g.0gb`

그 뒤에는 같은 재부팅 상태에서 `setup_mig.sh`를 반복 실행하지 않는다. 이미 인스턴스가 있는 경우 바로 측정 스크립트를 실행한다.

### MIG workload 측정

```bash
cd /home/leo/MIG
MODE=mig RUN_SECONDS=20 bash scripts/nsys_mig_compare.sh
```

실행기는 다음 순서를 지킨다.

1. 1g 컨테이너와 내부 `nsys` 실행
2. 2초 대기
3. 2g 컨테이너와 내부 `nsys` 실행
4. 두 workload 종료 후 결과 저장

생성 파일:

```text
nsys_results/mig_1g.nsys-rep
nsys_results/mig_2g.nsys-rep
```

### 비-MIG baseline 측정

비-MIG 비교는 별도 조건 실험이다. 현재 발표용 결과 디렉터리에는 최신 Toolkit/raw-device 조건과 직접 비교할 수 없는 이전 No-MIG report를 포함하지 않았다. 발표에 No-MIG를 넣으려면 MIG를 해제한 뒤 동일한 Toolkit, device 전달 방식, workload, 실행 시간으로 새로 수집한다.

현재 시스템을 다시 MIG로 사용할 경우, MIG 모드 변경 뒤 GPU reset 또는 재부팅이 필요할 수 있다. Thor에서 비-MIG 전환 후 `GPU requires reset` 상태가 발생했고, `nvidia-smi -r`는 지원되지 않았다.

## 3. Nsight Systems GUI에서 볼 것

`.nsys-rep` 파일을 Nsight Systems GUI에서 연다. 보고서마다 다음 순서로 확인한다.

### 3.1 Timeline

가장 먼저 시간축 전체를 본다.

확인할 것:

- CUDA API row에 kernel launch가 반복되는가?
- GPU row에 kernel 실행 구간이 실제로 채워져 있는가?
- MIG 1g와 2g의 실행 구간이 같은 시간대에 겹치는가?
- 두 workload 중 하나가 다른 workload 때문에 긴 빈 구간이나 대기 구간을 보이는가?
- GPU context 또는 CUDA stream이 반복적으로 전환되는가?

MIG 결과에서 각 report의 GPU kernel이 측정 구간에 지속적으로 나타나고, 실행기가 1g를 먼저 시작한 뒤 2g를 2초 후 시작했다는 로그가 있으면 병렬 실행을 뒷받침하는 자료가 된다. 두 인스턴스는 별도 `.nsys-rep`로 수집되므로 GUI multi-report view로 정렬할 수 있지만, 시각적 정렬만으로 동시성을 단정하지 말고 실행 로그와 workload 결과를 함께 제시한다.

### 3.2 CUDA API row

`cuLaunchKernelEx` 호출을 본다.

중요한 항목:

- 호출 횟수: `Num Calls`
- API 총 시간: `Total Time`
- 평균 호출 시간: `Avg`
- 호출 사이의 긴 간격

API 총 시간은 GPU kernel이 실제로 실행된 시간과 동일하지 않다. API 호출이 오래 걸리는 이유에는 launch overhead와 synchronization이 포함될 수 있으므로, 성능 판단은 반드시 GPU kernel summary와 함께 한다.

### 3.3 GPU kernel row

`cutlass3x_sm100_tensorop...`로 시작하는 matmul kernel을 찾는다.

중요한 항목:

- `Instances`: kernel 실행 횟수
- `Total Time`: 해당 kernel들의 총 GPU 시간
- `Avg`: kernel 1회의 평균 실행 시간
- `Med`: 중앙값
- `Max`: 최대 실행 시간
- timeline에서 kernel 간격이 일정한지 여부

이번 workload에서는 이 matmul kernel이 대표 kernel이다. 1g에서 평균 kernel 시간이 더 길고 실행 횟수가 적다면, 작은 MIG 인스턴스가 같은 연산을 처리하는 데 더 많은 시간이 필요하다는 해석이 가능하다. 단, 1g와 2g workload의 workload 양과 측정 시간이 반드시 같아야 정량 비교가 공정하다.

### 3.4 NVTX range

코드에는 `steady_state_matmul` NVTX range가 있다. 초기 CUDA context 생성과 tensor allocation 구간을 제외하고 이 range 안의 steady-state 구간을 비교한다.

발표에서는 다음처럼 설명한다.

> 초기화 비용은 측정 대상에서 제외하고, 동일한 matmul 반복 구간만 비교했다. 따라서 그래프의 차이는 주로 MIG 자원 크기와 GPU 실행 경쟁의 차이를 반영한다.

## 4. 터미널에서 숫자로 검증하기

Nsight Systems CLI의 정확한 report 이름은 underscore를 포함한다.

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --force-export=true nsys_results/mig_1g.nsys-rep
```

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --force-export=true nsys_results/mig_2g.nsys-rep
```

보고서에 다음 문구가 나오면 유효하지 않은 파일이다.

```text
SKIPPED: ... does not contain CUDA trace data.
SKIPPED: ... does not contain CUDA kernel data.
```

최신 발표용 MIG 결과는 Toolkit 1.19.1과 raw `--device` 방식으로 20초 수집했다.

| 조건 | 대표 kernel instances | 대표 kernel total time | 평균 kernel 시간 |
|---|---:|---:|---:|
| MIG 1g | 12,675 | 18.744 s | 1.479 ms |
| MIG 2g | 37,176 | 18.624 s | 0.501 ms |

이전에 수집한 No-MIG report는 Toolkit 1.20.0과 MIG UUID 환경 변수 사용 조건의 자료이므로 최신 MIG 결과와 직접 비교하지 않는다. No-MIG 비교를 발표에 포함하려면 Toolkit 1.19.1/raw `--device` 조건에서 동일한 `RUN_SECONDS`로 다시 수집해야 한다.

```text
kernel_instances_per_second = Instances / RUN_SECONDS
kernel_gpu_seconds_per_second = Total_Time_seconds / RUN_SECONDS
average_kernel_time = Total_Time / Instances
```

## 5. 발표에서의 설명 방식

### 핵심 주장

다음 세 문장 구조가 가장 안전하다.

1. MIG는 하나의 물리 GPU를 독립적인 GPU 인스턴스로 분할한다.
2. Nsight Systems timeline에서 1g와 2g workload가 동시에 실행되는 것을 확인하고, 각 report의 kernel statistics로 실행량과 실행 시간을 정량화했다.
3. No-MIG 조건에서는 두 workload가 하나의 GPU 자원을 공유하지만, MIG 조건에서는 각 workload가 할당된 인스턴스 자원 안에서 실행된다.

### 그래프 해석

- 1g의 평균 kernel 시간이 2g보다 길다: 1g에 할당된 GPU 자원이 작기 때문일 가능성이 있다.
- 2g의 kernel instances/s가 1g보다 높다: 더 큰 인스턴스가 같은 시간 동안 더 많은 연산을 처리한다는 근거다.
- No-MIG에서 두 workload의 실행 시간이 불규칙하거나 긴 지연이 보인다: 하나의 GPU 자원에서 두 workload가 경쟁하거나 serialization되는 현상일 수 있다.
- MIG에서 두 workload의 latency가 상대적으로 안정적이다: 자원 격리와 예측 가능성이 개선됐다는 근거다.

단정할 때는 항상 `가능성`, `관찰`, `근거`를 분리한다. 예를 들어 다음처럼 말한다.

> Timeline에서는 두 MIG workload가 겹치는 동안에도 각각의 GPU kernel이 진행됐다. 또한 1g와 2g의 kernel 평균 시간과 실행 횟수가 달랐다. 이는 인스턴스 크기에 따른 자원 배분 차이가 workload 처리량에 반영된 결과로 해석할 수 있다. 다만 MIG의 완전한 독립성을 주장하려면 반복 실험과 SM/메모리 사용률 측정이 추가로 필요하다.

### 발표 슬라이드 구성 예시

1. 실험 목적: MIG가 workload 자원을 분할하고 간섭을 줄이는가?
2. 환경: Jetson AGX Thor, PyTorch container, matmul workload, 1g + 2g MIG
3. 실험 조건: MIG와 No-MIG, 동일한 workload, 동일한 실행 시간
4. Nsight timeline screenshot: MIG 조건의 동시 실행
5. Kernel summary 표: instances, total time, average, median
6. No-MIG와 MIG 비교 그래프: instances/s, 평균 kernel time, GPU time/s
7. 결론과 한계: 자원 격리 효과와 추가 검증 항목

## 6. 이번 작업에서 사용한 명령어와 시행착오

### 현재 상태 확인

```bash
nvidia-smi -L
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

첫 명령은 GPU UUID와 MIG UUID를 확인하는 데 사용했다. 두 번째 명령은 측정 중 잔류 CUDA process가 있는지 확인하는 데 사용했다.

### 스크립트 문법 확인

```bash
bash -n /home/leo/MIG/scripts/nsys_mig_compare.sh
```

실제 GPU를 사용하지 않고 Bash 문법만 검사한다. 수정할 때마다 가장 먼저 실행하면 된다.

### 처음 시도: 호스트의 Nsight로 Docker를 감싸기

처음에는 다음 구조를 사용했다.

```text
sudo nsys profile ... docker run ... python3 ...
```

보고서 파일은 생성됐지만 `nsys stats`에서 CUDA trace가 없다고 나왔다. Docker daemon이 만든 컨테이너 process는 호스트 `nsys`의 일반 process-tree 주입 범위 밖에서 실행됐기 때문이다. `--cuda-trace-scope=system-wide`도 시도했지만 동시 system-wide session에서 공유 메모리 충돌이 발생했고, Docker process에 안정적으로 주입되지 않았다.

### 잘못된 report 이름

처음에는 다음처럼 입력했다.

```bash
nsys stats --report cudaapisum report.nsys-rep
nsys stats --report gpukernsum report.nsys-rep
```

현재 버전의 정확한 이름은 다음이다.

```text
cuda_api_sum
cuda_gpu_kern_sum
cuda_gpu_trace
```

### 컨테이너 내부 Nsight로 전환

PyTorch 이미지 안에 다음 실행 파일이 있음을 확인했다.

```bash
sudo docker run --rm --runtime nvidia \
  nvcr.io/nvidia/pytorch:25.08-py3 bash -lc \
  'command -v nsys'
```

출력:

```text
/usr/local/cuda/bin/nsys
```

그래서 최종 실행 구조는 다음처럼 변경했다.

```text
sudo docker run ... IMAGE nsys profile ... python3 ...
```

프로파일러가 Python process를 직접 감싸기 때문에 CUDA API와 kernel trace가 정상적으로 생성됐다.

### 동시 system-wide session의 실패

1g와 2g 각각에 호스트 `nsys profile --cuda-trace-scope=system-wide`를 동시에 실행했을 때 다음 경고가 발생했다.

```text
shared memory file at /dev/shm/cuda_injection_path_shm already exists
```

Nsight의 system-wide CUDA injection은 한 번에 여러 session을 안정적으로 실행하기 어렵다. 최종 방식은 각 컨테이너 내부에서 process-tree 범위로 별도 프로파일링하는 방식이다.

### 컨테이너 Python 오류

고정 시간 반복을 전달하는 과정에서 다음 오류도 발생했다.

```text
KeyError: 'RUN_SECONDS'
SyntaxError: unexpected character after line continuation character
```

첫 오류는 컨테이너에 `-e RUN_SECONDS=...`를 전달하지 않아 발생했다. 두 번째 오류는 Bash 문자열 안에서 Python `\\n`을 과도하게 escape했기 때문이다. 최종 스크립트는 컨테이너에 실행 시간을 전달하고, Python workload를 정상적인 multi-line 문자열로 실행한다.

### No-MIG 전환 후 GPU reset 문제

baseline을 위해 다음 명령이 실행된다.

```bash
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi
sudo nvidia-smi -mig 0
```

MIG 모드 변경 후 Thor에서는 GPU가 다음 상태가 될 수 있다.

```text
MIG Mode: GPU requires reset
```

이 경우 `nvidia-smi -r`가 `Not Supported`를 반환할 수 있다. 실제 해결 방법은 재부팅이었다. 따라서 MIG와 No-MIG 비교 실험 사이에는 GPU 상태를 확인하고, 필요하면 재부팅한 뒤 조건을 다시 설정한다.

### sudo 권한

Docker 실행과 `nvidia-smi` MIG 변경에는 sudo가 필요하다.

```bash
sudo -v
```

비밀번호를 스크립트에 저장하지 말고, 터미널에서 직접 인증한다. 현재 `setup_mig.sh`에는 비밀번호가 평문으로 들어 있으므로 연구 장비 외부에 공유하거나 Git에 올리면 안 된다.

## 7. 혼자 재현할 때의 체크리스트

- [ ] `nvidia-smi -L`로 현재 조건을 확인했다.
- [ ] 재부팅 직후라면 `setup_mig.sh`를 1회 실행하고 30초 기다렸다.
- [ ] 이후에는 설정 스크립트를 반복 실행하지 않았다.
- [ ] `sudo -v`를 실행했다.
- [ ] MIG 조건에서 1g가 먼저 시작되고 2초 후 2g가 시작되는지 확인했다.
- [ ] MIG와 No-MIG의 `RUN_SECONDS`를 동일하게 맞췄다.
- [ ] `.nsys-rep` 파일이 생성됐다.
- [ ] `cuda_api_sum`과 `cuda_gpu_kern_sum`이 `SKIPPED`되지 않았다.
- [ ] total time이 아니라 초당 실행량과 평균 kernel time을 비교했다.
- [ ] 같은 workload, 같은 matrix 크기, 같은 warm-up, 같은 측정 시간을 사용했다.
- [ ] 가능하면 최소 3회 반복하고 평균과 표준편차를 기록했다.

## 8. 다음에 추가하면 좋은 자료

Nsight Systems 결과만으로도 실행 시간과 kernel 간섭을 설명할 수 있다. 자원 할당 연구를 더 강하게 만들려면 다음 자료를 추가한다.

- `nvidia-smi`의 MIG 인스턴스 목록과 UUID screenshot
- MIG별 사용 메모리와 GPU utilization 기록
- 같은 workload를 단독 실행했을 때의 baseline
- MIG 1g 단독, 2g 단독, 1g+2g 동시 실행 결과
- No-MIG 단독, No-MIG 두 workload 동시 실행 결과
- 각 조건 최소 3회 반복 결과
- `RUN_SECONDS`가 동일한 raw report와 CSV summary

## 9. MPS on/off 실험

MPS는 MIG와 역할이 다르다. MIG는 GPU 자원을 하드웨어 인스턴스로 분할하고, MPS는 하나의 1g 인스턴스 안에서 여러 client process의 kernel을 함께 스케줄링하도록 돕는다. 이번 MPS 질문의 핵심은 1g 내부의 process scheduling이므로 2g는 필수 측정 대상이 아니다.

### 원본 스크립트 평가

`scripts/mps_test_1g.sh`는 1g MIG UUID를 찾고, MPS pipe/log 디렉터리를 만든 뒤 `nvidia-cuda-mps-control -d`를 실행하고 두 Python process를 동시에 시작한다. 따라서 MPS를 적용한다는 핵심 구조는 맞다. 하지만 `while True`와 `wait` 때문에 자동 종료되지 않고, process별 실행량과 latency를 기록하지 않으며, 실행 후 MPS daemon 정리가 없다.

`scripts/mps_test_2g.sh`는 2g에서 CUDA event로 100회 matmul latency를 측정한다. 1g workload가 이미 실행 중이라는 전제에서는 유용하지만, 수동 실행 순서에 의존하고 측정 시간이 짧으며 반복 통계가 없다. 원본은 보존하고, 고정 시간 Nsight 측정은 `scripts/nsys_mps_compare.sh`가 담당한다. MPS의 핵심 질문은 1g 내부 process scheduling이므로 최종 MPS 자료에서는 2g report를 사용하지 않았다.

### MPS 측정 실행

현재 MIG 인스턴스가 이미 설정된 상태에서는 다음 명령만 실행한다.

```bash
cd /home/leo/MIG
MODE=all RUN_SECONDS=20 bash scripts/nsys_mps_compare.sh
```

이 명령은 다음 순서로 실행한다.

1. 1g에서 MPS on, 두 client process를 시작한다.
2. MPS on 1g client를 각각 내부 Nsight로 프로파일링한다.
3. 같은 workload를 MPS off 조건에서 반복한다.

MPS on에서는 MPS server를 통해 실행되는 client를 바깥 Nsight가 안정적으로 추적하지 못할 수 있다. 그래서 1g의 두 client 각각에 컨테이너 내부 `nsys profile`을 직접 적용한다.

생성 파일:

```text
nsys_mps_results/mps_1g_a.nsys-rep
nsys_mps_results/mps_1g_b.nsys-rep
nsys_mps_results/nomps_1g.nsys-rep
```

### 이번 20초 결과

| 조건 | report | 대표 kernel instances | 평균 kernel 시간 |
|---|---|---:|---:|
| MPS on, 1g client A | `mps_1g_a.nsys-rep` | 10,913 | 0.860 ms |
| MPS on, 1g client B | `mps_1g_b.nsys-rep` | 10,928 | 0.859 ms |
| MPS off, 1g 두 process 합계 | `nomps_1g.nsys-rep` | 18,151 | 2.045 ms |

1g에서는 최신 20초 MPS on 측정에서 두 client가 각각 10,913회와 10,928회의 대표 kernel을 실행했고 평균 kernel 시간은 각각 0.860 ms와 0.859 ms였다. MPS off report는 두 process를 하나의 report로 집계해 18,151회의 kernel과 2.045 ms 평균을 보였다. 이 aggregate 값은 MPS on의 개별 client 평균과 직접 비교하면 안 된다. 현재 결과는 MPS on에서 두 client가 균등하게 실행된다는 근거로 사용하고, 총 처리량 우위 주장은 동일한 process별 측정과 반복 실험 후에만 한다.

### 발표에서 강조할 것

MPS의 효과를 `MPS가 GPU 자원을 추가했다`고 설명하면 안 된다. 정확한 표현은 다음과 같다.

> MIG가 1g와 2g 사이의 하드웨어 자원 경계를 제공하고, MPS는 그중 1g 인스턴스 내부에서 여러 CUDA process의 kernel 실행을 효율적으로 multiplexing한다. 따라서 이번 실험은 MIG의 인스턴스 격리와 MPS의 process-level scheduling 효과를 분리해 관찰한다.

Nsight GUI에서는 MPS on에서 `mps_1g_a`와 `mps_1g_b`를 각각 열어 kernel 실행 간격과 평균 시간을 비교한다. MPS off에서는 `nomps_1g` timeline에서 두 process의 kernel이 긴 간격과 변동을 보이는지 확인한다.

## 10. 다음에 해볼 만한 검증

### UMA bus bandwidth contention

Thor는 CPU와 GPU가 같은 물리 메모리 시스템을 사용하는 UMA 구조이므로, MIG가 GPU 내부의 compute 인스턴스를 분리해도 CPU/GPU 요청이 공통 SoC memory controller와 DRAM 경로에서 경쟁할 가능성이 있다. 다만 이것을 “칩셋 레벨 bus contention”이라고 단정하기보다, 현재 공개 자료로는 “공유 UMA memory path contention”이라고 표현하는 것이 정확하다. Thor의 구체적인 interconnect arbitration과 MIG별 memory QoS 보장 여부는 공개 자료만으로 확정할 수 없으며, counter 실험으로 확인해야 한다.

하드웨어 관점에서는 두 층을 구분한다.

- GPU 내부 층: MIG가 각 인스턴스에 compute/GPU 자원을 배정하고 인스턴스 간 실행 간섭을 줄인다.
- SoC 메모리 층: CPU, GPU, 그리고 다른 SoC 블록의 요청이 공통 memory controller/DRAM 경로를 사용할 수 있다. 이 층의 경합은 MIG가 자동으로 모두 제거한다고 가정하면 안 된다.

권장 실험은 다음 네 조건을 같은 workload와 같은 시간으로 비교하는 것이다.

1. MIG + CPU memory stress 없음
2. MIG + CPU memory stress 있음
3. No-MIG + CPU memory stress 없음
4. No-MIG + CPU memory stress 있음

GPU workload는 현재 matmul을 사용하고, CPU stress는 별도 CPU process가 큰 배열을 반복해서 읽고 쓰도록 만든다. CPU stress는 GPU 인스턴스가 아니라 시스템 UMA bandwidth를 압박한다.

Nsight Systems에서 볼 항목:

- GPU kernel의 `Avg`, `Med`, `Max`, `StdDev`
- kernel 사이의 빈 구간과 지연 증가
- CUDA API의 `cudaDeviceSynchronize` 대기 시간
- CPU thread의 memory loop와 GPU kernel timeline 겹침
- CPU utilization과 context switch 증가

Nsight Systems의 Thor SoC Metrics는 이 실험에 직접 사용할 수 있다. 현재 장치에서 확인된 metric set은 `t264`이며, 다음 계열을 수집 대상으로 삼는다.

- CPU read/write throughput
- GPU read/write throughput
- 전체 DRAM read/write throughput
- DBB read/write throughput

예시 명령은 다음과 같다. 실제 workload 명령은 마지막에 붙인다.

```bash
sudo nsys profile --soc-metrics=true --soc-metrics-set=t264 \
  --soc-metrics-frequency=1000 --trace=cuda,osrt,nvtx \
  --sample=none --cpuctxsw=process-tree -d 20 -o uma_mig_cpu_stress \
  <workload>
```

`tegrastats --interval 100`도 동시에 별도 로그로 남긴다. Nsight Systems는 timestamp로 GPU kernel과 SoC counter를 맞춰 볼 수 있고, `tegrastats`는 전력/온도/메모리 압박 등 시스템 상태를 보조한다. 발표 결론은 “MIG가 GPU execution partition은 유지하지만 공유 UMA memory path의 모든 contention을 제거하지는 않는다”처럼 제한한다.

가장 중요한 판정은 다음과 같다. CPU memory stress를 추가했을 때 CPU/GPU throughput counter가 함께 올라가고 GPU kernel 평균/최대 시간이 증가하면 UMA contention의 근거다. GPU kernel 시간이 증가하지 않고 CPU counter만 올라가면 해당 workload에서는 UMA 경합이 지배적이지 않다고 판단한다.

### CPU launch starvation

GPU kernel은 CPU가 launch하고 synchronize하므로, CPU가 중요한 workload에 점유되거나 scheduling delay를 겪으면 GPU가 idle해질 수 있다. 이 현상은 특히 짧은 kernel을 빠르게 반복하는 workload에서 잘 드러난다.

권장 비교 조건:

- CPU affinity를 1g workload와 겹치게 설정
- CPU affinity를 분리해 설정
- CPU stress 없음/있음
- MIG와 No-MIG 각각 반복

이 실험은 GPU workload를 짧은 kernel 반복으로 구성해야 한다. 큰 matmul 하나만 반복하면 GPU가 실행 중인 시간이 길어 CPU launch gap이 가려질 수 있다.

Nsight Systems에서 확인할 항목:

- CUDA API의 `cuLaunchKernelEx` 호출 시각과 GPU kernel 시작 시각 사이 간격
- GPU timeline의 빈 구간
- CPU thread state에서 `Running`, `Runnable`, `Blocked` 구간
- OS runtime, context switch, scheduler latency report
- NVTX range의 시작 시각과 첫 GPU kernel 시각 사이 지연

해석 기준은 다음과 같다. GPU row가 비어 있는데 다음 `cuLaunchKernelEx`가 늦게 호출되거나 CPU thread가 `Runnable`/다른 process에 밀려 있다면 CPU launch starvation 가능성이 있다. 반대로 CPU가 실행 중인데 GPU kernel이 계속 차 있다면 CPU가 병목이라고 단정할 수 없다. 이 경우 GPU execution 또는 UMA bandwidth가 원인일 수 있으므로 kernel latency와 SoC memory counter를 함께 봐야 한다.

현재 환경에서는 `nsys status --environment` 결과상 process-tree CPU profiling은 가능하지만 system-wide CPU profiling은 실패한다. 또한 CPU hardware metric은 `perf_event_paranoid=2` 때문에 root가 필요하다. 따라서 1차 실험은 workload container 안의 `--cpuctxsw=process-tree`로 수행하고, 시스템 전체 scheduler와 CPU PMU counter가 필요할 때만 `sudo`를 사용한다.

## 11. 권장 최종 실험 매트릭스

연구 목적에 가장 직접적인 순서는 다음과 같다.

| 단계 | 변경 요인 | 고정 요인 | 핵심 측정 |
|---|---|---|---|
| A | MIG 1g 단독/2g 단독/1g+2g 동시 | workload, CPU affinity | kernel latency, instances/s |
| B | No-MIG 단독/두 process 동시 | workload, CPU affinity | serialization, tail latency |
| C | 1g MPS off/on | MIG layout, workload | process fairness, throughput |
| D | CPU memory stress off/on | MIG layout, GPU workload | CPU/GPU/DRAM throughput, kernel slowdown |
| E | CPU affinity 분리/충돌 | MIG layout, memory stress | launch gap, GPU idle gap, context switch |

각 조건은 최소 3회 반복한다. D와 E가 “MIG의 효과가 어디까지인가”를 보여주는 핵심 확장이다. A-C는 자원 배분과 MPS scheduling을 설명하고, D는 UMA 공유 경로, E는 CPU 제어 경로를 검증한다.

## 12. 도구별 역할

- **Nsight Systems**: CUDA API, GPU kernel timeline, CPU thread state, context switch, NVTX, Thor SoC Metrics를 하나의 시간축에서 상관 분석한다.
- **Nsight Compute**: 대표 kernel의 occupancy, memory throughput, tensor pipe 등 kernel 내부 원인을 분석한다. Systems에서 병목 kernel을 고른 뒤 사용한다.
- **tegrastats**: Jetson 전체의 보조 상태 로그를 기록한다. CPU/GPU load, memory, EMC, temperature, power를 workload 시간과 맞춘다.
- **nvidia-smi**: MIG 인스턴스 UUID, 프로파일, 현재 process와 설정 상태를 기록한다. 대역폭 counter 도구로 사용하지 않는다.
- **perf**: CPU scheduler/PMU를 별도 검증할 때 사용한다. container 권한과 `perf_event_paranoid` 영향을 받으므로 Nsight CPU 결과와 교차 확인한다.

발표에서는 Nsight Systems screenshot을 주 증거로 사용하고, SoC Metrics와 `tegrastats`를 UMA 경합의 정량 보조 증거로 제시한다. Nsight Compute는 “왜 이 kernel이 느려졌는가”를 설명하는 후속 자료로 사용한다.

## 13. UMA 측정 결과

`scripts/nsys_uma_compare.sh`로 MIG 1g workload를 20초 동안 측정했다. `nostress`는 CPU memory stress가 없고, `stress`는 4개의 NumPy process가 각각 256 MiB 배열을 반복해서 읽고 쓰는 조건이다. 두 조건 모두 같은 MIG 1g UUID, 같은 CPU affinity, 같은 PyTorch matmul을 사용했다.

### 최종 결과

| 지표 | CPU stress 없음 | CPU stress 있음 | 변화 |
|---|---:|---:|---:|
| GPU matmul iterations | 22,418 | 21,030 | -6.2% |
| 평균 GPU kernel 시간 | 0.849 ms | 0.897 ms | +5.7% |
| CPU Read Throughput | 0.76% | 19.34% | 증가 |
| GPU Read Throughput | 33.48% | 30.16% | 감소 |
| DRAM Read Throughput | 33.62% | 49.43% | 증가 |
| DRAM Write Throughput | 6.04% | 24.83% | 증가 |

보고서와 보조 로그:

```text
nsys_uma_results/nostress_soc.nsys-rep
nsys_uma_results/stress_soc.nsys-rep
nsys_uma_results/nostress_tegrastats.log
nsys_uma_results/stress_tegrastats.log
nsys_uma_results/nostress_workload.txt
nsys_uma_results/stress_workload.txt
```

### 해석

CPU stress 조건에서 CPU read와 DRAM read/write throughput이 크게 증가했고 GPU read throughput은 감소했다. 같은 시점에 GPU matmul 평균 시간이 0.849 ms에서 0.897 ms로 증가하고 반복 처리량이 6.2% 감소했다. 이 결과는 MIG 1g의 GPU compute partition이 존재해도 Thor의 공유 UMA memory path에서 CPU와 GPU traffic이 경합할 수 있다는 실험적 근거다.

이번 결과만으로 CPU traffic이 항상 가장 치명적인 병목이라고 결론 내리면 안 된다. 깨끗한 20초 재측정에서 slowdown은 약 5.7%로 측정됐으므로 이 workload에서는 경합이 존재하지만 지배적인 병목은 아닐 수 있다. memory-bound kernel, 더 많은 CPU stress worker, 다른 CPU affinity, 또는 더 작은 GPU kernel을 추가해 민감도를 확인해야 한다.

또한 이 실험의 SoC report는 `--trace=none`으로 수집한 system-level metric report다. GPU kernel의 정확한 API/kernel 통계는 별도 CUDA Nsight report에서 확인하고, 이 report에서는 SoC throughput과 `tegrastats`를 시간축으로 맞춰 해석한다. 따라서 발표에서는 “SoC counter와 workload timing이 함께 변했다”고 표현하고, counter 하나만으로 인과관계를 단정하지 않는다.

### 실험 시 주의점

- CPU stress process의 CPU affinity를 명시해 MIG 인스턴스의 workload CPU와 겹침/분리를 통제한다.
- CPU stress의 배열이 LLC에만 머물지 않도록 충분히 크게 잡는다.
- CPU와 GPU workload의 warm-up을 측정 구간에서 제외한다.
- 같은 `RUN_SECONDS`, matrix 크기, kernel 수집 옵션을 사용한다.
- 한 조건을 최소 3회 반복하고 평균, 중앙값, 표준편차를 기록한다.
- MPS, MIG, CPU stress, UMA traffic을 한 번에 모두 바꾸지 말고 한 요인씩 비교한다.
