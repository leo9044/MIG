# Jetson AGX Thor MIG 연구: Nsight Systems 실험 및 발표 가이드

이 문서는 현재 연구의 Nsight Systems 결과를 확인하고, 같은 실험을 혼자 반복하며, 발표에서 MIG의 자원 분할 효과를 설명하기 위한 안내서다.

## 1. 연구에서 Nsight Systems를 쓰는 이유

Nsight Systems는 CUDA API 호출과 GPU kernel 실행을 시간축으로 보여준다. 따라서 다음 질문에 답할 수 있다.

- 두 workload가 실제로 동시에 GPU를 사용했는가?
- MIG를 사용했을 때 각 workload가 서로의 kernel 실행을 방해하는가?
- 1g와 2g 인스턴스의 자원 크기 차이가 kernel 처리량과 실행 시간에 반영되는가?
- MIG를 사용하지 않고 하나의 GPU를 공유했을 때 kernel이 한 GPU 자원을 두고 경쟁하는가?

Nsight Systems만으로 모든 하드웨어 자원 사용률을 증명할 수 있는 것은 아니다. MIG 인스턴스의 실제 SM 수, 메모리 용량, 프로파일별 정적 자원 정보는 `nvidia-smi`와 MIG 프로파일 문서를 함께 사용해야 한다. Nsight Systems는 주로 시간적 실행 행태와 간섭 여부를 보여주는 자료다.

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

비-MIG 비교가 필요할 때만 다음을 실행한다. 이 명령은 MIG 인스턴스를 삭제하고 MIG 모드를 끈다.

```bash
cd /home/leo/MIG
MODE=nomig RUN_SECONDS=20 bash scripts/nsys_mig_compare.sh
```

생성 파일:

```text
nsys_results/nomig_workload_a.nsys-rep
nsys_results/nomig_workload_b.nsys-rep
```

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

MIG 결과에서 1g와 2g가 같은 시간대에 계속 실행되면, 두 인스턴스가 병렬로 동작했다는 시각적 근거가 된다. 하나의 `.nsys-rep`는 하나의 workload만 담고 있으므로, 두 파일을 각각 열어 같은 시간축 기준으로 비교하거나 GUI의 timeline export/screenshot을 사용한다.

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

baseline도 같은 방식으로 확인한다.

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --force-export=true nsys_results/nomig_workload_a.nsys-rep

nsys stats --report cuda_api_sum,cuda_gpu_kern_sum \
  --force-export=true nsys_results/nomig_workload_b.nsys-rep
```

보고서에 다음 문구가 나오면 유효하지 않은 파일이다.

```text
SKIPPED: ... does not contain CUDA trace data.
SKIPPED: ... does not contain CUDA kernel data.
```

현재 확보된 결과의 대표 수치는 다음과 같다.

| 조건 | 파일 | 대표 kernel instances | 대표 kernel total time |
|---|---|---:|---:|
| MIG 1g | `mig_1g.nsys-rep` | 1,947 | 1.746 s |
| MIG 2g | `mig_2g.nsys-rep` | 4,292 | 1.660 s |
| No MIG A | `nomig_workload_a.nsys-rep` | 30,618 | 18.207 s |
| No MIG B | `nomig_workload_b.nsys-rep` | 29,297 | 18.311 s |

주의: MIG 자료는 당시 3초 수집, baseline은 20초 수집이었다. 따라서 위의 총합 시간만 직접 비교하면 안 된다. 발표 자료를 새로 만들 때는 반드시 같은 `RUN_SECONDS`로 다시 수집하고, 다음 값도 계산한다.

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

MPS는 MIG와 역할이 다르다. MIG는 GPU 자원을 하드웨어 인스턴스로 분할하고, MPS는 하나의 CUDA context에서 여러 client process의 kernel을 함께 스케줄링하도록 돕는다. 이번 구성에서는 MPS는 1g에만 적용하고 2g는 비교용 workload로 유지한다.

### 원본 스크립트 평가

`scripts/mps_test_1g.sh`는 1g MIG UUID를 찾고, MPS pipe/log 디렉터리를 만든 뒤 `nvidia-cuda-mps-control -d`를 실행하고 두 Python process를 동시에 시작한다. 따라서 MPS를 적용한다는 핵심 구조는 맞다. 하지만 `while True`와 `wait` 때문에 자동 종료되지 않고, process별 실행량과 latency를 기록하지 않으며, 실행 후 MPS daemon 정리가 없다.

`scripts/mps_test_2g.sh`는 2g에서 CUDA event로 100회 matmul latency를 측정한다. 1g workload가 이미 실행 중이라는 전제에서는 유용하지만, 수동 실행 순서에 의존하고 측정 시간이 짧으며 반복 통계가 없다. 원본은 보존하고, 고정 시간 Nsight 측정은 `scripts/nsys_mps_compare.sh`가 담당한다.

### MPS 측정 실행

현재 MIG 인스턴스가 이미 설정된 상태에서는 다음 명령만 실행한다.

```bash
cd /home/leo/MIG
MODE=all RUN_SECONDS=20 bash scripts/nsys_mps_compare.sh
```

이 명령은 다음 순서로 실행한다.

1. 1g에서 MPS on, 두 client process를 시작한다.
2. 2초 후 2g latency workload를 시작한다.
3. MPS on 1g client를 각각 내부 Nsight로 프로파일링한다.
4. 같은 방식으로 MPS off 조건을 반복한다.

MPS on에서는 MPS server를 통해 실행되는 client를 바깥 Nsight가 안정적으로 추적하지 못할 수 있다. 그래서 1g의 두 client 각각에 컨테이너 내부 `nsys profile`을 직접 적용한다.

생성 파일:

```text
nsys_mps_results/mps_1g_a.nsys-rep
nsys_mps_results/mps_1g_b.nsys-rep
nsys_mps_results/mps_2g.nsys-rep
nsys_mps_results/nomps_1g.nsys-rep
nsys_mps_results/nomps_2g.nsys-rep
```

### 이번 20초 결과

| 조건 | report | 대표 kernel instances | 평균 kernel 시간 |
|---|---|---:|---:|
| MPS on, 1g client A | `mps_1g_a.nsys-rep` | 8,353 | 1.117 ms |
| MPS on, 1g client B | `mps_1g_b.nsys-rep` | 8,383 | 1.110 ms |
| MPS off, 1g 두 process 합계 | `nomps_1g.nsys-rep` | 11,912 | 3.105 ms |
| MPS on, 2g | `mps_2g.nsys-rep` | 37,635 | 0.400 ms |
| MPS off, 2g | `nomps_2g.nsys-rep` | 34,713 | 0.446 ms |

1g에서는 MPS on의 두 client가 각각 약 8.3k kernel을 실행해 합계 약 16.7k회를 기록했다. MPS off는 두 process가 하나의 CUDA 실행 흐름에서 경쟁하면서 합계 11.9k회, 평균 3.105ms로 측정됐다. 이 결과는 MPS가 같은 MIG 인스턴스 안의 여러 process를 더 균등하고 효율적으로 스케줄링할 수 있다는 근거다.

2g에서는 MPS on/off가 모두 1g와 다른 MIG 인스턴스에서 실행된다. 두 조건의 차이가 작다면, 1g 내부의 MPS 스케줄링이 2g 자원에 큰 간섭을 주지 않았다는 MIG 격리 근거로 설명할 수 있다. 이번 측정에서는 MPS on의 2g kernel 평균이 0.400ms, off가 0.446ms였다. 이 차이는 유망하지만 단일 실행 결과이므로 최소 3회 반복 후 평균과 표준편차로 발표해야 한다.

### 발표에서 강조할 것

MPS의 효과를 `MPS가 GPU 자원을 추가했다`고 설명하면 안 된다. 정확한 표현은 다음과 같다.

> MIG가 1g와 2g 사이의 하드웨어 자원 경계를 제공하고, MPS는 그중 1g 인스턴스 내부에서 여러 CUDA process의 kernel 실행을 효율적으로 multiplexing한다. 따라서 이번 실험은 MIG의 인스턴스 격리와 MPS의 process-level scheduling 효과를 분리해 관찰한다.

Nsight GUI에서는 MPS on에서 `mps_1g_a`와 `mps_1g_b`를 각각 열어 kernel 실행 간격과 평균 시간을 비교한다. MPS off에서는 `nomps_1g` timeline에서 두 process의 kernel이 긴 간격과 변동을 보이는지 확인한다. 2g report는 `mps_2g`와 `nomps_2g`의 kernel 평균, 최대값, latency 분포를 비교해 1g MPS가 다른 MIG 인스턴스에 미치는 영향을 설명하는 데 사용한다.
