# Jetson Thor MIG/MPS 실험

Jetson AGX Thor에서 MIG(Multi-Instance GPU)와 MPS(Multi-Process Service)의 동시 실행 특성을 검증하기 위한 실험 저장소입니다. 실제 실행 원시 CSV, JSON metadata, tegrastats 로그 및 분석 문서를 함께 보관합니다.

## 실험 환경

| 항목 | 값 |
|---|---|
| 플랫폼 | NVIDIA Jetson AGX Thor |
| JetPack | 7.2.1 |
| GPU driver | 595.78 |
| CUDA | 13.2 |
| 컨테이너 이미지 | nvcr.io/nvidia/pytorch:25.08-py3 |
| 실험 MIG 구성 | 2g.0gb + 1g.0gb |
| 성능 측정 | CUDA event latency, synchronized wall time, tegrastats |

성능 수치는 Nsight Systems의 CUDA API 호출 시간이 아니라 CUDA event와 torch.cuda.synchronize()로 수집한 GPU 작업 시간 및 원시 CSV에서 계산합니다.

## 저장소 구성

| 경로 | 설명 |
|---|---|
| [mig_test_scenario.md](mig_test_scenario.md) | 실험 목적·가설·MPS 경량 worker 추가 실험 설계 |
| [scripts/](scripts) | 현재 실험 실행·측정·분석 스크립트 |
| [legacy_scripts/](legacy_scripts) | 초기 Docker/MIG 동작 검증 및 장치 매핑 참고용 스크립트 |
| [results/](results) | 실제 실행에서 생성된 원시 CSV, JSON metadata, tegrastats 로그 |
| [reports/actual_20260830/](reports/actual_20260830) | 결과 그래프, 실험별 해석, 데이터 객관성 감사 |

## 현재 실험 스크립트

| 스크립트 | 용도 | 상태 |
|---|---|---|
| scripts/setup_mig.sh | 재부팅 후 MIG 모드와 2g·1g 인스턴스 생성 | 실행 전 필요 |
| scripts/exp_1_1_mig.sh | 1g를 먼저, 2g를 뒤이어 실행하는 MIG 동시 workload | 단일 관측 결과 보존 |
| scripts/exp_1_1_nomig.sh | MIG를 끈 Non-MIG baseline | 마지막에 실행; MIG를 비활성화함 |
| scripts/exp_1_2_shared_mem.sh | 2g 단독과 1g+2g 동시 부하 비교 | 단일 관측 결과 보존 |
| scripts/exp_1_3_fault.sh | 1g OOM fault attempt | **실행 금지**: Jetson host OOM을 유발했음 |
| scripts/exp_3_1_mps.sh | 기존 heavy matmul MPS pilot | 기간 변수 오류가 있어 성능 결론에 사용하지 않음 |
| scripts/exp_3_1_mps_lightweight.sh | 4/8개의 경량 worker에서 MPS on/off 각 3회 반복 | 현재 권장 MPS 실험 |
| scripts/mps_microbatch_worker.py | 경량 MPS worker의 batch CUDA-event 측정 | 경량 MPS 실행기가 호출 |
| scripts/analyze_mps_lightweight.py | 경량 MPS raw CSV 집계·그래프 생성 | --out으로 보고서 경로 지정 가능 |
| scripts/generate_report.py | 기존 원시 결과의 전체 요약 생성기 | 20260830 결과 전용 |

모든 MIG workload는 1g 연결 작업을 2g보다 먼저 시작하도록 구성했습니다. setup_mig.sh는 multi-user.target으로 전환하므로 GUI/SSH 환경에 영향을 줄 수 있습니다.

## 빠른 시작

    cd /home/leo/MIG
    ./scripts/setup_mig.sh
    cd scripts
    ./exp_3_1_mps_lightweight.sh

setup_mig.sh는 재부팅 뒤 한 번 실행하며 sudo가 필요합니다. 경량 MPS 추가 실험은 약 6분 걸립니다.

실행 뒤 생성된 결과 디렉터리를 분석하려면 별도 보고서 경로를 사용합니다. Docker가 원시 결과를 root 소유로 만들 수 있기 때문입니다.

    python3 scripts/analyze_mps_lightweight.py \
      results/<timestamp>_mps_lightweight \
      --out reports/<timestamp>_mps_lightweight_analysis

## Jetson Docker와 MIG 장치 매핑

### 왜 수동 --device 매핑이 필요한가

Jetson의 NVIDIA Container Toolkit 경로에서는 환경 변수만으로 원하는 MIG instance를 안정적으로 선택하지 못한 사례가 있다. [NVIDIA/NemoClaw issue #9154](https://github.com/NVIDIA/NemoClaw/issues/9154)는 Jetson의 CSV 기반 장치 마운트 경로와 MIG device selection 제약을 보고한다. 이 저장소의 검증 스크립트는 그 제약을 우회하기 위해 runtime, MIG UUID 환경 변수, 그리고 필요한 NVIDIA device node를 함께 넘긴다.

- CUDA_VISIBLE_DEVICES의 MIG UUID는 컨테이너 안에서 CUDA가 선택할 logical device를 지정한다.
- --device는 호스트 device node를 컨테이너에 실제로 노출한다. CUDA_VISIBLE_DEVICES만으로 device node 접근 권한이 생기지는 않는다.
- nvidia0는 NVIDIA GPU driver의 GPU device interface이고, nvidiactl은 control interface다. CUDA 초기화와 driver query에 필요하다.
- nvidia-capXX node는 MIG capability/GI·CI 접근에 관계된 node다. 현재 실험에서는 1g에 cap21/cap22, 2g에 cap12/cap13을 사용했다.
- 컨테이너 안의 nvidia-smi -L은 노출된 device node와 runtime이 열거 가능한 GPU/MIG instance를 보여 주는 진단 수단이다. 한 instance만 보이는 것은 visibility 확인에는 유용하지만, 그 자체가 완전한 성능·fault isolation 증명은 아니다.

최소 visibility 확인 예시는 다음과 같다.

    sudo docker run --rm --runtime nvidia \
      --device /dev/nvidia-caps/nvidia-cap12 \
      --device /dev/nvidia-caps/nvidia-cap13 \
      --device /dev/nvidia0 \
      --device /dev/nvidiactl \
      nvcr.io/nvidia/pytorch:25.08-py3 nvidia-smi -L

이 명령은 컨테이너 생성 뒤 종료하고 nvidia-smi -L만 실행한다. PyTorch/CUDA workload를 실행하는 명령은 아니다.

### 안전 경고: cap 번호를 프로덕션에 하드코딩하지 말 것

nvidia-capXX의 실제 번호와 MIG instance의 대응은 현재 부팅·현재 MIG 생성 순서에서만 검증된 값이다. 재부팅, driver 초기화, MIG instance 재생성 뒤에도 같은 cap 번호가 같은 1g/2g instance를 가리킨다고 가정하면 안 된다.

따라서 다음 원칙을 지킨다.

1. MIG UUID는 매 실행 전 nvidia-smi -L로 동적으로 찾는다.
2. 수동 cap 매핑은 현재 부팅에서 최소 visibility test와 CUDA smoke test를 통과한 뒤에만 사용한다.
3. 1g과 2g의 cap mapping을 문서화하고 run 결과에 nvidia-smi -L 출력을 보존한다.
4. 이 방식은 NVIDIA Container Toolkit의 공식 지원 경로가 아닌 랩실 검증용 workaround로 취급한다. 차량·프로덕션 배포에서 cap 번호를 고정하지 않는다.

## Legacy 스크립트

legacy_scripts는 과거의 결과를 재현하거나 현재 실험 결과를 만드는 소스가 아니다. Docker runtime과 MIG device option의 최소 동작을 확인한 참고 자료다.

| 스크립트 | 실제 역할과 주의사항 |
|---|---|
| legacy_scripts/setup_mig.sh | multi-user.target 전환, persistence mode, MIG mode, profile 83(2g)·78(1g) 생성. 재부팅 뒤 사용하며 GUI/SSH에 영향을 줄 수 있다. 하드코딩 sudo password는 제거했다. |
| legacy_scripts/test_1g.sh | 1g에서 2000×2000 matmul을 warm-up한 뒤 입력 대기한다. 먼저 실행한 뒤 Enter 전 별도 터미널에서 test.sh를 시작한다. |
| legacy_scripts/test.sh | 2g에서 8000×8000 matmul을 무한 반복해 부하를 만든다. test_1g.sh가 먼저 준비된 뒤 실행한다. |
| legacy_scripts/sm_count_1g.sh | 기존 MIG instance를 파괴하고 profile 78의 1g만 생성한 뒤 PyTorch device property의 SM count를 출력한다. 기존 실험 구성을 파괴하므로 진단 전용이다. |
| legacy_scripts/mig_container_basic.sh | MIG instance와 수동 device mapping으로 컨테이너 PyTorch matmul이 동작하는지 보는 smoke test다. |
| legacy_scripts/mps_test_1g.sh | 1g 안에서 MPS daemon과 무거운 worker 두 개를 실행하는 초기 pilot이다. 무한 workload이므로 측정용으로 사용하지 않는다. |
| legacy_scripts/mps_test_2g.sh | 2g latency를 한 번 측정하는 초기 pilot이다. mps_test_1g.sh 부하와 짝으로 사용됐지만 현재 CSV 기반 실험으로 대체됐다. |

삭제한 legacy compare 스크립트는 Nsight CUDA API timing 중심의 과거 실험이었다. 또한 sm_count_2_instances.sh는 이름과 달리 2g를 세지 않고 1g만 조회하면서 모든 cap node를 노출하므로 제거했다.

## 결과 해석 범위

실제 결과와 원시 데이터의 신뢰성·한계는 [객관성 감사](reports/actual_20260830/objective_data_audit.md)에 정리돼 있다. 수치 해석은 항상 장비, MIG profile, workload, run 순서의 범위 안에서만 한다.
