# Jetson Thor MIG/MPS 실험

Jetson AGX Thor에서 MIG(Multi-Instance GPU)와 MPS(Multi-Process Service)의 동시 실행 특성을 측정한 실험 저장소입니다. 이 저장소에는 실제로 수집한 원시 CSV·tegrastats 로그·발표용 보고서를 포함합니다.

## 실험 환경

| 항목 | 값 |
|---|---|
| 플랫폼 | NVIDIA Jetson AGX Thor |
| JetPack | 7.2.1 |
| GPU driver | 595.78 |
| CUDA | 13.2 |
| 컨테이너 이미지 | nvcr.io/nvidia/pytorch:25.08-py3 |
| MIG 구성 | 2g.0gb + 1g.0gb |
| 측정 | CUDA event latency, synchronized wall time, tegrastats |

성능 수치는 Nsight Systems의 CUDA API 호출 시간이 아니라 CUDA event와 torch.cuda.synchronize()로 수집한 GPU 작업 시간 및 원시 CSV에서 계산합니다.

## 저장소 구성

| 경로 | 설명 |
|---|---|
| [mig_test_scenario.md](mig_test_scenario.md) | 실험 목적·가설·MPS 경량 worker 추가 실험 설계 |
| [scripts/](scripts) | 현재 실험 실행·측정·분석 스크립트 |
| [results/](results) | 실제 실행에서 생성된 원시 CSV, JSON metadata, tegrastats 로그 |
| [reports/actual_20260830/](reports/actual_20260830) | 발표용 그래프, 실험별 보고서, 데이터 객관성 감사 |
| [legacy_scripts/](legacy_scripts) | 이전에 검증한 Docker/MIG 장치 옵션 참고용 스크립트. 새 실험의 결과에는 사용하지 않음 |

## 현재 스크립트

| 스크립트 | 용도 | 상태 |
|---|---|---|
| scripts/setup_mig.sh | 재부팅 후 MIG 모드와 2g·1g 인스턴스 생성 | 실행 전 필요 |
| scripts/exp_1_1_mig.sh | 1g 먼저, 2g를 뒤이어 실행하는 MIG 동시 workload | 단일 관측 결과 보존 |
| scripts/exp_1_1_nomig.sh | MIG를 끈 Non-MIG baseline | 마지막에 실행; MIG를 비활성화함 |
| scripts/exp_1_2_shared_mem.sh | 2g 단독과 1g+2g 동시 부하 비교 | 단일 관측 결과 보존 |
| scripts/exp_1_3_fault.sh | 1g OOM fault attempt | **실행 금지**: Jetson host OOM을 유발했음 |
| scripts/exp_3_1_mps.sh | 기존 heavy matmul MPS pilot | 기간 변수 오류가 있어 발표 성능 결론에 사용하지 않음 |
| scripts/exp_3_1_mps_lightweight.sh | 4/8개의 경량 worker에서 MPS on/off 각 3회 반복 | 현재 권장 MPS 실험 |
| scripts/mps_microbatch_worker.py | 경량 MPS worker의 batch CUDA-event 측정 | 위 스크립트가 호출 |
| scripts/analyze_mps_lightweight.py | 경량 MPS raw CSV 집계·그래프 생성 | --out으로 쓰기 가능한 보고서 경로 지정 가능 |
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

## 발표 자료와 결과 해석

[발표 준비 인덱스](reports/actual_20260830/README.md)에서 시작하세요. 발표 전에 반드시 [객관성 감사](reports/actual_20260830/objective_data_audit.md)를 읽어야 합니다.

- 실험 1-1: 이 실행에서 MIG aggregate throughput이 +7.8%였다는 관측만 사용합니다.
- 실험 1-2: 1g 동시 부하에서 2g 성능 저하가 관찰됐지만, DRAM 경합 원인이 증명된 것은 아닙니다.
- 실험 1-3: MIG fault isolation 성공이 아니라 host OOM 발생 증거입니다.
- 경량 MPS 실험: 4/8 worker, 조건별 3회 반복에서 MPS on throughput 이득이 관찰되지 않았습니다. 이 workload·1g slice 설정에만 해당합니다.

## Legacy 정리 원칙

legacy_scripts의 Docker/MIG 장치 매핑은 참고용으로 남겼습니다. 과거 Nsight 기반 *compare.sh 스크립트는 CUDA API timing 중심의 오래된 실험이라 제거했습니다.
