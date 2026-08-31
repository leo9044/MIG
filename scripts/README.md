# 실행 스크립트

성능 결과는 `benchmark.py`가 기록한 반복별 CUDA-event latency와 동기화된 wall-clock 시간에서 계산한다. Nsight trace는 필요할 때의 정성적 보조 증거이며, CUDA API 시간은 애플리케이션 성능 수치로 사용하지 않는다.

## 실행 순서와 안전 주의

재부팅 뒤에는 먼저 `./setup_mig.sh`를 한 번 실행한다. MIG 조건에서 1g 컨테이너가 먼저 장치를 잡아야 하므로, 1g을 포함한 스크립트는 내부적으로 1g을 먼저 시작한다. Non-MIG 기준선은 MIG를 해제하므로 모든 MIG 실험 뒤에 실행해야 하며, 이후 MIG 실험 전에는 `setup_mig.sh`를 다시 실행한다.

`exp_1_3_fault.sh`는 Jetson UMA 호스트 OOM을 유발한 이력이 있다. 현재 보존된 결과는 이 방식이 안전한 fault-isolation 검증 방법이 아님을 보여 주므로, 운영 중인 장비에서 재실행하지 않는다.

```bash
./setup_mig.sh
./exp_1_1_mig.sh 30                 # 1-1: 1g + 2g 동시 실행
./exp_1_2_shared_mem.sh 30          # 1-2: 2g 단독과 1g+2g 비교
./exp_3_1_mps_lightweight.sh 20     # 3-1 추가: 경량 worker MPS on/off
./exp_1_1_nomig.sh 30               # 1-1 Non-MIG 기준선: 반드시 마지막
./analyze_results.py ../results
```

각 실행은 타임스탬프가 붙은 원시 결과 디렉터리를 만들며, CSV·JSON 메타데이터·`tegrastats.log`·관측된 MIG topology를 저장한다.

## 파일 역할

- `setup_mig.sh`: 재부팅 후 MIG를 활성화하고 실험용 2g+1g 구성을 만든다.
- `run_experiment.sh`: Docker 장치 연결과 결과 디렉터리 생성을 공통으로 처리하는 내부 실행기다.
- `benchmark.py`: matmul 반복의 CUDA-event latency와 wall time을 CSV로 기록한다.
- `mps_microbatch_worker.py`: 경량 다중-worker MPS 재검증용 worker다.
- `analyze_results.py`: 원시 CSV에서 요약 수치와 그래프를 생성한다.
- `generate_report.py`: 보존된 원시 결과로 전체 요약 보고서를 다시 생성한다.
