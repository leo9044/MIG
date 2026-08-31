# 실험 문서 인덱스

## 읽기 순서

1. [객관성 감사](objective_data_audit.md) — 각 결과에서 데이터가 뒷받침하는 범위와 한계를 정리한다.
2. [전체 요약](experiment_report.md) — 실제 CSV에서 계산한 전체 수치와 그래프를 모은다.
3. [실험 1-1](exp_1_1_report.md) — MIG/Non-MIG 동시 실행 비교.
4. [실험 1-2](exp_1_2_report.md) — 1g 동시 부하가 2g에 주는 간섭.
5. [실험 3-1 추가](exp_3_1_lightweight_report.md) — 경량 다중 worker MPS 재검증.
6. [실험 1-3](exp_1_3_report.md) — fault injection에서 호스트 OOM이 확인된 이유.
7. [실험 2 향후 계획](exp_2_future_plan.md) — 원인 규명을 위한 후속 실험 설계.

## 해석 원칙

- 수치는 CUDA API 시간이 아니라 반복별 CUDA event latency와 실제 관측 시간에서 계산했다.
- 모든 조건은 단일 또는 제한된 반복 실험이다. 따라서 경향을 관찰한 것이며 일반적 인과관계를 증명한 것은 아니다.
- 실험 1-2의 성능 저하는 공유 DRAM/SoC 자원 경합 가설과 일치하지만, 원인을 확정하지는 않는다.
- 실험 1-3은 MIG fault isolation 성공의 증거가 아니다. 무한 CUDA 할당이 Jetson 호스트 OOM을 유발한 증거다.
- 경량 MPS 추가 실험에서는 4·8 worker, 각 3회 반복에서 MPS on의 처리량 이득이 관찰되지 않았다. kernel overlap trace 검증은 남아 있다.
