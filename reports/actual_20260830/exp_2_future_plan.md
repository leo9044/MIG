# 실험 2 — 향후 계획: Jetson SoC 병목 원인 좁히기

실험 2는 아직 수행하지 않았다. 실험 1-2에서 관찰된 간섭의 원인을 확정하지 않고, 가능한 원인을 좁히기 위한 후속 실험으로 둔다.

## 목적과 가설

MIG slice 간 성능 간섭이 DRAM/메모리 컨트롤러 공유, CPU 메모리 traffic, 전력·클럭 제한 중 무엇과 더 관련 있는지 관찰한다.

## 계획 방법

1. 2g 단독 matmul을 기준선으로 30초씩 최소 3회 실행한다.
2. 비교 조건 A: 1g matmul을 동시 실행한다.
3. 비교 조건 B: GPU 동시 부하는 없이 호스트 CPU memory stress만 부여한다.
4. 모든 조건에 CUDA-event CSV, tegrastats(100 ms), Nsight Systems kernel/NVTX timeline을 수집한다.
5. 2g throughput·p95 latency와 EMC/메모리 사용·GPU clock·온도를 같은 시간축에서 비교한다.

## 판정 기준

- 1g 부하와 CPU memory stress 모두에서 2g 성능 저하와 메모리 지표 변화가 재현되면 공유 메모리 경로 가설이 강해진다.
- 성능 저하가 온도/전력/클럭 변화와 함께 나타나면 thermal/power 제한 가능성을 분리해 논의한다.
- 이 실험도 인과 증명보다 가설의 우선순위를 정하는 자료로 발표한다.
