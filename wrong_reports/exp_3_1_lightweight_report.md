# 실험 3-1 (Lightweight): 1g MIG 내부 MPS 비교 보고서

## 1. 실험 개요

본 실험은 1g MIG slice에서 더 가벼운 작업을 동시에 4개 실행할 때, MPS on/off 상태의 차이를 비교한다. 이전의 무거운 workload보다 경량 워커를 사용함으로써, MPS가 실제로 어떤 역할을 하는지 더 명확하게 확인하고자 했다.

- 관련 스크립트: [MIG/scripts/exp_3_1_mps_lightweight.sh](../scripts/exp_3_1_mps_lightweight.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_3_1_light](../experiments/fresh_20260830/exp_3_1_light)

## 2. 실험 방식

- 1g MIG 인스턴스를 확보한다.
- MPS를 활성화한 상태와 비활성화한 상태에서 네 개의 경량 CUDA worker를 실행한다.
- 각 worker는 동일한 행렬 연산을 반복 수행하며, 수행 횟수를 집계한다.
- 각 조건에서 `iterations` 값을 비교한다.

## 3. 실행 상태 확인

결과 파일이 실제로 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_on_a.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_on_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_on_b.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_on_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_on_c.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_on_c.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_on_d.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_on_d.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_off_a.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_off_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_off_b.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_off_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_off_c.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_off_c.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light/mps_off_d.nsys-rep](../experiments/fresh_20260830/exp_3_1_light/mps_off_d.nsys-rep)

## 4. 결과 해석

실험 로그에서 얻은 반복 횟수는 다음과 같은 패턴을 보였다.

- MPS on: 약 34k 수준
- MPS off: 약 34k 수준

즉, 거의 차이가 없었다. 경량 workload에서도 MPS가 유의미한 throughput gain을 만들지 못했다.

이 결과는 중요한 점을 보여준다.

- 무거운 workload가 아니라 가벼운 동시 workload에서도 MPS 이점이 거의 없다는 것
- 따라서 1g slice에서 MPS가 “일반적으로 안 되는 것”이 아니라, 이 아키텍처에서 성능 gain이 매우 제한적이라는 점
- MPS가 동작 자체는 했지만, 실제 병목 해소 효과는 거의 없었다는 점

## 5. 결론

본 실험은 이전 실험의 결과를 강화하는 역할을 한다. 단순히 workload가 무거워서 그렇다는 해석이 아니라, 가벼운 멀티 워커에서도 MPS gain이 거의 존재하지 않았기 때문에, 1g MIG 환경에서 MPS의 실질적 유효성이 낮다는 결론이 더 설득력이 있다.

## 6. 한계

- 이 실험은 “MPS가 동작한다”를 확인하는 수준을 넘어서 실질적 gain을 검증하는 데 초점을 두고 있지만, 최종적인 성능 수치 비교는 더 정밀한 Nsight 분석이 필요하다.
- 전체적인 병목 원인은 아직 메모리 경로 또는 UMA 구조를 보완적으로 검증해야 한다.
