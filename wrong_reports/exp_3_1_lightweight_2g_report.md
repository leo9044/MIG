# 실험 3-1 (Lightweight on 2g): 2g MIG 내부 MPS 비교 보고서

## 1. 실험 개요

본 실험은 2g MIG slice에서 동일한 경량 멀티 워커 workload를 실행할 때, MPS on/off 상태의 차이를 비교한다. 1g 실험이 거의 gain이 없었다는 점을 보완하기 위해, 더 큰 slice에서도 동일한 현상이 재현되는지 확인하는 것이 목적이다.

- 관련 스크립트: [MIG/scripts/exp_3_1_mps_lightweight_2g.sh](../scripts/exp_3_1_mps_lightweight_2g.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_3_1_light_2g](../experiments/fresh_20260830/exp_3_1_light_2g)

## 2. 실험 방식

- 2g MIG 인스턴스를 준비한다.
- MPS를 활성화한 상태와 비활성화한 상태에서 네 개의 경량 CUDA worker를 동시에 실행한다.
- 각 workload는 동일한 행렬 연산 패턴을 수행하고, 반복 횟수를 집계한다.

## 3. 실행 상태 확인

다음 결과 파일이 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_on_a.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_on_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_on_b.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_on_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_on_c.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_on_c.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_on_d.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_on_d.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_off_a.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_off_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_off_b.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_off_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_off_c.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_off_c.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_light_2g/mps_off_d.nsys-rep](../experiments/fresh_20260830/exp_3_1_light_2g/mps_off_d.nsys-rep)

## 4. 결과 해석

로그 기준으로 MPS on과 off 간 반복 횟수 차이는 거의 없었고, 1g 실험과 동일하게 “MPS가 동작은 하지만 실제 이득이 거의 없다”는 해석이 유지된다.

이는 다음을 의미한다.

- 2g slice에서도 MPS gain이 크지 않음
- 단순히 1g 파티션에서만 발생하는 문제가 아님
- MPS의 효과가 실제로 제한된 아키텍처적 특성 때문일 가능성이 높음

특히 2g slice는 더 큰 리소스를 제공하는 조건인데도 차이가 거의 없었다. 이는 MPS가 이 플랫폼에서 강력한 throughput enhancer로 작동하지 않음을 보여준다.

## 5. 결론

본 실험은 “MPS의 사소한 이점이 1g에서만 나타나는 현상인가?”라는 의문을 해결하는 데 도움을 준다. 결과적으로 2g에서도 큰 차이가 없었으므로, MPS gain의 부재는 1g에 국한된 문제가 아니라 플랫폼 전체의 특성으로 이해하는 것이 더 타당하다.

## 6. 한계

- 최종 판정은 trace 레벨 분석에서 이루어져야 한다.
- 이 결과만으로는 메모리 대역폭 병목의 정확한 원인을 확정할 수 없다.
- 다만, MPS가 실질적 성능 향상을 내지 않는다는 근거는 충분히 확보되었다.
