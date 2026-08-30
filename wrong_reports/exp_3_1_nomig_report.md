# 실험 3-1 (Non-MIG): Bare-Metal MPS 기준선 보고서

## 1. 실험 개요

본 실험은 MIG를 비활성화한 상태에서 MPS on/off를 비교함으로써, 이전 MIG 실험에서 나타난 MPS의 제한적 효과가 MIG 때문인지, 아니면 Jetson 플랫폼 자체의 구조적 한계 때문인지 판별하기 위한 기준선 실험이다.

- 관련 스크립트: [MIG/scripts/exp_3_1_nomig_mps.sh](../scripts/exp_3_1_nomig_mps.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps](../experiments/fresh_20260830/exp_3_1_nomig_mps)

## 2. 실험 방식

- MIG 비활성화 상태로 복원한다.
- MPS 데몬을 실행한 뒤, 4개의 CUDA worker를 동시에 실행한다.
- MPS를 끈 상태에서도 동일한 워크로드를 실행한다.
- 비교 대상으로 사용되는 반복 횟수(iterations)를 획득한다.

## 3. 실행 상태 확인

다음 결과 파일이 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_a.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_b.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_c.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_c.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_d.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_on_nomig_d.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_a.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_b.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_c.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_c.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_d.nsys-rep](../experiments/fresh_20260830/exp_3_1_nomig_mps/mps_off_nomig_d.nsys-rep)

## 4. 실제 결과

실험 로그에서 확인된 반복 횟수는 다음과 같은 범위를 보여주었다.

- MPS on: 약 41,200~41,300
- MPS off: 약 42,000~42,200

차이는 2~3% 수준으로, 실질적 성능 gain으로 보기 어렵다.

## 5. 해석

이 실험의 중요성은 단순히 “MPS가 동작했다”가 아니라, “MIG가 아니어도 MPS의 이점이 크지 않다”는 점에 있다. 즉, 지금까지 관찰된 MPS 비효율은 MIG 파티션 문제보다는 Jetson 환경의 구조적 병목일 가능성이 더 크다.

특히 다음 해석이 타당하다.

- MPS 효과가 거의 없다는 점이 MIG 한정 현상이 아니라는 것
- 따라서 적어도 현재 실험 조건에서는 MPS가 큰 throughput benefit를 제공하지 않는다는 것
- 최종적 병목 원인은 공유 메모리/UMA 경합, 메모리 대역폭 제한, 또는 플랫폼 고유의 scheduling 한계일 가능성이 높다는 것

## 6. 결론

본 실험은 bare-metal non-MIG 기준선을 제공하는 실험으로서, MIG 비교 실험 결과를 해석할 때 핵심적인 기준점이 된다. 실제로 MPS on/off 차이가 거의 없다는 점은, 이전 MIG 실험에서 보인 결과가 특정 환경만의 현상이 아니라, 이 플랫폼 전반의 특성에 가깝다는 해석을 뒷받침한다.

## 7. 한계

- 이 결과는 “MPS가 완전히 무의미하다”는 결론을 내리기에는 부족하다.
- 다만, 현재 실험 조건에서 MPS는 실질적인 성능 향상을 만들지 못했다는 점은 충분히 확인되었다.
- 추가적으로 tegrastats, memory bandwidth, utilization 비교를 통해 구조적 병목을 더 정확히 검증할 필요가 있다.
