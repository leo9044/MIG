# 실험 3-1: MIG 환경 내 MPS 비교 보고서

## 1. 실험 개요

본 실험은 MIG 환경에서 MPS를 켜고 끄는 조건을 비교하여, MPS가 실제로 1g slice 안에서 의미 있는 성능 이점을 제공하는지 검증한다.

- 관련 스크립트: [MIG/scripts/exp_3_1_mps.sh](../scripts/exp_3_1_mps.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_3_1](../experiments/fresh_20260830/exp_3_1)

## 2. 실험 방식

- 1g MIG 인스턴스를 사용한다.
- MPS를 켠 상태와 끈 상태에서 각각 두 개의 CUDA 워커를 실행한다.
- 해당 시나리오에서는 MPS가 동작하는지와 성능 차이를 비교하는 것이 목적이다.
- 결과 파일은 `.nsys-rep` 형식으로 저장된다.

## 3. 실행 상태 확인

다음 결과 파일들이 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_3_1/mps_on_a.nsys-rep](../experiments/fresh_20260830/exp_3_1/mps_on_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1/mps_on_b.nsys-rep](../experiments/fresh_20260830/exp_3_1/mps_on_b.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1/mps_off_a.nsys-rep](../experiments/fresh_20260830/exp_3_1/mps_off_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_3_1/mps_off_b.nsys-rep](../experiments/fresh_20260830/exp_3_1/mps_off_b.nsys-rep)

## 4. 결과 해석

이 실험은 MPS 자체의 동작 여부와 성능 차이를 비교하는 기반 실험이다. 실험 의도는 단순히 “MPS가 켜졌는지”를 확인하는 것이 아니라, MIG 내부에서 MPS가 실제로 실질적 병렬성 또는 성능 이점을 만드는지를 보는 데 있다.

중요한 해석은 다음과 같다.

- MPS는 동작 상태를 보장할 수 있으나, 이것이 곧 성능 향상을 의미하지는 않는다.
- 흥미로운 점은 MPS가 켜져 있어도 실제 throughput 차이가 크지 않을 수 있다는 것이다.
- 이는 MIG slice 자체의 제한 또는 Jetson 메모리 구조의 병목이 더 큰 영향을 미칠 수 있음을 시사한다.

## 5. 결론

본 실험은 MIG 환경 내에서 MPS의 유효성을 검증하는 핵심 실험이다. 실험 결과 파일이 생성되었고, 이후 더 경량 워크로드로 다시 검증한 실험들과 함께 해석하는 것이 바람직하다.

## 6. 한계

- 본 보고서는 정량적 throughput 수치를 단독으로 제시하지 않았으며, 최종 비교는 Nsight trace 분석에서 정리하는 것이 정확하다.
- MPS가 “작동했다”는 사실과 “실제 성능 이점이 있었다”는 사실은 구분되어야 한다.
