# 실험 1-1: Non-MIG 기준선 보고서

## 1. 실험 개요

본 실험은 MIG를 비활성화한 상태에서 GPU 전체를 단일 자원으로 사용했을 때의 baseline 성능을 측정한다. 이 결과는 MIG 환경과의 비교를 위해 필수적인 기준선이다.

- 관련 스크립트: [MIG/scripts/exp_1_1_nomig.sh](../scripts/exp_1_1_nomig.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_1_1/nomig](../experiments/fresh_20260830/exp_1_1/nomig)

## 2. 실험 방식

- `nvidia-smi -mig 0`를 사용해 MIG 기능을 비활성화한다.
- 전체 GPU를 단일 CUDA device처럼 사용한다.
- 동일한 행렬 곱셈 workload를 두 번 반복 실행한다.
- Nsight Systems로 trace를 수집한다.

## 3. 실행 상태 확인

결과 디렉터리에 다음 파일이 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_1_1/nomig/nomig_workload_a.nsys-rep](../experiments/fresh_20260830/exp_1_1/nomig/nomig_workload_a.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_1_1/nomig/nomig_workload_b.nsys-rep](../experiments/fresh_20260830/exp_1_1/nomig/nomig_workload_b.nsys-rep)

이것은 실험이 정상적으로 수행되었고, 비-MIG baseline 자료가 확보되었음을 의미한다.

## 4. 의미

본 실험의 의미는 단순히 “GPU 전체에서 돌리면 빨라지냐”를 보는 것이 아니라, MIG 환경의 성능 감소가 실제로 파티션 구조 때문에 발생하는지, 아니면 전체 GPU baseline 자체가 다른 패턴을 보이는지 비교하는 데 있다.

즉:

- non-MIG baseline은 MIG 실험의 상대 기준점이다.
- 본 데이터와 MIG 데이터 비교를 통해, 파티션 분할이 실제로 성능을 악화시키는지 판단할 수 있다.

## 5. 결론

이 실험은 MIG 실험의 측정 기준선을 제공하는 실험으로서, 이후 1-3, 2, 3-1 등 다른 실험의 해석을 위한 필수 자료로 사용된다. 결과 파일이 정상 생성되었고, 실험 자료로서의 가치가 있다.

## 6. 한계

- 본 보고서는 비-MIG baseline 자체의 정량적 수치를 직접 보고하지 않는다.
- 한 번의 항목이 아니라 반복 측정 둘이므로, 결과는 비교 분석 단계에서 요약해야 한다.
