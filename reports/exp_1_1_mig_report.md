# 실험 1-1: MIG 기반 기준선 보고서

## 1. 실험 개요

본 실험은 MIG 환경에서 동일한 GPU 연산 워크로드를 각각 1g 및 2g 파티션에서 실행할 때, 파티션 단독 성능이 어떻게 나타나는지를 확인하는 것을 목표로 한다.

- 관련 스크립트: [MIG/scripts/exp_1_1_mig.sh](../scripts/exp_1_1_mig.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_1_1/mig](../experiments/fresh_20260830/exp_1_1/mig)

## 2. 실험 방식

- MIG 모드가 활성화되어 있는 상태에서 1g/2g 인스턴스를 확인한다.
- 각 파티션에 동일한 CUDA 행렬 연산 워크로드를 수행한다.
- Nsight Systems로 CUDA trace를 수집한다.
- 결과 파일은 `.nsys-rep` 형식으로 저장된다.

## 3. 실행 상태 확인

실험 결과 디렉터리에 다음 파일이 존재한다는 점을 확인했다.

- [MIG/experiments/fresh_20260830/exp_1_1/mig/mig_1g.nsys-rep](../experiments/fresh_20260830/exp_1_1/mig/mig_1g.nsys-rep)
- [MIG/experiments/fresh_20260830/exp_1_1/mig/mig_2g.nsys-rep](../experiments/fresh_20260830/exp_1_1/mig/mig_2g.nsys-rep)

이는 해당 스크립트가 정상적으로 실행되어 결과를 남겼음을 의미한다.

## 4. 결과 해석

이 실험은 단일 파티션 내부에서의 baseline 성능을 확보하는 용도로 중요하다. 본 실험이 의미하는 것은 다음과 같다.

- 1g 및 2g slice 각각이 독립적으로 CUDA workload를 수행할 수 있는가
- 파티션 크기별 성능 편차가 존재하는가
- 이후의 MPS 비교와 fault-isolation 비교를 위한 기준점이 되는가

다만 본 보고서에서는 원시 `.nsys-rep` 파일 내용을 직접 열어 수치값을 계산하지 않았기 때문에, 성능 숫자 자체를 단정적으로 서술하지 않는다. 해당 파일은 이후 성능 비교용 baseline으로 활용해야 한다.

## 5. 결론

본 실험은 MIG 환경에서의 기준선을 확보하는 실험으로서, 이후의 MPS 및 비-MIG 비교를 가능하게 만드는 핵심 전제 조건을 형성한다. 실험 파일이 정상적으로 생성되었고, 결과 데이터는 이후 비교 분석의 기준점으로 사용 가능하다.

## 6. 한계

- 본 보고서는 결과 파일의 정량적 수치 분석을 포함하지 않고, 결과 파일 생성 여부와 실험 적합성 중심으로 정리한다.
- throughput, latency, p95 값 비교는 별도의 Nsight 분석 단계에서 수행하는 것이 가장 정확하다.
