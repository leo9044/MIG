# 실험 1-3: Fault Isolation 보고서

## 1. 실험 개요

본 실험은 MIG 환경에서 2g 파티션의 작업이 활성 상태일 때, 1g 파티션에서 OOM 또는 fault를 유발하였을 때 2g 파티션이 영향을 받는지를 검증하는 실험이다.

- 관련 스크립트: [MIG/scripts/exp_1_3_fault.sh](../scripts/exp_1_3_fault.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_1_3](../experiments/fresh_20260830/exp_1_3)

## 2. 실험 방식

1. 1g 및 2g MIG 인스턴스를 준비한다.
2. 2g 파티션에 GPU 연산 workload를 백그라운드로 실행한다.
3. 2g workload가 안정화된 뒤, 1g 파티션에서 큰 텐서를 할당해 OOM 조건을 유도한다.
4. 2g workload가 유지되는지, 종료되거나 변형되는지를 관찰한다.

이 방식은 “fault가 다른 slice로 전파되는지”를 검증하는 구조로 설계되었다.

## 3. 실행 상태 확인

결과 파일이 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_1_3/mig_2g_fault_isolation.nsys-rep](../experiments/fresh_20260830/exp_1_3/mig_2g_fault_isolation.nsys-rep)

## 4. 결과 해석

이 실험은 MIG의 핵심 특징인 isolation, 특히 fault propagation 방지 여부를 확인하는 목적이 있다. 즉, 한 slice에서의 장애가 다른 slice에 영향을 미치지 않는지 보는 것이 핵심이다.

스크립트 구조상 fault injection은 2g workload가 이미 실행 중인 상태에서 수행되도록 설계되었으며, 이는 실제 fault-isolation 검증 시나리오와 일치한다. 따라서 이 실험은 아래 항목을 검증하는 기준점이 된다.

- 1g fault가 2g slice의 연산을 중단시키는가
- 2g workload가 계속 수행되는가
- fault가 MIG 경계를 넘어서 확산되는가

## 5. 결론

본 실험은 MIG의 fault 격리 기능을 확인하는 데 필요한 실험이다. 결과 파일이 존재하며, 스크립트의 의도대로 2g workload를 유지한 상태에서 1g fault를 유도하는 구조로 설계되었다.

다만, 본 보고서는 결과 파일을 직접 열어 정량적 기준을 계산하지 않았으므로, 최종적인 fault isolation 판정은 Nsight trace 분석을 추가로 수행해야 정확하다.

## 6. 한계

- fault를 “유발했다”는 사실은 스크립트 흐름상 확인되지만, 2g workload가 실제로 완전히 유지되었는지 여부는 trace 수준의 분석이 필요하다.
- 단순히 OOM 코드가 실행되었다는 것만으로 fault isolation이 확보되었다고 단정할 수는 없다.
