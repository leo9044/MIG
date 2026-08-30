# 실험 2: Shared-Memory 경합 가설 검증 보고서

## 1. 실험 개요

본 실험은 Jetson 환경에서 MIG가 compute slice를 나누더라도, 최종적으로는 공유 메모리 경로와 DRAM/UMA 경합이 병목으로 작용할 가능성이 있다는 가설을 검증하는 실험이다.

- 관련 스크립트: [MIG/scripts/exp_2_shared_mem.sh](../scripts/exp_2_shared_mem.sh)
- 결과 디렉터리: [MIG/experiments/fresh_20260830/exp_2](../experiments/fresh_20260830/exp_2)

## 2. 실험 방식

- 호스트 수준에서 CPU 메모리 stress를 발생시킨다.
- GPU 측에서는 1g MIG slice에서 동일한 행렬 곱셈 workload를 실행한다.
- 이때 메모리 경합이 throughput 또는 latency에 어떤 영향을 주는지를 확인한다.

핵심 목적은 다음과 같다.

- MIG slice 내부에서도 실제로 메모리 경합이 발생하는가
- 성능 저하가 compute 단위보다 memory path 관점에서 설명되는가
- MPS 효과가 제한되는 배경으로 공유 메모리 병목이 작용하는가

## 3. 실행 상태 확인

결과 파일이 생성되었음을 확인했다.

- [MIG/experiments/fresh_20260830/exp_2/shared_mem_1g_stress.nsys-rep](../experiments/fresh_20260830/exp_2/shared_mem_1g_stress.nsys-rep)

## 4. 결과 해석

이 실험은 “MIG가 문제인가?”보다 “그래도 실제 병목이 메모리 경로에 있지 않는가?”를 확인하는 데 의미가 있다. Jetson Thor는 SoC 기반 구조이므로 GPU, CPU, memory controller가 공유 메모리 경로를 이용할 가능성이 크다. 따라서 단순히 GPU만 보아서는 실제 병목을 놓칠 수 있다.

즉, 이 실험은 다음 질문에 답하는 데 초점을 둔다.

- 행렬 연산을 수행하는 GPU가 격리되더라도, 전체 시스템 메모리 경로는 공유되는가
- 공유 자원 경합이 MIG 성능 저하를 설명하는가
- MPS보다 더 근본적인 병목은 어떤 메모리 구조에서 발생하는가

## 5. 결론

본 실험은 MIG의 compute isolation이 충분히 보장되더라도, 최종적인 performance bottleneck이 memory path에서 발생할 수 있다는 점을 검증하는 실험이다. 결과 파일은 존재하며, 아키텍처적 해석을 위한 중요 자료이다.

## 6. 한계

- 이 실험은 “가설 검증” 수준의 실험이다.
- 실제로 메모리 대역폭 병목이 정량적으로 유의미했는지 확인하려면 추가적인 tegrastats, GPU utilization, DRAM utilization 비교가 필요하다.
- 결과의 해석은 trace 분석을 보완해야 정확하다.
