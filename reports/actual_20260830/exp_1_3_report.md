# 실험 1-3 — Fault isolation 시도: 호스트 OOM 증거

## 목적

원래 목적은 1g MIG 인스턴스에서 OOM을 유발했을 때 2g 작업이 계속 실행되는지 확인하는 것이었다.

## 방법

1. 1g 컨테이너를 먼저 시작했다.
2. 8초 뒤 1g에서 4096×4096 FP32 CUDA tensor(약 64 MiB)를 반복 할당하도록 했다.
3. 2초 뒤 2g에서 2000×2000 matmul을 32초간 반복하면서 매 반복 CUDA-event latency를 CSV에 저장했다.
4. 같은 구간에 호스트 `tegrastats`와 systemd journal을 확인했다.

## 실제로 확인된 증거

- 보존 원시 결과: [`results/20260830_181838_fault`](../../results/20260830_181838_fault)
- 2g CSV: `fault_target.csv`, 56,966회 완료
- 2g 요약: 평균 0.450 ms, p95 0.409 ms, 최대 308.332 ms
- 1g에서 기대한 `expected_cuda_oom` JSON은 생성되지 않았다.
- systemd journal은 18:18:39에 1g fault 컨테이너 시작을 기록하고, 18:19:16에 `wireplumber.service` 프로세스가 **OOM killer**에 의해 종료됐음을 기록한다.
- 캡처용 증거 파일: [exp_1_3_oom_evidence.txt](exp_1_3_oom_evidence.txt)

## 결론

증명된 것은 **Jetson 호스트에서 OOM killer가 동작했다**는 사실이다. 증명되지 않은 것은 “1g CUDA OOM이 2g에 완전히 격리됐다”는 주장이다. 왜냐하면 1g 컨테이너의 정상적인 CUDA OOM 완료 기록이 없고, 호스트 서비스가 OOM killer로 종료됐기 때문이다.

따라서 이 실험의 발표 자료는 성능 그래프가 아니라 journal 증거를 캡처해 다음처럼 제시하는 것이 정확하다.

> “무한 CUDA 할당은 MIG slice 내부 실패로 제한되지 않고 Jetson UMA 호스트 OOM을 유발했습니다. 따라서 이 방식은 fault isolation 성공 검증이 아니라 안전하지 않은 fault injection 방식의 확인입니다.”

## 캡처 명령

```bash
journalctl --since '2026-08-30 18:18:30' --until '2026-08-30 18:20:10' --no-pager \
  | rg -i 'fault_oom|OOM killer|oom-kill|docker run'
```

이 출력에서 1g `fault_oom` Docker 명령과 뒤따르는 `A process ... killed by the OOM killer` 줄이 한 화면에 보이게 캡처한다.
