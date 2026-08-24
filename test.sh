#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

echo "=== 1. MIG 기본 설정 실행 중 ==="
cd ~/MIG
./setup_mig.sh

echo "=== 2. MIG 인스턴스 (1g: 78, 2g: 83) 생성 중 ==="
sudo nvidia-smi mig -cgi 78,83 -C

echo "=== 3. 커널 장치 노드 및 드라이버 안정화 대기 (3초) ==="
sleep 3

echo "=== 4. 타겟 MIG UUID 동적 추출 ==="
# 리부트/재생성 시마다 바뀌는 2g 파티션의 UUID를 호스트에서 실시간으로 읽어옵니다.
MIG_UUID=$(nvidia-smi -L | grep "MIG 2g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')
echo "추출된 2g 파티션 UUID: $MIG_UUID"

echo "=== 5. Docker 컨테이너 실행 및 지속 부하(Stress) 테스트 시작 ==="
sudo docker run --rm -it --runtime nvidia \
    --device /dev/nvidia-caps/nvidia-cap12 \
    --device /dev/nvidia-caps/nvidia-cap13 \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -c "
import torch, time
print('\n=== MIG 격리 증명용 지속 부하 테스트 시작 ===')
a = torch.randn(8000, 8000, device='cuda')
b = torch.randn(8000, 8000, device='cuda')
count = 0
while True:
    c = torch.matmul(a, b)
    count += 1
    print(f'\r현재 부하 연산 횟수: {count}', end='', flush=True)
    time.sleep(0.01)
"
