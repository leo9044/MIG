#!/bin/bash
set -e

echo "=== 1g 파티션 Docker 컨테이너 지속 부하 시작 ==="
sudo docker run --rm -it --runtime nvidia \
    --device /dev/nvidia-caps/nvidia-cap21 \
    --device /dev/nvidia-caps/nvidia-cap22 \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -c "
import torch, time
print('\n=== [1g 파티션] MIG 격리 증명용 지속 부하 테스트 시작 ===')
# 1GB 메모리에 맞게 행렬 크기를 5000x5000으로 약간 줄여서 안전하게 구동합니다.
a = torch.randn(5000, 5000, device='cuda')
b = torch.randn(5000, 5000, device='cuda')
count = 0
while True:
    c = torch.matmul(a, b)
    count += 1
    print(f'\r[1g] 현재 부하 연산 횟수: {count}', end='', flush=True)
    time.sleep(0.01)
"
