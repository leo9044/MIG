#!/bin/bash
set -e

echo "=== 0. MIG 기본 설정 실행 중 ==="
cd ~/MIG
./setup_mig.sh
sudo nvidia-smi mig -cgi 78,83 -C
sleep 3

echo "=== 1. 1g 타겟 MIG UUID 동적 추출 ==="
MIG_UUID_1G=$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')

echo "=== 2. [1g 파티션] Docker 지속 부하 시작 ==="
sudo docker run --rm -it --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$MIG_UUID_1G" \
    --device /dev/nvidia-caps/nvidia-cap21 \
    --device /dev/nvidia-caps/nvidia-cap22 \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -c "
import torch, time
print('\n=== [1g 파티션] MIG 격리 증명용 지속 부하 테스트 시작 ===')
a = torch.randn(5000, 5000, device='cuda')
b = torch.randn(5000, 5000, device='cuda')
count = 0
while True:
    c = torch.matmul(a, b)
    count += 1
    print(f'\r[1g] 현재 부하 연산 횟수: {count}', end='', flush=True)
    time.sleep(0.01)
"
