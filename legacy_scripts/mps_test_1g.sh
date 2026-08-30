#!/bin/bash
set -e

echo "=== 1. [VLM] 1g 타겟 MIG UUID 동적 추출 ==="
MIG_UUID_1G=$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')

echo "=== 2. [VLM / 1g 파티션] MPS 기반 다중 프로세스 지속 부하 시작 ==="
sudo docker run --rm -it --runtime nvidia \
    --cpuset-cpus="2-11" \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$MIG_UUID_1G" \
    --device /dev/nvidia-caps/nvidia-cap21 \
    --device /dev/nvidia-caps/nvidia-cap22 \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    bash -c "
echo '[VLM / 1g] MPS 데몬 세팅 및 시작 중...'
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p \$CUDA_MPS_PIPE_DIRECTORY \$CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

# 👇 [핵심 추가] MPS 서버가 완전히 초기화될 때까지 2초 대기
echo 'MPS 서버 초기화 대기 중 (2초)...'
sleep 2

echo '[VLM / 1g] 2개의 프로세스를 MPS 상에서 동시 실행합니다.'

# 워크로드 A (백그라운드 실행, -u 옵션으로 즉시 출력)
python3 -u -c \"
import torch, time
a = torch.randn(2000, 2000, device='cuda')
count = 0
while True:
    c = torch.matmul(a, a)
    count += 1
    if count % 1000 == 0:
        print(f'\r[Process A] 연산 횟수: {count}   ', end='', flush=True)
\" &

# 워크로드 B (백그라운드 실행, -u 옵션으로 즉시 출력)
python3 -u -c \"
import torch, time
b = torch.randn(2000, 2000, device='cuda')
count = 0
while True:
    c = torch.matmul(b, b)
    count += 1
    if count % 1000 == 0:
        print(f'\n[Process B] 연산 횟수: {count}   ', end='', flush=True)
\" &

# 백그라운드 프로세스가 종료될 때까지 대기
wait
"
