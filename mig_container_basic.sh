#!/bin/bash

# 에러 발생 시 즉시 중단
set -e

echo "=== 1. MIG 기본 설정 실행 중 ==="
cd ~/MIG
./setup_mig.sh

echo "=== 2. MIG 인스턴스 (1g: 78, 2g: 83) 생성 중 ==="
sudo nvidia-smi mig -cgi 78,83 -C

echo "=== 3. 커널 장치 노드 안정화 대기 (3초) ==="
sleep 3

echo "=== 4. Docker 컨테이너 실행 및 파이토치 연산 검증 ==="
sudo docker run --rm --runtime nvidia \
    --device /dev/nvidia-caps/nvidia-cap12 \
    --device /dev/nvidia-caps/nvidia-cap13 \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -c "import torch; print('\n=== 순수 수동 마운트 재검증 ==='); a = torch.randn(10000, 10000, device='cuda'); b = torch.randn(10000, 10000, device='cuda'); c = torch.matmul(a, b); print('검증 성공: 툴킷의 전체 개방 도움 없이, 4개의 수동 노드만으로 연산이 작동합니다.')"
