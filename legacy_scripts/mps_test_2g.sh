#!/bin/bash
set -e

echo "=== 1. [VLA] 2g 타겟 MIG UUID 동적 추출 ==="
MIG_UUID_2G=$(nvidia-smi -L | grep "MIG 2g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')

echo "=== 2. [VLA / 2g 파티션] 엄격한 격리 기반 레이턴시 측정 시작 ==="
sudo docker run --rm -it --runtime nvidia \
    --cpuset-cpus="0-1" \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$MIG_UUID_2G" \
    --device /dev/nvidia-caps/nvidia-cap12 \
    --device /dev/nvidia-caps/nvidia-cap13 \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -u -c "
import torch

print('\n=== [VLA / 2g] 순수 GPU 하드웨어 레이턴시 측정 시작 ===')

# 1. 텐서 초기화
a = torch.randn(2000, 2000, device='cuda')
b = torch.randn(2000, 2000, device='cuda')

# 2. CUDA 하드웨어 타이머 객체 생성
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 3. 웜업
print('GPU 웜업 중...')
for _ in range(10):
    _ = torch.matmul(a, b)
torch.cuda.synchronize()

# 4. 순수 하드웨어 레이턴시 측정 (이미 1g 부하가 돌고 있으므로 대기 없이 바로 시작)
print('\n[✅ 초기화 완료] 바로 하드웨어 레이턴시 측정을 시작합니다...')
start_event.record()

for _ in range(100):
    c = torch.matmul(a, b)

end_event.record()
torch.cuda.synchronize()

# 5. 결과 출력
elapsed_time_ms = start_event.elapsed_time(end_event)
print('-' * 50)
print(f'100회 연산 순수 GPU 소요 시간: {elapsed_time_ms:.3f} ms')
print(f'1회 연산 평균 레이턴시: {elapsed_time_ms / 100:.3f} ms')
print('-' * 50)
"
