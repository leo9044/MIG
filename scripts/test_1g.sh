#!/bin/bash
set -e

echo "=== 1. 1g 타겟 MIG UUID 동적 추출 ==="
MIG_UUID_1G=$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')

echo "=== 2. [1g 파티션] Docker 하드웨어 레이턴시 측정 시작 ==="
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
import torch

print('\n=== [1g 파티션] 순수 GPU 하드웨어 레이턴시 측정 시작 ===')

# 1. 텐서 초기화 (1g 메모리 한계 초과를 방지하기 위해 2000x2000 사용)
a = torch.randn(2000, 2000, device='cuda')
b = torch.randn(2000, 2000, device='cuda')

# 2. CUDA 하드웨어 타이머 객체 생성
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 3. 웜업 (CUDA 컨텍스트 초기화 병목 및 클럭 부스팅 대기)
print('GPU 웜업 중...')
for _ in range(10):
    _ = torch.matmul(a, b)
torch.cuda.synchronize()

# --- 추가된 대기 로직 ---
print('\n[✅ 초기화 완료] 메모리 대역폭 병목 구간을 무사히 통과했습니다.')
print('-' * 70)
input('>>> 터미널을 열어 2g 부하(test.sh)를 실행하세요. 부하가 시작되면 여기로 돌아와 [Enter]를 누르세요! <<<')
print('-' * 70)
# ------------------------

# 4. 순수 하드웨어 레이턴시 측정 (100회 반복)
print('\n하드웨어 레이턴시 측정 진행 중...')
start_event.record()

for _ in range(100):
    c = torch.matmul(a, b)

end_event.record()
torch.cuda.synchronize() # 하드웨어 타이머 기록이 끝날 때까지 CPU 대기

# 5. 결과 출력
elapsed_time_ms = start_event.elapsed_time(end_event)
print('-' * 50)
print(f'100회 연산 순수 GPU 소요 시간: {elapsed_time_ms:.3f} ms')
print(f'1회 연산 평균 레이턴시: {elapsed_time_ms / 100:.3f} ms')
print('-' * 50)
"
