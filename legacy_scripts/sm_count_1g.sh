echo "=== 1. 기존 MIG 파괴 및 1g(Profile 78) 단독 생성 ==="
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi
sudo nvidia-smi mig -cgi 78 -C
sleep 3

echo "=== 2. UUID 및 캡 디바이스 동적 추출 ==="
MIG_UUID_1G_ALONE=$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')

CAP_DEVICES=""
for cap in /dev/nvidia-caps/nvidia-cap*; do
    CAP_DEVICES="$CAP_DEVICES --device $cap"
done

echo "=== 3. 1g 단독 상태 측정 시작 ==="
sudo docker run --rm -it --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$MIG_UUID_1G_ALONE" \
    $CAP_DEVICES \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'\n▶ [1g 단독 상태] 디바이스 명: {props.name}')
print(f'▶ [1g 단독 상태] 실제 SM 개수: {props.multi_processor_count}개\n')
"
