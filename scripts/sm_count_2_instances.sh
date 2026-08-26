MIG_UUID_1G=$(nvidia-smi -L | grep "MIG 1g.0gb" | awk -F'UUID: ' '{print $2}' | tr -d ')')

# 존재하는 모든 cap 디바이스를 동적으로 추출하여 --device 옵션으로 변환
CAP_DEVICES=""
for cap in /dev/nvidia-caps/nvidia-cap*; do
    CAP_DEVICES="$CAP_DEVICES --device $cap"
done

sudo docker run --rm -it --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES="all" \
    -e CUDA_VISIBLE_DEVICES="$MIG_UUID_1G" \
    $CAP_DEVICES \
    --device /dev/nvidia0 \
    --device /dev/nvidiactl \
    --device /dev/nvidia-uvm \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    python3 -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'\n▶ [동시 분할 상태] 디바이스 명: {props.name}')
print(f'▶ [동시 분할 상태] 실제 SM 개수: {props.multi_processor_count}개\n')
"
