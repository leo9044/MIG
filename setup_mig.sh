#!/bin/bash

# 비밀번호 변수 설정
PASSWORD="nvidia"

echo "=== MIG 설정 스크립트를 시작합니다 ==="

# 1. 그래픽 인터페이스 종료 (컴퓨트 모드 전환)
echo "1. multi-user.target 전환 중..."
echo "$PASSWORD" | sudo -S systemctl isolate multi-user.target

# 2. persistence 모드 활성화
echo "2. NVIDIA Persistence Mode 활성화 중..."
echo "$PASSWORD" | sudo -S nvidia-smi -pm 1

# 3. MIG 모드 활성화
echo "3. MIG 모드 활성화 중..."
echo "$PASSWORD" | sudo -S nvidia-smi -mig 1

echo "=== MIG 설정이 완료되었습니다 ==="

echo "=== 4. MIG 기본 설정 실행 중 ==="
sudo nvidia-smi mig -cgi 78,83 -C
echo "=== 드라이버 안정화 대기 중... ==="
sleep 3
