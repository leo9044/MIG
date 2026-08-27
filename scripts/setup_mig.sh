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

echo "=== 드라이버 안정화 대기 중... ==="
sleep 3

#4. 권장 프로파일 동시 생성 (83번=2g.0gb+gfx, 78번=1g.0gb+me)
echo "4. 인스턴스 생성 중...(78번 단독생성시 오류 발생)"
echo "$PASSWORD" | sudo -S nvidia-smi mig -cgi 83,78 -C

echo "=== MIG 파티셔닝 완료 ==="
echo "--------------------------------------------------"
nvidia-smi -L
echo "--------------------------------------------------"
echo "위 목록에서 두 개의 MIG UUID(2g, 1g)가 정상적으로 출력되는지 확인하세요."

