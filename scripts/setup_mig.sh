#!/usr/bin/env bash
# Run once after reboot, before any MIG experiment.
set -Eeuo pipefail

echo '[setup] Switching to multi-user.target, enabling persistence and MIG mode.'
sudo systemctl isolate multi-user.target
sudo nvidia-smi -pm 1
sudo nvidia-smi -mig 1
sleep 3
echo '[setup] Creating 2g first (profile 83), then 1g (profile 78): required creation order.'
sudo nvidia-smi mig -cgi 83,78 -C
sleep 3
nvidia-smi -L
