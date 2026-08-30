#!/usr/bin/env bash
# Experiment 1-3: intentional recoverable CUDA OOM in 1g while 2g is measured.
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiment.sh" fault "${1:-30}"
