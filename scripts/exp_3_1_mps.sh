#!/usr/bin/env bash
# Experiment 3-1: two processes in the same 1g MIG slice, MPS off then on.
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SECONDS="${1:-30}"
"$DIR/run_experiment.sh" mps-off "$RUN_SECONDS"
"$DIR/run_experiment.sh" mps-on "$RUN_SECONDS"
