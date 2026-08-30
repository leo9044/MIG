#!/usr/bin/env bash
# Experiment 1-1 (Non-MIG): two independent workloads sharing the whole GPU.
# Run this after the MIG cases: it disables MIG mode.
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiment.sh" nomig "${1:-30}"
