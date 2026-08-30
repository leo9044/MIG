#!/usr/bin/env bash
# Experiment 1-1 (MIG): concurrent workloads on 1g and 2g slices.
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/run_experiment.sh" mig "${1:-30}"
