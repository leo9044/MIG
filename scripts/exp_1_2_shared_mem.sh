#!/usr/bin/env bash
# Experiment 1-2: 2g-alone baseline followed by concurrent 1g/2g contention.
set -Eeuo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SECONDS="${1:-30}"
"$DIR/run_experiment.sh" uma-solo "$RUN_SECONDS"
"$DIR/run_experiment.sh" uma "$RUN_SECONDS"
