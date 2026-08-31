#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RANDOM_BASELINE="${RANDOM_BASELINE:-1000}"
SEED="${SEED:-13}"
OUT_ROOT="${OUT_ROOT:-$NEURON_DIR/coverage_output/all_dialects_lape_residual_projection_random_exclude_selected}"
export OUT_ROOT

if ! [[ "$RANDOM_BASELINE" =~ ^[0-9]+$ ]] || [[ "$RANDOM_BASELINE" -le 0 ]]; then
  echo "RANDOM_BASELINE must be a positive integer; got: $RANDOM_BASELINE" >&2
  exit 1
fi

echo "Running residual projection for ALLaM and Fanar with random baselines."
echo "Random baseline samples: $RANDOM_BASELINE"
echo "Random baseline excludes LAPE-selected neurons: true"
echo "Seed: $SEED"
echo "Output root: $OUT_ROOT"

exec "$SCRIPT_DIR/run_all_dialects_lape_residual_projection_allam_fanar_instruct.sh" \
  --random_baseline "$RANDOM_BASELINE" \
  --random_exclude_selected \
  --seed "$SEED" \
  "$@"
