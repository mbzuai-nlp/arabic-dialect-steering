#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 {allam|fanar} DIALECT [extra measure_lape_residual_projection.py args...]" >&2
  exit 2
fi

MODEL_KEY="$1"
DIALECT="$2"
shift 2

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COVERAGE_SCRIPT_DIR="$(cd "$JOB_DIR/.." && pwd)"
NEURON_DIR="$(cd "$COVERAGE_SCRIPT_DIR/../.." && pwd)"

RANDOM_BASELINE="${RANDOM_BASELINE:-1000}"
SEED="${SEED:-13}"
OUT_ROOT="${OUT_ROOT:-$NEURON_DIR/coverage_output/all_dialects_lape_residual_projection_random_exclude_selected}"

if ! [[ "$RANDOM_BASELINE" =~ ^[0-9]+$ ]] || [[ "$RANDOM_BASELINE" -le 0 ]]; then
  echo "RANDOM_BASELINE must be a positive integer; got: $RANDOM_BASELINE" >&2
  exit 1
fi

case "$MODEL_KEY" in
  allam)
    export RUN_ALLAM=1
    export RUN_FANAR=0
    ;;
  fanar)
    export RUN_ALLAM=0
    export RUN_FANAR=1
    ;;
  *)
    echo "Unknown model key: $MODEL_KEY. Expected allam or fanar." >&2
    exit 2
    ;;
esac

export DIALECTS="$DIALECT"
export OUT_ROOT

echo "Running residual projection job: model=$MODEL_KEY dialect=$DIALECT"
echo "Random baseline samples: $RANDOM_BASELINE"
echo "Random baseline excludes LAPE-selected neurons: true"
echo "Seed: $SEED"
echo "Output root: $OUT_ROOT"

exec "$COVERAGE_SCRIPT_DIR/run_all_dialects_lape_residual_projection_allam_fanar_instruct.sh" \
  --random_baseline "$RANDOM_BASELINE" \
  --random_exclude_selected \
  --seed "$SEED" \
  "$@"
