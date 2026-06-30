#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$NEURON_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$NEURON_DIR/measure_lape_residual_projection.py" \
  --model allam \
  --neurons_dir "$NEURON_DIR/extraction_output/lape_allam_lape1p0_act95p0_no-special" \
  --vector_path "$REPO_DIR/arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt" \
  --target_dialect CAI \
  --out_dir "$NEURON_DIR/coverage_output/allam_cairo_residual_projection" \
  --dtype auto \
  --device_map auto
