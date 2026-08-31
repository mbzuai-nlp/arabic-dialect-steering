#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$NEURON_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$NEURON_DIR/measure_lape_vector_coverage.py" \
  --neurons_dir "$NEURON_DIR/extraction_output/lape_allam_lape1p0_act95p0_no-special" \
  --vectors_dir "$REPO_DIR/arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview" \
  --out_dir "$NEURON_DIR/coverage_output/allam_cairo" \
  --dialects CAI \
  --dialect_map CAI=Cairo \
  --no-strict
