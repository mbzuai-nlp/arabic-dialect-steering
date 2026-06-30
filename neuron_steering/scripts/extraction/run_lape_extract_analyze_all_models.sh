#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$NEURON_DIR/lape_extract.py" --model allam --data_manifest "$NEURON_DIR/experiment.jsonl" --out_dir "$NEURON_DIR/extraction_output/lape_allam_lape1p0_act95p0_no-special" --seq_len 512 --batch_size 32 --dtype auto --device_map auto --lape_percentile 1.0 --activation_percentile 95.0
"$PYTHON_BIN" "$NEURON_DIR/lape_analyze.py" --input_dir "$NEURON_DIR/extraction_output/lape_allam_lape1p0_act95p0_no-special" --out_dir "$NEURON_DIR/extraction_output/lape_allam_lape1p0_act95p0_no-special/analysis"

"$PYTHON_BIN" "$NEURON_DIR/lape_extract.py" --model fanar --data_manifest "$NEURON_DIR/experiment.jsonl" --out_dir "$NEURON_DIR/extraction_output/lape_fanar_lape1p0_act95p0_no-special" --seq_len 512 --batch_size 64 --dtype auto --device_map auto --lape_percentile 1.0 --activation_percentile 95.0
"$PYTHON_BIN" "$NEURON_DIR/lape_analyze.py" --input_dir "$NEURON_DIR/extraction_output/lape_fanar_lape1p0_act95p0_no-special" --out_dir "$NEURON_DIR/extraction_output/lape_fanar_lape1p0_act95p0_no-special/analysis"

"$PYTHON_BIN" "$NEURON_DIR/lape_extract.py" --model jais2 --data_manifest "$NEURON_DIR/experiment.jsonl" --out_dir "$NEURON_DIR/extraction_output/lape_jais2_lape1p0_act95p0_no-special" --seq_len 512 --batch_size 64 --dtype auto --device_map auto --lape_percentile 1.0 --activation_percentile 95.0
"$PYTHON_BIN" "$NEURON_DIR/lape_analyze.py" --input_dir "$NEURON_DIR/extraction_output/lape_jais2_lape1p0_act95p0_no-special" --out_dir "$NEURON_DIR/extraction_output/lape_jais2_lape1p0_act95p0_no-special/analysis"
