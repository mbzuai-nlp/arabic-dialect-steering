#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change these with environment variables if needed:
#   MODEL="Qwen/Qwen2.5-7B-Instruct" bash run_sampled_cairo_msa_vectors.sh
#   SPLITS_DIR="/path/to/sample_splits" bash run_sampled_cairo_msa_vectors.sh
#   SIZES="1k 2k" bash run_sampled_cairo_msa_vectors.sh
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-QCRI/Fanar-1-9B-Instruct}"
LAYER="${LAYER:-16}"
SPLITS_DIR="${SPLITS_DIR:-$SCRIPT_DIR/data/sample_splits}"
RUNS_DIR="${RUNS_DIR:-$SCRIPT_DIR/sample_split_vector_runs_fanar}"
read -r -a SIZES_ARRAY <<< "${SIZES:-1k 2k 4k 6k 12k}"

EXTRACT_SCRIPT="$SCRIPT_DIR/extract_dialect_vectors_fast.py"

for SIZE in "${SIZES_ARRAY[@]}"; do
  CAIRO_SPLIT="$SPLITS_DIR/MADAR.corpus.Cairo.sample_${SIZE}.tsv"
  MSA_SPLIT="$SPLITS_DIR/MADAR.corpus.MSA.sample_${SIZE}.tsv"
  RUN_DIR="$RUNS_DIR/$SIZE"
  RUN_DATA_DIR="$RUN_DIR/data"

  if [[ ! -f "$CAIRO_SPLIT" ]]; then
    echo "Missing Cairo split: $CAIRO_SPLIT" >&2
    exit 1
  fi
  if [[ ! -f "$MSA_SPLIT" ]]; then
    echo "Missing MSA split: $MSA_SPLIT" >&2
    exit 1
  fi

  mkdir -p "$RUN_DATA_DIR"
  ln -sfn "$CAIRO_SPLIT" "$RUN_DATA_DIR/MADAR.corpus.Cairo.tsv"
  ln -sfn "$MSA_SPLIT" "$RUN_DATA_DIR/MADAR.corpus.MSA.tsv"

  echo "============================================================"
  echo "Extracting Cairo - MSA steering vector for sample size: $SIZE"
  echo "Cairo: $CAIRO_SPLIT"
  echo "MSA:   $MSA_SPLIT"
  echo "Run:   $RUN_DIR"
  echo "============================================================"

  (
    cd "$RUN_DIR"
    "$PYTHON" "$EXTRACT_SCRIPT" \
      --model "$MODEL" \
      --dialects Cairo \
      --data-dir "$RUN_DATA_DIR" \
      --layer "$LAYER" \
      --output-dir "$RUN_DIR/analysis_results_dialect" \
      --force-reextract
  )
done

echo "Done. Vectors are under: $RUNS_DIR/<size>/dialect_vectors/<model_name>/Cairo_response_avg_diff.pt"
