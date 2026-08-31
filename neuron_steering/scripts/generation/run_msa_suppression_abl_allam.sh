#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$NEURON_DIR/.." && pwd)"

# Set CONDA_ENV="" to use the currently active Python instead.
CONDA_ENV="${CONDA_ENV-steering}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if command -v conda >/dev/null 2>&1 && [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" "$PYTHON_BIN")
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

STEER_SCRIPT="$NEURON_DIR/steer_dialect_generation.py"
NEURONS_DIR="$NEURON_DIR/extraction_output/lape_allam_lape1p0_act95p0_no-special"
EVAL_DIR="$REPO_DIR/arabic_steering_vector/eval_data"
OUT_DIR="$NEURON_DIR/generation_output/MSA_coeff_abl_allam"

MODEL="${MODEL:-allam}"
MSA_DIALECT="${MSA_DIALECT:-MSA}"
ALPHA="${ALPHA:-2}"
MSA_GAMMAS="${MSA_GAMMAS:-0 0.25 0.5 0.75 1}"
COMPETITOR_GAMMA="${COMPETITOR_GAMMA:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INTERVENTION_MODE="${INTERVENTION_MODE:-decode}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-1234}"

mkdir -p "$OUT_DIR/egy" "$OUT_DIR/mar"

gamma_label() {
  local gamma="$1"
  echo "${gamma//./p}"
}

run_one() {
  local gamma="$1"
  local eval_file="$2"
  local target_dialect="$3"
  local competitor_dialect="$4"
  local split="$5"
  local label="$6"
  local out_subdir="$OUT_DIR/$split"
  local out_file="$out_subdir/allam_${label}_alpha${ALPHA}_gamma$(gamma_label "$gamma").jsonl"

  echo "============================================================"
  echo "ALLaM MSA suppression ablation: ${label}, alpha=${ALPHA}, gamma=${gamma}"
  echo "Output: ${out_file}"
  echo "============================================================"

  "${PYTHON_CMD[@]}" "$STEER_SCRIPT" \
    --model "$MODEL" \
    --neurons_dir "$NEURONS_DIR" \
    --prompt_file "$eval_file" \
    --prompt_field prompt \
    --target_dialect "$target_dialect" \
    --msa_dialect "$MSA_DIALECT" \
    --suppress_competitor_dialects "$competitor_dialect" \
    --out_file "$out_file" \
    --batch_size "$BATCH_SIZE" \
    --alpha "$ALPHA" \
    --gamma "$gamma" \
    --competitor_gamma "$COMPETITOR_GAMMA" \
    --intervention_mode "$INTERVENTION_MODE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED"
}

for gamma in $MSA_GAMMAS; do
  run_one "$gamma" "$EVAL_DIR/egy.jsonl" CAI RAB "egy" "egy_to_cairo"
  run_one "$gamma" "$EVAL_DIR/mar.jsonl" RAB CAI "mar" "mar_to_rabat"
done

echo "Done. Outputs are under: $OUT_DIR"
