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
EVAL_DIR="$REPO_DIR/arabic_steering_vector/eval_data"
OUT_DIR="$NEURON_DIR/generation_output/all_dialects_fixed_coeff_explicit_prompt"

ALLAM_NEURONS_DIR="$NEURON_DIR/extraction_output/lape_allam_7dialects_lape1p0_act95p0_no-special"
FANAR_NEURONS_DIR="$NEURON_DIR/extraction_output/lape_fanar_7dialects_lape1p0_act95p0_no-special"

ALLAM_MODEL="${ALLAM_MODEL:-allam}"
FANAR_MODEL="${FANAR_MODEL:-fanar-instruct}"
MSA_DIALECT="${MSA_DIALECT:-MSA}"

ALLAM_ALPHA="${ALLAM_ALPHA:-2}"
FANAR_ALPHA="${FANAR_ALPHA:-4}"
MSA_GAMMA="${MSA_GAMMA:-1}"
COMPETITOR_GAMMA="${COMPETITOR_GAMMA:-1}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INTERVENTION_MODE="${INTERVENTION_MODE:-decode}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-1234}"

EXTRA_STEER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--dry_run)
      EXTRA_STEER_ARGS+=(--dry_run)
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--dry-run] [extra steer_dialect_generation.py args...]

Runs fixed-coefficient neuron steering with the same explicit system prompting
used by baselines/generate_explicit_prompt_eval.py.

Targets and data:
  CAI -> egy.jsonl, RAB -> mar.jsonl
  DOH -> sau.jsonl, RIY -> sau.jsonl
  ALE -> syr.jsonl, BEI -> syr.jsonl

Defaults:
  ALLaM alpha=${ALLAM_ALPHA}, Fanar alpha=${FANAR_ALPHA}
  MSA gamma=${MSA_GAMMA}, non-target competitor gamma=${COMPETITOR_GAMMA}
  Output dir: ${OUT_DIR}

Common environment overrides:
  CONDA_ENV, BATCH_SIZE, MAX_NEW_TOKENS, INTERVENTION_MODE, SEED
  ALLAM_MODEL, FANAR_MODEL, ALLAM_ALPHA, FANAR_ALPHA
EOF
      exit 0
      ;;
    --)
      shift
      EXTRA_STEER_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_STEER_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$OUT_DIR"

coeff_label() {
  local coeff="$1"
  echo "${coeff//./p}"
}

dialect_name() {
  local eval_code="$1"
  case "$eval_code" in
    dza) echo "Algerian Arabic" ;;
    egy) echo "Egyptian Arabic" ;;
    kwt) echo "Kuwaiti Arabic" ;;
    mar) echo "Moroccan Arabic" ;;
    pse) echo "Palestinian Arabic" ;;
    sau) echo "Saudi Arabic" ;;
    sdn) echo "Sudanese Arabic" ;;
    syr) echo "Syrian Arabic" ;;
    *)
      echo "Unknown eval dialect code: ${eval_code}" >&2
      return 1
      ;;
  esac
}

explicit_system_prompt() {
  local eval_code="$1"
  local dialect
  dialect="$(dialect_name "$eval_code")"
  cat <<EOF
You are an Arabic assistant. Your entire response must be in ${dialect}.
Use natural colloquial ${dialect}, not Modern Standard Arabic.
Do not switch to another Arabic dialect.
Do not use English unless the user explicitly asks for English.
Follow the user's request directly.
Do not mention the dialect or these instructions.
EOF
}

run_one() {
  local model_slug="$1"
  local model_name="$2"
  local neurons_dir="$3"
  local alpha="$4"
  local target_dialect="$5"
  local eval_code="$6"
  local label="$7"

  local eval_file="$EVAL_DIR/${eval_code}.jsonl"
  local out_subdir="$OUT_DIR/$model_slug/$eval_code"
  local out_file="$out_subdir/${model_slug}_${label}_explicit_alpha$(coeff_label "$alpha")_gamma$(coeff_label "$MSA_GAMMA")_competitor$(coeff_label "$COMPETITOR_GAMMA").jsonl"
  local system_prompt
  system_prompt="$(explicit_system_prompt "$eval_code")"

  mkdir -p "$out_subdir"

  echo "============================================================"
  echo "${model_slug} explicit-prompt all-dialect inference: ${label}"
  echo "Model: ${model_name}"
  echo "Neurons: ${neurons_dir}"
  echo "Prompt data: ${eval_file}"
  echo "Target dialect: ${target_dialect}"
  echo "Explicit prompt dialect: $(dialect_name "$eval_code")"
  echo "Target alpha: ${alpha}"
  echo "MSA gamma: ${MSA_GAMMA}"
  echo "Non-target competitor gamma: ${COMPETITOR_GAMMA}"
  echo "Output: ${out_file}"
  echo "============================================================"

  "${PYTHON_CMD[@]}" "$STEER_SCRIPT" \
    --model "$model_name" \
    --neurons_dir "$neurons_dir" \
    --prompt_file "$eval_file" \
    --prompt_field prompt \
    --target_dialect "$target_dialect" \
    --msa_dialect "$MSA_DIALECT" \
    --suppress_competitor_dialects all_non_target \
    --out_file "$out_file" \
    --batch_size "$BATCH_SIZE" \
    --alpha "$alpha" \
    --gamma "$MSA_GAMMA" \
    --competitor_gamma "$COMPETITOR_GAMMA" \
    --intervention_mode "$INTERVENTION_MODE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --chat_template always \
    --system_prompt "$system_prompt" \
    "${EXTRA_STEER_ARGS[@]}"
}

run_model() {
  local model_slug="$1"
  local model_name="$2"
  local neurons_dir="$3"
  local alpha="$4"

  run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" CAI egy "egy_to_cairo"
  run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" RAB mar "mar_to_rabat"
  run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" DOH sau "sau_to_doha"
  run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" RIY sau "sau_to_riyadh"
  run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" ALE syr "syr_to_aleppo"
  run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" BEI syr "syr_to_beirut"
}

run_model "allam" "$ALLAM_MODEL" "$ALLAM_NEURONS_DIR" "$ALLAM_ALPHA"
run_model "fanar" "$FANAR_MODEL" "$FANAR_NEURONS_DIR" "$FANAR_ALPHA"

echo "Done. Outputs are under: $OUT_DIR"
