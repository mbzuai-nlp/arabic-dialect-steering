#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$NEURON_DIR/.." && pwd)"

# Set CONDA_ENV="" to use the currently active Python.
CONDA_ENV="${CONDA_ENV-steering}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if command -v conda >/dev/null 2>&1 && [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" "$PYTHON_BIN")
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

STEER_SCRIPT="$NEURON_DIR/steer_dialect_generation.py"
INPUT_JSON="${INPUT_JSON:-$REPO_DIR/arabic_steering_vector/eval_data/msa_samples_300.json}"
WORK_DIR="${WORK_DIR:-$NEURON_DIR/generation_output/msa_samples_300_all_dialects/_prepared}"
PROMPT_JSONL="${PROMPT_JSONL:-$WORK_DIR/msa_samples_300.prompts.jsonl}"
OUT_ROOT="${OUT_ROOT:-$NEURON_DIR/generation_output/msa_samples_300_all_dialects}"

ALLAM_MODEL="${ALLAM_MODEL:-allam}"
ALLAM_NEURONS_DIR="${ALLAM_NEURONS_DIR:-$NEURON_DIR/extraction_output/lape_allam_7dialects_lape1p0_act95p0_no-special}"
ALLAM_ALPHA="${ALLAM_ALPHA:-2}"

FANAR_MODEL="${FANAR_MODEL:-fanar-instruct}"
FANAR_NEURONS_DIR="${FANAR_NEURONS_DIR:-$NEURON_DIR/extraction_output/lape_fanar_7dialects_lape1p0_act95p0_no-special}"
FANAR_ALPHA="${FANAR_ALPHA:-4}"

JAIS2_MODEL="${JAIS2_MODEL:-jais2}"
JAIS2_NEURONS_DIR="${JAIS2_NEURONS_DIR:-$NEURON_DIR/extraction_output/lape_jais2_7dialects_lape1p0_act95p0_no-special}"
JAIS2_ALPHA="${JAIS2_ALPHA:-2}"

RUN_ALLAM="${RUN_ALLAM:-1}"
RUN_FANAR="${RUN_FANAR:-1}"
RUN_JAIS2="${RUN_JAIS2:-0}"

MSA_DIALECT="${MSA_DIALECT:-MSA}"
MSA_GAMMA="${MSA_GAMMA:-1}"
COMPETITOR_GAMMA="${COMPETITOR_GAMMA:-1}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INTERVENTION_MODE="${INTERVENTION_MODE:-decode}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-1234}"

# Default: pure neuron steering with no dialect instruction in the prompt.
# Set USE_EXPLICIT_PROMPT=1 only if you want an explicit dialect system prompt.
USE_EXPLICIT_PROMPT="${USE_EXPLICIT_PROMPT:-0}"

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

Runs neuron-steered inference on:
  ${INPUT_JSON}

By default it runs ALLaM and Fanar Instruct over every non-MSA dialect found in
the corresponding 7-dialect LAPE extraction directories:
  CAI RAB DOH RIY ALE BEI

No explicit dialect prompt is used by default; generation is steered only by the
neuron intervention unless USE_EXPLICIT_PROMPT=1 is set.

Useful environment overrides:
  INPUT_JSON, OUT_ROOT, CONDA_ENV, PYTHON_BIN
  RUN_ALLAM, RUN_FANAR, RUN_JAIS2
  ALLAM_MODEL, FANAR_MODEL, JAIS2_MODEL
  ALLAM_NEURONS_DIR, FANAR_NEURONS_DIR, JAIS2_NEURONS_DIR
  ALLAM_ALPHA, FANAR_ALPHA, JAIS2_ALPHA
  MSA_GAMMA, COMPETITOR_GAMMA, BATCH_SIZE, MAX_NEW_TOKENS
  INTERVENTION_MODE, SEED, USE_EXPLICIT_PROMPT
  DIALECTS="CAI RAB"   # optional subset; default discovers all non-MSA dialects

Extra arguments are forwarded to steer_dialect_generation.py.
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

is_enabled() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    0|false|no|n|off) return 1 ;;
    *)
      echo "Boolean env flags must be one of 1/0, true/false, yes/no, on/off; got: $1" >&2
      exit 1
      ;;
  esac
}

coeff_label() {
  local coeff="$1"
  echo "${coeff//./p}"
}

dialect_slug() {
  local dialect="$1"
  echo "${dialect,,}"
}

dialect_display_name() {
  local dialect="$1"
  case "$dialect" in
    CAI) echo "Egyptian Arabic, especially Cairo Arabic" ;;
    RAB) echo "Moroccan Arabic (Darija), especially Rabat Arabic" ;;
    DOH) echo "Qatari Gulf Arabic, especially Doha Arabic" ;;
    RIY) echo "Saudi Arabic, especially Riyadh/Najdi Arabic" ;;
    ALE) echo "Syrian Arabic, especially Aleppo Arabic" ;;
    BEI) echo "Lebanese Arabic, especially Beirut Arabic" ;;
    *) echo "$dialect Arabic dialect" ;;
  esac
}

system_prompt_for_dialect() {
  local dialect="$1"
  local dialect_name
  dialect_name="$(dialect_display_name "$dialect")"
  cat <<EOF
You are an Arabic assistant. Your entire response must be in ${dialect_name}.
Use natural colloquial ${dialect_name}, not Modern Standard Arabic.
Do not switch to another Arabic dialect.
Do not use English unless the user explicitly asks for English.
Follow the user's request directly.
Do not mention the dialect or these instructions.
EOF
}

prepare_prompts() {
  mkdir -p "$WORK_DIR"
  "${PYTHON_CMD[@]}" - "$INPUT_JSON" "$PROMPT_JSONL" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and isinstance(data.get("samples"), list):
    samples = data["samples"]
elif isinstance(data, list):
    samples = data
else:
    raise SystemExit(f"Expected a list or an object with a 'samples' list in {src}")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for idx, item in enumerate(samples, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"Sample {idx} is not an object")
        prompt = item.get("prompt")
        if prompt is None:
            raise SystemExit(f"Sample {idx} is missing the 'prompt' field")
        record = {"id": str(item.get("id", idx)), "prompt": str(prompt)}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} prompts to {dst}")
PY
}

discover_dialects() {
  local neurons_dir="$1"
  "${PYTHON_CMD[@]}" - "$neurons_dir" "${DIALECTS:-}" "$MSA_DIALECT" <<'PY'
import json
import sys
from pathlib import Path

neurons_dir = Path(sys.argv[1])
requested_text = sys.argv[2].strip()
msa = sys.argv[3].strip().lower()

if requested_text:
    dialects = requested_text.split()
else:
    summary_path = neurons_dir / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing run_summary.json in {neurons_dir}; set DIALECTS explicitly.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dialects = [str(d) for d in summary.get("dialects", [])]

for dialect in dialects:
    if dialect.strip().lower() != msa:
        print(dialect)
PY
}

run_one() {
  local model_slug="$1"
  local model_name="$2"
  local neurons_dir="$3"
  local alpha="$4"
  local target_dialect="$5"

  local dialect_slug_value
  dialect_slug_value="$(dialect_slug "$target_dialect")"
  local out_dir="$OUT_ROOT/$model_slug/$dialect_slug_value"
  local out_file="$out_dir/${model_slug}_${dialect_slug_value}_msa_samples_alpha$(coeff_label "$alpha")_gamma$(coeff_label "$MSA_GAMMA")_competitor$(coeff_label "$COMPETITOR_GAMMA").jsonl"

  local prompt_args=()
  if is_enabled "$USE_EXPLICIT_PROMPT"; then
    prompt_args=(
      --chat_template always
      --system_prompt "$(system_prompt_for_dialect "$target_dialect")"
    )
  fi

  mkdir -p "$out_dir"

  echo "============================================================"
  echo "${model_slug} MSA-sample neuron steering: ${target_dialect}"
  echo "Model: ${model_name}"
  echo "Neurons: ${neurons_dir}"
  echo "Prompt data: ${PROMPT_JSONL}"
  echo "Output: ${out_file}"
  echo "Target alpha: ${alpha}"
  echo "MSA gamma: ${MSA_GAMMA}"
  echo "Non-target competitor gamma: ${COMPETITOR_GAMMA}"
  echo "Explicit prompt: ${USE_EXPLICIT_PROMPT}"
  echo "============================================================"

  "${PYTHON_CMD[@]}" "$STEER_SCRIPT" \
    --model "$model_name" \
    --neurons_dir "$neurons_dir" \
    --prompt_file "$PROMPT_JSONL" \
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
    "${prompt_args[@]}" \
    "${EXTRA_STEER_ARGS[@]}"
}

run_model() {
  local model_slug="$1"
  local model_name="$2"
  local neurons_dir="$3"
  local alpha="$4"

  if [[ ! -d "$neurons_dir" ]]; then
    echo "Skipping $model_slug; neurons directory does not exist: $neurons_dir" >&2
    return
  fi

  local discovered
  discovered="$(discover_dialects "$neurons_dir")"
  if [[ -z "$discovered" ]]; then
    echo "No non-MSA dialects found for $model_slug in $neurons_dir" >&2
    return
  fi

  while IFS= read -r dialect; do
    [[ -n "$dialect" ]] || continue
    run_one "$model_slug" "$model_name" "$neurons_dir" "$alpha" "$dialect"
  done <<<"$discovered"
}

prepare_prompts

ran_any=0
if is_enabled "$RUN_ALLAM"; then
  run_model "allam" "$ALLAM_MODEL" "$ALLAM_NEURONS_DIR" "$ALLAM_ALPHA"
  ran_any=1
fi
if is_enabled "$RUN_FANAR"; then
  run_model "fanar" "$FANAR_MODEL" "$FANAR_NEURONS_DIR" "$FANAR_ALPHA"
  ran_any=1
fi
if is_enabled "$RUN_JAIS2"; then
  run_model "jais2" "$JAIS2_MODEL" "$JAIS2_NEURONS_DIR" "$JAIS2_ALPHA"
  ran_any=1
fi

if [[ "$ran_any" -eq 0 ]]; then
  echo "Nothing to run: RUN_ALLAM, RUN_FANAR, and RUN_JAIS2 are disabled." >&2
  exit 1
fi

echo "Done. Outputs are under: $OUT_ROOT"
