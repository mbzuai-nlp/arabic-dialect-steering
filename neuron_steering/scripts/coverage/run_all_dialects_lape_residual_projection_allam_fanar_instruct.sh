#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$NEURON_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<<"$PYTHON_BIN"
OUT_ROOT="${OUT_ROOT:-$NEURON_DIR/coverage_output/all_dialects_lape_residual_projection}"

# Map LAPE/MADAR dialect codes to dialect-vector filenames.
DIALECT_MAP="${DIALECT_MAP:-CAI=Cairo,RAB=Rabat,BEI=Beirut,DOH=Doha,ALE=Aleppo,DAM=Damascus,JED=Jeddah,RIY=Riyadh,TUN=Tunis,KHA=Khartoum}"

ALLAM_MODEL="${ALLAM_MODEL:-allam}"
ALLAM_NEURONS_DIR="${ALLAM_NEURONS_DIR:-$NEURON_DIR/extraction_output/lape_allam_7dialects_lape1p0_act95p0_no-special}"
ALLAM_VECTORS_DIR="${ALLAM_VECTORS_DIR:-$REPO_DIR/arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview}"

# The available Fanar LAPE extraction directory is named "fanar"; the projection
# model and vectors used here are Fanar Instruct. Override FANAR_NEURONS_DIR if
# you have a Fanar Instruct-specific LAPE extraction.
FANAR_MODEL="${FANAR_MODEL:-fanar-instruct}"
FANAR_NEURONS_DIR="${FANAR_NEURONS_DIR:-$NEURON_DIR/extraction_output/lape_fanar_7dialects_lape1p0_act95p0_no-special}"
FANAR_VECTORS_DIR="${FANAR_VECTORS_DIR:-$REPO_DIR/arabic_steering_vector/dialect_vectors/Fanar-1-9B-Instruct}"

RUN_ALLAM="${RUN_ALLAM:-1}"
RUN_FANAR="${RUN_FANAR:-1}"

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

discover_dialects() {
  local neurons_dir="$1"
  local vectors_dir="$2"

  "${PYTHON_CMD[@]}" - "$neurons_dir" "$vectors_dir" "$DIALECT_MAP" "${DIALECTS:-}" <<'PY'
import json
import sys
from pathlib import Path

neurons_dir = Path(sys.argv[1])
vectors_dir = Path(sys.argv[2])
mapping_text = sys.argv[3]
dialects_filter_text = sys.argv[4].strip()

mapping = {}
for item in mapping_text.split(","):
    if not item:
        continue
    key, value = item.split("=", 1)
    mapping[key] = value

requested = set(dialects_filter_text.split()) if dialects_filter_text else None
summary_path = neurons_dir / "run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
dialects = [str(d) for d in summary.get("dialects", [])]

missing = []
for dialect in dialects:
    if dialect == "MSA":
        continue
    if requested is not None and dialect not in requested:
        continue

    vector_name = mapping.get(dialect, dialect)
    candidates = [
        vectors_dir / f"{vector_name}_response_avg_diff.pt",
        vectors_dir / f"{dialect}_response_avg_diff.pt",
    ]
    vector_path = next((path for path in candidates if path.exists()), None)
    if vector_path is None:
        missing.append(dialect)
        continue

    print(f"{dialect}\t{vector_name}\t{vector_path}")

if missing:
    print(
        "Skipping dialects without vectors in "
        f"{vectors_dir}: {', '.join(missing)}",
        file=sys.stderr,
    )
PY
}

aggregate_summaries() {
  local run_dir="$1"

  "${PYTHON_CMD[@]}" - "$run_dir" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_paths = sorted(run_dir.glob("*/residual_projection_summary.csv"))
if not summary_paths:
    raise SystemExit(0)

rows = []
fieldnames = []
for path in summary_paths:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for field in reader.fieldnames or []:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in reader:
            row.setdefault("output_dir", str(path.parent))
            if "output_dir" not in fieldnames:
                fieldnames.append("output_dir")
            rows.append(row)

out_path = run_dir / "residual_projection_summary_all_dialects.csv"
with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})

print(out_path)
PY
}

run_projection_coverage() {
  local run_name="$1"
  local model="$2"
  local neurons_dir="$3"
  local vectors_dir="$4"
  shift 4

  local run_dir="$OUT_ROOT/$run_name"
  local discovered
  discovered="$(discover_dialects "$neurons_dir" "$vectors_dir")"

  if [[ -z "$discovered" ]]; then
    echo "No shared non-MSA dialect vectors found for $run_name." >&2
    exit 1
  fi

  echo "==> Running $run_name residual-projection coverage"
  echo "    Model: $model"
  echo "    Neurons: $neurons_dir"
  echo "    Vectors: $vectors_dir"
  echo "    Output: $run_dir"

  while IFS=$'\t' read -r dialect vector_name vector_path; do
    [[ -n "$dialect" ]] || continue

    local dialect_out_dir="$run_dir/${dialect,,}_${vector_name,,}"
    echo "    - $dialect -> $vector_name"

    "${PYTHON_CMD[@]}" "$NEURON_DIR/measure_lape_residual_projection.py" \
      --model "$model" \
      --neurons_dir "$neurons_dir" \
      --vector_path "$vector_path" \
      --target_dialect "$dialect" \
      --out_dir "$dialect_out_dir" \
      --dtype auto \
      --device_map auto \
      "$@"
  done <<<"$discovered"

  local aggregate_path
  aggregate_path="$(aggregate_summaries "$run_dir")"
  if [[ -n "$aggregate_path" ]]; then
    echo "    Aggregate summary: $aggregate_path"
  fi
}

ran_any=0
if is_enabled "$RUN_ALLAM"; then
  run_projection_coverage "allam_7dialects" "$ALLAM_MODEL" "$ALLAM_NEURONS_DIR" "$ALLAM_VECTORS_DIR" "$@"
  ran_any=1
fi
if is_enabled "$RUN_FANAR"; then
  run_projection_coverage "fanar_instruct_7dialects" "$FANAR_MODEL" "$FANAR_NEURONS_DIR" "$FANAR_VECTORS_DIR" "$@"
  ran_any=1
fi

if [[ "$ran_any" -eq 0 ]]; then
  echo "Nothing to run: RUN_ALLAM and RUN_FANAR are both disabled." >&2
  exit 1
fi

echo "Done. Residual-projection coverage outputs are under: $OUT_ROOT"
