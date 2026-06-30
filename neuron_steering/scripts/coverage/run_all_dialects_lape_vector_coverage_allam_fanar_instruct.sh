#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec "$SCRIPT_DIR/run_all_dialects_lape_residual_projection_allam_fanar_instruct.sh" "$@"
