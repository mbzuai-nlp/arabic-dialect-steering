#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NON_TARGET_GAMMAS="0.5"
exec "$SCRIPT_DIR/run_non_target_suppression_abl_allam.sh" "$@"
