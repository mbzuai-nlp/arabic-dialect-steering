#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in "$JOB_DIR"/allam_*.sh "$JOB_DIR"/fanar_*.sh; do
  echo "==> $(basename "$script")"
  bash "$script" "$@"
done
