#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$JOB_DIR/_run_one_model_dialect.sh" allam ALE "$@"
