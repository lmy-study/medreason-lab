#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <checkpoint/global_step_N/actor>" >&2
  exit 2
fi
python "$PROJECT_DIR/scripts/model_merger.py" --local_dir "$1"
