#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
# EasyR1/v0.3.2 must be installed separately. Check environment before GPU training.
python -m verl.trainer.main config="${PROJECT_DIR}/configs/medvlm_2b_dapo.yaml" "$@"
