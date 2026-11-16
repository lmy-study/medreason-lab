#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/../../medvlm-rl/models/MedReason-VL-2B-RL" && pwd)"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Export model to $MODEL_DIR before starting vLLM" >&2
  exit 2
fi
exec vllm serve "$MODEL_DIR" \
  --served-model-name medreason-vl-2b-rl \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes "$@"
