#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <path-to-global_step_N/actor>" >&2
  exit 2
fi
ACTOR="$(cd "$ROOT" && cd "$1" && pwd)"
python "$ROOT/scripts/model_merger.py" --local_dir "$ACTOR"
mkdir -p "$ROOT/models/MedReason-VL-2B-RL"
cp -a "$ACTOR/huggingface/." "$ROOT/models/MedReason-VL-2B-RL/"
# Save the original training base processor/tokenizer if not already included.
MODEL_DEST="$ROOT/models/MedReason-VL-2B-RL" python - <<'PYCODE'
import os
from pathlib import Path
from transformers import AutoProcessor

output = Path(os.environ["MODEL_DEST"])
if not (output / "config.json").exists():
    raise FileNotFoundError(f"Missing model config: {output / 'config.json'}")
if not list(output.glob("*.safetensors")) and not list(output.glob("pytorch_model*.bin")):
    raise FileNotFoundError(f"No exported model weights found in {output}")
if not (output / "processor_config.json").exists() and not (output / "preprocessor_config.json").exists():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    processor.save_pretrained(str(output))
PYCODE
echo "Exported inference model: $ROOT/models/MedReason-VL-2B-RL"
