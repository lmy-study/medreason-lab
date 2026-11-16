"""Shared configuration for MedReason Lab model serving.

Environment variables override defaults; no secrets are committed.
"""
import os
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
RL_ROOT = LAB_ROOT / "medvlm-rl"
MODEL_DIR = RL_ROOT / "models" / "MedReason-VL-2B-RL"
MODEL_NAME = os.getenv("MEDREASON_MODEL", "medreason-vl-2b-rl")
MODEL_BASE_URL = os.getenv("MEDREASON_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_API_KEY = os.getenv("MEDREASON_API_KEY", "EMPTY")

LLM_CONFIG = {
    "model_name": MODEL_NAME,
    "base_url": MODEL_BASE_URL,
    "api_key": MODEL_API_KEY,
    "temperature": float(os.getenv("MEDREASON_TEMPERATURE", "0.2")),
    "max_tokens": int(os.getenv("MEDREASON_MAX_TOKENS", "2048")),
}
MEM0_CONFIG = {"api_key": os.getenv("MEM0_API_KEY", "")}
