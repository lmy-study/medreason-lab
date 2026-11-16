"""Agent runtime configuration; loaded from environment."""
import os

LLM_CONFIG = {
    "api_key": os.getenv("MEDREASON_API_KEY", "EMPTY"),
    "model_name": os.getenv("MEDREASON_MODEL", "medreason-vl-2b-rl"),
    "base_url": os.getenv("MEDREASON_BASE_URL", "http://127.0.0.1:8000/v1"),
    "temperature": float(os.getenv("MEDREASON_TEMPERATURE", "0.2")),
    "max_tokens": int(os.getenv("MEDREASON_MAX_TOKENS", "2048")),
}
MEM0_CONFIG = {"api_key": os.getenv("MEM0_API_KEY", "")}
