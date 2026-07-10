"""
Configuration and env resolution for Cohortex.

Loads `.env` from the project root and exposes the config directory plus a few
global defaults. Everything else lives in YAML under `configs/` and is loaded on
demand by the runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Where backends.yaml, agents/*.yaml, crews/*.yaml, vaults.yaml live.
CONFIG_DIR = Path(os.getenv("COHORTEX_CONFIG_DIR", str(PROJECT_ROOT / "configs")))


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _backends_yaml() -> dict:
    p = CONFIG_DIR / "backends.yaml"
    try:
        return load_yaml(p) if p.exists() else {}
    except Exception:
        return {}


_BY = _backends_yaml()

# Global default backend + model. Precedence: env var > backends.yaml > built-in.
# Overridable per-agent in a profile.
DEFAULT_BACKEND = os.getenv("COHORTEX_BACKEND") or _BY.get("default_backend") or "ollama"
DEFAULT_MODEL = os.getenv("COHORTEX_MODEL") or _BY.get("default_model") or ""  # "" → backend default

# Local Ollama endpoint(s). Comma-separated fallbacks are allowed.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_FALLBACK_URLS = [
    u.strip() for u in os.getenv("OLLAMA_FALLBACK_URLS", "").split(",") if u.strip()
]
# CPU inference on large context windows (long-context mode) routinely exceeds
# a short timeout — 300s default, override per environment or per-backend.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))

# Default vault store location (ChromaDB) + embedding model.
VAULT_DB_PATH = Path(os.getenv("COHORTEX_DB_PATH", str(PROJECT_ROOT / ".vaults")))
EMBED_MODEL = os.getenv("COHORTEX_EMBED_MODEL", "nomic-embed-text")
EMBED_PROVIDER = os.getenv("COHORTEX_EMBED_PROVIDER", "ollama")  # ollama|sentence_transformers|openai

TOP_K_DEFAULT = int(os.getenv("COHORTEX_TOP_K", "4"))
