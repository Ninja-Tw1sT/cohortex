"""Anthropic (Claude) backend."""
from __future__ import annotations

import os

from . import register


# Configurable — override per-agent via the profile `model:` field, or globally
# in configs/backends.yaml. Public model IDs rotate; set this to whatever your
# account has access to (e.g. claude-sonnet-4-5, claude-opus-4-5).
DEFAULT_MODEL = "claude-sonnet-4-5"


@register("anthropic")
class AnthropicBackend:
    def __init__(self, model: str | None = None, api_key: str | None = None, **_):
        self.model = model or DEFAULT_MODEL
        self._key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def chat(self, messages, *, temperature: float = 0.3, max_tokens: int | None = None, **opts) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Anthropic backend needs: pip install anthropic") from e
        if not self._key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=self._key)
        # Anthropic takes the system prompt as a top-level arg, not a message.
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        if not convo:  # Anthropic 400s on an empty messages list
            raise RuntimeError("Anthropic requires at least one user/assistant message")
        resp = client.messages.create(
            model=self.model,
            system=system or None,
            messages=convo,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()
