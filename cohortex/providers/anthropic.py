"""Anthropic (Claude) backend."""
from __future__ import annotations

import os

from . import register


@register("anthropic")
class AnthropicBackend:
    def __init__(self, model: str | None = None, **_):
        self.model = model or "claude-sonnet-5"
        self._key = os.getenv("ANTHROPIC_API_KEY", "")

    def chat(self, messages, *, temperature: float = 0.3, max_tokens: int = 1024, **opts) -> str:
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
        resp = client.messages.create(
            model=self.model,
            system=system or None,
            messages=convo,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()
