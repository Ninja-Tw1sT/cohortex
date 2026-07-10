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
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        if not convo:  # Anthropic 400s on an empty messages list
            raise RuntimeError("Anthropic requires at least one user/assistant message")
        # Structured system with cache_control: in supervisor loops the same system
        # prompt is sent every round — ephemeral caching avoids re-tokenizing it.
        system_param = (
            [{"type": "text", "text": system_text,
              "cache_control": {"type": "ephemeral"}}]
            if system_text else None
        )
        resp = client.messages.create(
            model=self.model,
            system=system_param,
            messages=convo,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        )
        inp = getattr(resp.usage, "input_tokens", 0)
        out = getattr(resp.usage, "output_tokens", 0)
        self.last_usage = {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
        }
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip()

    def chat_stream(self, messages, *, temperature: float = 0.3, max_tokens: int | None = None, **opts):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Anthropic backend needs: pip install anthropic") from e
        if not self._key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=self._key)
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        if not convo:
            raise RuntimeError("Anthropic requires at least one user/assistant message")
        system_param = (
            [{"type": "text", "text": system_text,
              "cache_control": {"type": "ephemeral"}}]
            if system_text else None
        )
        with client.messages.stream(
            model=self.model,
            system=system_param,
            messages=convo,
            temperature=temperature,
            max_tokens=max_tokens or 1024,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
            inp = getattr(final.usage, "input_tokens", 0)
            out = getattr(final.usage, "output_tokens", 0)
            self.last_usage = {
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": inp + out,
                "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0),
            }
