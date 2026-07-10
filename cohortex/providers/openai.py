"""OpenAI + OpenAI-compatible backends. Grok (xAI) reuses the same client with a base_url."""
from __future__ import annotations

import os

from . import register


class _OpenAICompatible:
    _api_key_env = "OPENAI_API_KEY"
    _base_url: str | None = None
    _default_model = "gpt-4o-mini"

    def __init__(self, model: str | None = None, api_key: str | None = None, **_):
        self.model = model or self._default_model
        self._key = api_key or os.getenv(self._api_key_env, "")

    def chat(self, messages, *, temperature: float = 0.3, max_tokens: int | None = None, **opts) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("OpenAI backend needs: pip install openai") from e
        if not self._key:
            raise RuntimeError(f"{self._api_key_env} is not set")
        client = OpenAI(api_key=self._key, base_url=self._base_url)
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        resp = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, **extra,
        )
        u = resp.usage
        self.last_usage = {
            "prompt_tokens": u.prompt_tokens if u else 0,
            "completion_tokens": u.completion_tokens if u else 0,
            "total_tokens": u.total_tokens if u else 0,
        }
        return (resp.choices[0].message.content or "").strip()

    def chat_stream(self, messages, *, temperature: float = 0.3, max_tokens: int | None = None, **opts):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("OpenAI backend needs: pip install openai") from e
        if not self._key:
            raise RuntimeError(f"{self._api_key_env} is not set")
        client = OpenAI(api_key=self._key, base_url=self._base_url)
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        stream = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            stream=True, stream_options={"include_usage": True}, **extra,
        )
        for chunk in stream:
            if chunk.usage:
                u = chunk.usage
                self.last_usage = {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                }
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


@register("openai")
class OpenAIBackend(_OpenAICompatible):
    _api_key_env = "OPENAI_API_KEY"
    _default_model = "gpt-4o-mini"


@register("grok")
class GrokBackend(_OpenAICompatible):
    _api_key_env = "XAI_API_KEY"
    _base_url = "https://api.x.ai/v1"
    _default_model = "grok-2-latest"
