"""Google Gemini backend."""
from __future__ import annotations

import os

from . import register


@register("gemini")
class GeminiBackend:
    def __init__(self, model: str | None = None, api_key: str | None = None, **_):
        self.model = model or "gemini-2.5-flash"
        self._key = api_key or os.getenv("GEMINI_API_KEY", "")

    def chat(self, messages, *, temperature: float = 0.3, **opts) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError("Gemini backend needs: pip install google-genai") from e
        if not self._key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=self._key)
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        # Preserve turn structure: Gemini uses roles "user" and "model".
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages if m["role"] != "system"
        ]
        resp = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system or None, temperature=temperature,
            ),
        )
        u = getattr(resp, "usage_metadata", None)
        p = getattr(u, "prompt_token_count", 0) or 0 if u else 0
        c = getattr(u, "candidates_token_count", 0) or 0 if u else 0
        self.last_usage = {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
        }
        return (resp.text or "").strip()

    def chat_stream(self, messages, *, temperature: float = 0.3, **opts):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError("Gemini backend needs: pip install google-genai") from e
        if not self._key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = genai.Client(api_key=self._key)
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages if m["role"] != "system"
        ]
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system or None, temperature=temperature,
            ),
        )
        last_chunk = None
        for chunk in stream:
            last_chunk = chunk
            if chunk.text:
                yield chunk.text
        u = getattr(last_chunk, "usage_metadata", None) if last_chunk else None
        p = getattr(u, "prompt_token_count", 0) or 0 if u else 0
        c = getattr(u, "candidates_token_count", 0) or 0 if u else 0
        self.last_usage = {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
        }
