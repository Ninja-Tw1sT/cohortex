"""Google Gemini backend."""
from __future__ import annotations

import os

from . import register


@register("gemini")
class GeminiBackend:
    def __init__(self, model: str | None = None, **_):
        self.model = model or "gemini-2.5-flash"
        self._key = os.getenv("GEMINI_API_KEY", "")

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
        convo = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
        )
        resp = client.models.generate_content(
            model=self.model,
            contents=convo,
            config=types.GenerateContentConfig(
                system_instruction=system or None, temperature=temperature,
            ),
        )
        return (resp.text or "").strip()
