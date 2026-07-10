"""Local Ollama backend (no API key). Adapts ai-workflow's rag/generate routing."""
from __future__ import annotations

from . import register


@register("ollama")
class OllamaBackend:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 fallback_urls: list[str] | None = None, **_):
        from cohortex import config
        self.model = model or "phi3:mini"
        self._urls = [base_url or config.OLLAMA_BASE_URL]
        self._urls += fallback_urls if fallback_urls is not None else config.OLLAMA_FALLBACK_URLS

    def chat(self, messages, *, temperature: float = 0.3, num_ctx: int | None = None, **opts) -> str:
        import httpx
        options = {"temperature": temperature}
        if num_ctx:
            options["num_ctx"] = num_ctx
        last = None
        for url in self._urls:
            try:
                r = httpx.post(
                    f"{url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": options,
                    },
                    timeout=180,
                )
                r.raise_for_status()
                data = r.json()
                p = data.get("prompt_eval_count", 0)
                c = data.get("eval_count", 0)
                self.last_usage = {
                    "prompt_tokens": p,
                    "completion_tokens": c,
                    "total_tokens": p + c,
                }
                return data["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"Ollama chat failed on {self._urls}: {last}")
