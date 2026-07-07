"""
Embedding providers. Ollama by default (local, no key), with sentence-transformers
as an offline fallback and OpenAI as an option. Adapts ai-workflow's
rag/ingest._get_embedding_function.
"""
from __future__ import annotations

from cohortex import config


class Embedder:
    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.EMBED_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL

    def embed_batch(self, texts):
        import httpx
        texts = list(texts)
        # Prefer the batch endpoint (/api/embed takes a list); fall back to the
        # older single-prompt /api/embeddings for pre-batch Ollama versions.
        try:
            r = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            r.raise_for_status()
            embs = r.json().get("embeddings")
            if embs and len(embs) == len(texts):
                return embs
        except Exception:  # noqa: BLE001 - fall through to per-item endpoint
            pass
        out = []
        for t in texts:
            r = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=60,
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model)

    def embed_batch(self, texts):
        return [v.tolist() for v in self._m.encode(list(texts))]


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        import os
        self.model = model
        self._key = os.getenv("OPENAI_API_KEY", "")

    def embed_batch(self, texts):
        from openai import OpenAI
        client = OpenAI(api_key=self._key)
        resp = client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]


def get_embedder(provider: str | None = None, model: str | None = None) -> Embedder:
    provider = provider or config.EMBED_PROVIDER
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    if provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small")
    # Default: Ollama. Probe cheaply via /api/tags (no embedding call); fall back
    # to local sentence-transformers if the server is unreachable.
    try:
        import httpx
        httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3).raise_for_status()
        return OllamaEmbedder(model)
    except Exception:
        return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
