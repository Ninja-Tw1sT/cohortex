"""
LLM backend abstraction.

Every backend implements `chat(messages) -> str`. Backends register themselves
by name; `get_backend()` resolves the global default (from config) or a named
override, and reads that provider's API key from the environment.

Adding a provider = one file that defines a class and `@register("name")`s it.
Cloud SDKs are imported lazily inside `chat()`, so a local-only install needs
nothing beyond httpx.

Streaming is optional: a backend may also implement `chat_stream(messages, ...)
-> Iterator[str]`, yielding text chunks and setting `self.last_usage` once the
stream ends (same contract as `chat()`). Not part of the `LLMBackend` Protocol
below (so fake/test backends never need it) — callers check for it with
`getattr(backend, "chat_stream", None)`, same duck-typing already used for
`last_usage`. All four bundled providers implement it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

__all__ = ["LLMBackend", "Message", "register", "get_backend",
           "available_backends", "FallbackBackend"]


@runtime_checkable
class LLMBackend(Protocol):
    name: str
    model: str

    def chat(self, messages: list[Message], *, temperature: float = 0.3, **opts) -> str:
        ...


_BUILDERS: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _BUILDERS[name] = cls
        return cls
    return deco


def _ensure_loaded() -> None:
    # Import the adapter modules so their @register side effects run.
    from . import ollama, openai, anthropic, gemini  # noqa: F401


class FallbackBackend:
    """Try each backend in order; return the first that succeeds."""

    def __init__(self, backends: list[LLMBackend]):
        self._backends = backends
        self.name = "+".join(b.name for b in backends)
        self.model = backends[0].model

    def chat(self, messages, *, temperature=0.3, **opts) -> str:
        errors = []
        for b in self._backends:
            try:
                result = b.chat(messages, temperature=temperature, **opts)
                self.last_usage = getattr(b, "last_usage", None)
                return result
            except Exception as e:  # noqa: BLE001
                errors.append(f"{b.name}: {e}")
        raise RuntimeError("All backends failed -> " + " | ".join(errors))

    def chat_stream(self, messages, *, temperature=0.3, **opts):
        # Falling back to a *different* backend mid-stream would replay output
        # the caller already saw — only fall back before the first chunk of a
        # given backend's stream; once any chunk is yielded, propagate errors.
        errors = []
        for b in self._backends:
            stream_fn = getattr(b, "chat_stream", None)
            if not stream_fn:
                errors.append(f"{b.name}: no chat_stream support")
                continue
            started = False
            try:
                for chunk in stream_fn(messages, temperature=temperature, **opts):
                    started = True
                    yield chunk
                self.last_usage = getattr(b, "last_usage", None)
                return
            except Exception as e:  # noqa: BLE001
                errors.append(f"{b.name}: {e}")
                if started:
                    raise RuntimeError(f"backend {b.name} failed mid-stream: {e}") from e
        raise RuntimeError("All backends failed (streaming) -> " + " | ".join(errors))


def get_backend(name: str | None = None, model: str | None = None,
                fallback: list[str] | None = None, **cfg) -> LLMBackend:
    """Build a backend by name (or the global default) with an optional fallback chain."""
    from cohortex import config
    _ensure_loaded()

    def build(n: str) -> LLMBackend:
        if n not in _BUILDERS:
            raise ValueError(f"Unknown backend {n!r}. Available: {sorted(_BUILDERS)}")
        return _BUILDERS[n](model=model, **cfg)

    primary = build(name or config.DEFAULT_BACKEND)
    if fallback:
        return FallbackBackend([primary, *(build(n) for n in fallback)])
    return primary


def available_backends() -> list[str]:
    _ensure_loaded()
    return sorted(_BUILDERS)
