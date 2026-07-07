"""
LLM backend abstraction.

Every backend implements `chat(messages) -> str`. Backends register themselves
by name; `get_backend()` resolves the global default (from config) or a named
override, and reads that provider's API key from the environment.

Adding a provider = one file that defines a class and `@register("name")`s it.
Cloud SDKs are imported lazily inside `chat()`, so a local-only install needs
nothing beyond httpx.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


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
                return b.chat(messages, temperature=temperature, **opts)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{b.name}: {e}")
        raise RuntimeError("All backends failed -> " + " | ".join(errors))


def get_backend(name: str | None = None, model: str | None = None,
                fallback: list[str] | None = None, **cfg) -> LLMBackend:
    """Build a backend by name (or the global default) with an optional fallback chain."""
    from vaultmind import config
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
