"""
A tiny tool registry. Decorate a function with @tool and it becomes available to
any agent whose profile lists it. Agents call tools via a ReAct loop (see agent.py).
"""
from __future__ import annotations

import re
from typing import Callable

_TOOLS: dict[str, dict] = {}


def tool(fn: Callable | None = None, *, name: str | None = None):
    """Register a single-string-argument tool. Its docstring is shown to the agent."""
    def deco(f: Callable):
        _TOOLS[name or f.__name__] = {"fn": f, "doc": (f.__doc__ or "").strip().splitlines()[0]}
        return f
    return deco(fn) if fn else deco


class ToolRegistry:
    """A per-agent view over the global tool set (only the names the profile lists)."""

    def __init__(self, names: list[str] | None = None):
        self._names = list(names) if names else list(_TOOLS)

    def specs(self) -> dict[str, str]:
        return {n: _TOOLS[n]["doc"] for n in self._names if n in _TOOLS}

    def has(self, name: str) -> bool:
        return name in self._names and name in _TOOLS

    def run(self, name: str, arg: str) -> str:
        if name not in _TOOLS:
            return f"error: unknown tool {name!r}"
        try:
            return str(_TOOLS[name]["fn"](arg))
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"


# ── Built-in tools ──────────────────────────────────────────────────────────
@tool
def calculator(expr: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * (4 + 1)'."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        return "error: only numbers and + - * / ( ) are allowed"
    return str(eval(expr, {"__builtins__": {}}))  # noqa: S307 - sanitised above


@tool
def word_count(text: str) -> str:
    """Count the words in a string."""
    return str(len(text.split()))
