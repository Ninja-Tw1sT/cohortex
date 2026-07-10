"""
A tiny tool registry. Decorate a function with @tool and it becomes available to
any agent whose profile lists it. Agents call tools via a ReAct loop (see agent.py).

Alongside the static @tool-decorated builtins (module-global, safe to share), a
ToolRegistry can also carry an `extra` map of dynamically-built tools (e.g. Tool
Shed's user-defined HTTP tools). `extra` lives on the ToolRegistry *instance*, not
the global _TOOLS dict — callers building agents for concurrent, multi-tenant runs
(Cohortex Studio's sidecar) must pass a fresh dict per run. Mutating _TOOLS itself
would let one run's tool definition silently overwrite another's mid-flight.
"""
from __future__ import annotations

import ast
import ipaddress
import operator
import socket
from typing import Callable
from urllib.parse import urlsplit

_TOOLS: dict[str, dict] = {}


def tool(fn: Callable | None = None, *, name: str | None = None):
    """Register a single-string-argument tool. Its docstring is shown to the agent."""
    def deco(f: Callable):
        _TOOLS[name or f.__name__] = {"fn": f, "doc": (f.__doc__ or "").strip().splitlines()[0]}
        return f
    return deco(fn) if fn else deco


class ToolRegistry:
    """A per-agent view over the global tool set (only the names the profile lists),
    plus an optional instance-scoped `extra` map of dynamically-built tools that
    takes precedence over (and never touches) the global registry."""

    def __init__(self, names: list[str] | None = None, extra: dict[str, dict] | None = None):
        self._names = list(names) if names else list(_TOOLS)
        self._extra = extra or {}

    def _lookup(self, name: str) -> dict | None:
        return self._extra.get(name) or _TOOLS.get(name)

    def specs(self) -> dict[str, str]:
        out = {}
        for n in self._names:
            entry = self._lookup(n)
            if entry:
                out[n] = entry["doc"]
        return out

    def has(self, name: str) -> bool:
        return name in self._names and self._lookup(name) is not None

    def run(self, name: str, arg: str) -> str:
        entry = self._lookup(name)
        if entry is None:
            return f"error: unknown tool {name!r}"
        try:
            return str(entry["fn"](arg))
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"


# ── Built-in tools ──────────────────────────────────────────────────────────
_ARITH_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_arith(node):
    """Evaluate an arithmetic AST with a strict node allowlist (no eval, no names)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_arith(node.left), _safe_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_arith(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expr: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * (4 + 1)'."""
    try:
        return str(_safe_arith(ast.parse(expr, mode="eval").body))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return "error: not a valid arithmetic expression"


@tool
def word_count(text: str) -> str:
    """Count the words in a string."""
    return str(len(text.split()))


# ── Dynamic HTTP tools (Tool Shed) ──────────────────────────────────────────
# User-defined tools that call an external URL. Never registered globally —
# see the ToolRegistry docstring above for why.
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def _is_safe_host(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() in _BLOCKED_HOSTNAMES:
        return False
    try:
        return _is_public_ip(ipaddress.ip_address(host))
    except ValueError:
        pass  # not a literal IP — resolve it and check every address below
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        addr = info[4][0].split("%")[0]  # strip IPv6 zone id
        try:
            if not _is_public_ip(ipaddress.ip_address(addr)):
                return False
        except ValueError:
            return False
    return True


def _is_safe_url(url: str) -> bool:
    """Reject anything but a plain http(s) URL whose host resolves only to public
    IPs. Re-checked (re-resolved) on every call, not just at tool-creation time —
    DNS for an initially-safe host can be repointed at an internal address between
    registration and a later run ("DNS rebinding")."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    return _is_safe_host(parts.hostname)


def _http_tool_fn(method: str, url_template: str, headers: dict[str, str]) -> Callable[[str], str]:
    import httpx

    def _run(arg: str) -> str:
        try:
            url = url_template.format(input=arg) if "{input}" in url_template else url_template
        except (KeyError, IndexError, ValueError):
            return "error: malformed url template"
        if not _is_safe_url(url):
            return "error: URL not allowed (blocked host or private/internal IP range)"
        try:
            if method == "POST":
                resp = httpx.post(url, json={"input": arg}, headers=headers, timeout=15.0, follow_redirects=False)
            else:
                params = None if "{input}" in url_template else {"q": arg}
                resp = httpx.get(url, params=params, headers=headers, timeout=15.0, follow_redirects=False)
            resp.raise_for_status()
            return resp.text[:2000]
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"

    return _run


def make_dynamic_tool(name: str, description: str, kind: str, **config) -> dict | None:
    """Build a {'fn', 'doc'} entry for a ToolRegistry's `extra` map. Returns None
    for kind="builtin" (those already live in the global registry under their own
    name — nothing to build)."""
    if kind == "builtin":
        return None
    if kind != "http":
        raise ValueError(f"tool {name!r}: unknown kind {kind!r}")

    method = (config.get("method") or "GET").upper()
    if method not in ("GET", "POST"):
        raise ValueError(f"tool {name!r}: unsupported method {method!r}")
    url_template = config.get("url_template") or ""
    if not url_template:
        raise ValueError(f"tool {name!r}: http tool requires a url_template")
    try:
        parts = urlsplit(url_template)
    except ValueError:
        raise ValueError(f"tool {name!r}: malformed url_template")
    # The host must be a fixed literal — letting the agent's own tool argument
    # steer which host gets called would make this an open proxy.
    if "{input}" in (parts.netloc or ""):
        raise ValueError(f"tool {name!r}: url_template's host may not depend on {{input}} "
                          f"— only the path/query may use it")
    if not _is_safe_host(parts.hostname):
        raise ValueError(f"tool {name!r}: url_template points to a blocked host/IP range")
    headers = dict(config.get("headers") or {})

    doc = (description or f"{method} {url_template}").strip().splitlines()[0]
    return {"fn": _http_tool_fn(method, url_template, headers), "doc": doc}
