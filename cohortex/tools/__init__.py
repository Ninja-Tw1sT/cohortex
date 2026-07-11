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
import math
import operator
import re
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


def _parse_hex_color(raw: str) -> tuple[int, int, int]:
    s = raw.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(f"{raw!r} is not a valid hex color")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _relative_luminance(r: int, g: int, b: int) -> float:
    def lin(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


@tool
def contrast_ratio(pair: str) -> str:
    """Compute the WCAG 2.x contrast ratio between two hex colors, e.g. '2E86AB, FFFFFF', and report AA/AAA pass/fail for normal and large text."""
    parts = [p for p in re.split(r"[,\s]+", pair.strip()) if p]
    if len(parts) != 2:
        return "error: expected two hex colors separated by a comma or space, e.g. '2E86AB, FFFFFF'"
    try:
        c1, c2 = _parse_hex_color(parts[0]), _parse_hex_color(parts[1])
    except ValueError as e:
        return f"error: {e}"
    l1, l2 = _relative_luminance(*c1), _relative_luminance(*c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    ratio = (lighter + 0.05) / (darker + 0.05)
    aa_normal = "pass" if ratio >= 4.5 else "fail"
    aa_large = "pass" if ratio >= 3.0 else "fail"
    aaa_normal = "pass" if ratio >= 7.0 else "fail"
    aaa_large = "pass" if ratio >= 4.5 else "fail"
    return (
        f"contrast ratio {ratio:.2f}:1 — "
        f"AA normal text: {aa_normal}, AA large text: {aa_large}, "
        f"AAA normal text: {aaa_normal}, AAA large text: {aaa_large}"
    )


@tool
def shannon_entropy(text: str) -> str:
    """Compute the Shannon entropy (bits/byte) of a string's UTF-8 bytes — a standard static-analysis signal for detecting packed, encrypted, or compressed content (plain text/code typically runs ~4-6 bits/byte; ~7.2-8 suggests packing or encryption)."""
    data = text.encode("utf-8", errors="surrogateescape")
    if not data:
        return "error: empty input"
    counts: dict[int, int] = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    n = len(data)
    entropy = max(0.0, -sum((c / n) * math.log2(c / n) for c in counts.values()))
    if entropy >= 7.2:
        verdict = "high — consistent with compressed, encrypted, or packed data"
    elif entropy >= 6.0:
        verdict = "moderate-high — denser than typical plain text, worth a closer look"
    elif entropy >= 3.5:
        verdict = "typical — consistent with plain text or uncompressed code"
    else:
        verdict = "low — consistent with highly repetitive or uniform data"
    return f"entropy: {entropy:.2f} bits/byte (max 8.0) — {verdict}"


_DEFANG_SCHEME_RE = re.compile(r"(?i)\bhttp(s?)://")
_DEFANG_HOST_RE = re.compile(
    r"\b(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}|(?:\d{1,3}\.){3}\d{1,3})\b"
)


@tool
def defang_iocs(text: str) -> str:
    """Defang IPs, domains, and URLs in a block of text (e.g. 1.2.3.4 -> 1[.]2[.]3[.]4, http:// -> hxxp[://]) so indicators of compromise can be shared in a report without becoming live, clickable links."""
    text = _DEFANG_SCHEME_RE.sub(lambda m: f"hxxp{m.group(1)}[://]", text)
    text = _DEFANG_HOST_RE.sub(lambda m: m.group(0).replace(".", "[.]"), text)
    return text


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
