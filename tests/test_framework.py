"""
Unit tests with fake backends — no network, no Ollama, no keys.

Runnable via `pytest` or directly: `python tests/test_framework.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cohortex.agent import Agent
from cohortex.docsource import DocumentSource
from cohortex.jsonutil import first_json
from cohortex.orchestrator import Crew
from cohortex.profiles import AgentProfile
from cohortex.prompts import build_messages
from cohortex.providers import FallbackBackend, register
from cohortex.runtime import build_agent
from cohortex.tools import (
    ToolRegistry,
    calculator,
    contrast_ratio,
    defang_iocs,
    make_dynamic_tool,
    shannon_entropy,
    word_count,
)


@register("test-capture")
class CaptureBackend:
    """Records the api_key/base_url it was constructed with — proves build_agent
    forwards AgentProfile.api_key/base_url through get_backend to the backend's
    own constructor, not just to callers that build backends by hand."""

    def __init__(self, model=None, api_key=None, base_url=None, **_):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def chat(self, messages, *, temperature=0.3, **opts):
        return "captured"


class ConstBackend:
    """Always returns the same string."""
    name = "const"
    model = "const"

    def __init__(self, text: str):
        self._text = text

    def chat(self, messages, *, temperature=0.3, **opts):
        return self._text


class UsageBackend:
    """Returns a fixed string and sets last_usage — proves Agent threads usage into meta."""
    name = "usage-test"
    model = "test"

    def __init__(self, text: str = "output"):
        self._text = text
        self.last_usage = None

    def chat(self, messages, *, temperature=0.3, **opts):
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return self._text


class ScriptedBackend:
    """Returns responses in order (last one repeats)."""
    name = "scripted"
    model = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def chat(self, messages, *, temperature=0.3, **opts):
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r


class MessageCaptureBackend:
    """Records the messages and opts it was called with — proves what actually
    reached the backend (e.g. num_ctx, full document text in the prompt)."""
    name = "capture-messages"
    model = "capture-messages"

    def __init__(self, text: str = "ok"):
        self._text = text
        self.last_messages = None
        self.last_opts = None

    def chat(self, messages, *, temperature=0.3, **opts):
        self.last_messages = messages
        self.last_opts = opts
        return self._text


class StreamingBackend:
    """Yields chunks then sets last_usage — the streaming counterpart to UsageBackend."""
    name = "streaming-test"
    model = "test"

    def __init__(self, chunks=("hello", " ", "world")):
        self._chunks = list(chunks)
        self.last_usage = None

    def chat(self, messages, *, temperature=0.3, **opts):
        return "".join(self._chunks)

    def chat_stream(self, messages, *, temperature=0.3, **opts):
        for c in self._chunks:
            yield c
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class ConnectFailBackend:
    """chat_stream fails before yielding anything — FallbackBackend should move
    on to the next backend, same as a plain chat() failure would."""
    name = "connect-fail"
    model = "test"

    def chat(self, messages, *, temperature=0.3, **opts):
        raise RuntimeError("should not be called")

    def chat_stream(self, messages, *, temperature=0.3, **opts):
        raise RuntimeError("connection refused")


class MidStreamFailBackend:
    """Yields one chunk then raises — FallbackBackend must NOT fall back here,
    since the caller already saw real output; retrying a different backend
    would duplicate/corrupt what's already been streamed out."""
    name = "midstream-fail"
    model = "test"

    def chat(self, messages, *, temperature=0.3, **opts):
        raise RuntimeError("should not be called")

    def chat_stream(self, messages, *, temperature=0.3, **opts):
        yield "partial"
        raise RuntimeError("boom")


def _agent(name, backend, tools=None):
    profile = AgentProfile(name=name, role=name, goal="do the job", tools=tools or [])
    reg = ToolRegistry(tools) if tools else None
    return Agent(profile, backend, tools=reg)


# ── profiles ────────────────────────────────────────────────────────────────
def test_profile_from_dict_ignores_unknown_keys():
    p = AgentProfile.from_dict(
        {"name": "r", "role": "Researcher", "goal": "g", "vaults": ["kb"], "bogus": 1}
    )
    assert p.name == "r" and p.role == "Researcher" and p.vaults == ["kb"]
    assert not hasattr(p, "bogus")


def test_build_agent_forwards_api_key_and_base_url_to_backend():
    profile = AgentProfile(
        name="r", role="Researcher", goal="g",
        backend="test-capture", api_key="secret-key", base_url="http://example.test",
    )
    agent = build_agent(profile)
    assert agent.backend.api_key == "secret-key"
    assert agent.backend.base_url == "http://example.test"


def test_build_agent_leaves_api_key_and_base_url_unset_by_default():
    profile = AgentProfile(name="r", role="Researcher", goal="g", backend="test-capture")
    agent = build_agent(profile)
    assert agent.backend.api_key is None
    assert agent.backend.base_url is None


# ── token accounting ────────────────────────────────────────────────────────
def test_agent_run_includes_usage_in_meta():
    a = _agent("solo", UsageBackend())
    result = a.run("task")
    assert result.meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_agent_run_without_usage_backend_has_no_usage_key():
    a = _agent("solo", ConstBackend("hi"))
    result = a.run("task")
    assert "usage" not in result.meta


def test_react_loop_accumulates_usage_across_tool_steps():
    backend = ScriptedBackend([
        '{"tool": "calculator", "input": "6 * 7"}',
        '{"answer": "42"}',
    ])
    backend.last_usage = None

    # Monkey-patch ScriptedBackend to set last_usage on each chat call
    original_chat = backend.chat
    call_count = [0]
    def chat_with_usage(messages, *, temperature=0.3, **opts):
        result = original_chat(messages, temperature=temperature, **opts)
        call_count[0] += 1
        backend.last_usage = {"prompt_tokens": 10 * call_count[0], "completion_tokens": 5, "total_tokens": 10 * call_count[0] + 5}
        return result
    backend.chat = chat_with_usage

    a = _agent("mathy", backend, tools=["calculator"])
    result = a.run("what is 6 times 7")
    assert result.meta["usage"]["prompt_tokens"] == 30  # 10 + 20
    assert result.meta["usage"]["completion_tokens"] == 10  # 5 + 5
    assert result.meta["usage"]["total_tokens"] == 40  # 15 + 25


# ── streaming ────────────────────────────────────────────────────────────────
def test_agent_run_stream_yields_deltas_then_done_with_correct_output_and_usage():
    a = _agent("solo", StreamingBackend(("hel", "lo ", "world")))
    events = list(a.run_stream("task"))

    deltas = [e for e in events if e["type"] == "delta"]
    done = [e for e in events if e["type"] == "done"]
    assert [d["text"] for d in deltas] == ["hel", "lo ", "world"]
    assert len(done) == 1
    result = done[0]["result"]
    assert result.output == "hello world"
    assert result.meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_agent_run_stream_falls_back_to_single_delta_without_chat_stream():
    a = _agent("solo", ConstBackend("hi there"))
    events = list(a.run_stream("task"))
    assert len(events) == 2
    assert events[0] == {"type": "delta", "text": "hi there"}
    assert events[1]["type"] == "done"
    assert events[1]["result"].output == "hi there"


def test_agent_run_stream_with_tools_yields_single_delta_and_done():
    backend = ScriptedBackend([
        '{"tool": "calculator", "input": "6 * 7"}',
        '{"answer": "42"}',
    ])
    a = _agent("mathy", backend, tools=["calculator"])
    events = list(a.run_stream("what is 6 times 7"))
    assert len(events) == 2
    assert events[0] == {"type": "delta", "text": "42"}
    assert events[1]["result"].output == "42"
    assert events[1]["result"].meta.get("tool_steps") == 2


def test_fallback_chat_stream_skips_backend_with_no_streaming_support():
    fb = FallbackBackend([ConstBackend("no-stream"), StreamingBackend(("ok",))])
    chunks = list(fb.chat_stream([{"role": "user", "content": "hi"}]))
    assert chunks == ["ok"]
    assert fb.last_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_fallback_chat_stream_falls_back_before_any_chunk_yielded():
    fb = FallbackBackend([ConnectFailBackend(), StreamingBackend(("ok",))])
    chunks = list(fb.chat_stream([{"role": "user", "content": "hi"}]))
    assert chunks == ["ok"]


def test_fallback_chat_stream_does_not_fall_back_mid_stream():
    fb = FallbackBackend([MidStreamFailBackend(), StreamingBackend(("should-not-appear",))])
    gen = fb.chat_stream([{"role": "user", "content": "hi"}])
    assert next(gen) == "partial"
    try:
        next(gen)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "midstream-fail" in str(e)


# ── long-context (opposite of RAG) ─────────────────────────────────────────
def test_document_source_loads_full_files_verbatim(tmp_path):
    (tmp_path / "a.md").write_text("alpha content", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta content", encoding="utf-8")
    source = DocumentSource("docs", dir=str(tmp_path))
    hits = source.load()
    assert len(hits) == 2
    assert all(h["full_context"] for h in hits)
    docs = {h["document"] for h in hits}
    assert docs == {"alpha content", "beta content"}
    assert source.char_count() == len("alpha content") + len("beta content")


def test_build_messages_renders_retrieved_and_full_context_separately():
    profile = AgentProfile(name="a", role="Analyst", goal="answer")
    hits = [
        {"document": "chunk text", "title": "chunk.md", "distance": 0.1},
        {"document": "whole file text", "title": "file.md", "full_context": True},
    ]
    messages = build_messages(profile, "the task", hits)
    user_content = messages[1]["content"]
    assert "## Context from knowledge vault(s)" in user_content
    assert "## Full source document(s)" in user_content
    assert "chunk text" in user_content
    assert "whole file text" in user_content


def test_agent_with_context_docs_sends_full_document_and_reports_chars(tmp_path):
    (tmp_path / "manual.md").write_text("the secret code is BLUE-42-FALCON", encoding="utf-8")
    source = DocumentSource("manual", dir=str(tmp_path))
    profile = AgentProfile(name="ops", role="Ops", goal="answer from the document",
                           context_docs=[str(tmp_path)])
    backend = MessageCaptureBackend("BLUE-42-FALCON")
    agent = Agent(profile, backend, doc_sources=[source])
    result = agent.run("what is the code?")

    assert "BLUE-42-FALCON" in backend.last_messages[1]["content"]
    assert "## Full source document(s)" in backend.last_messages[1]["content"]
    assert result.meta["full_context_chars"] == len("the secret code is BLUE-42-FALCON")


def test_agent_forwards_num_ctx_to_backend_when_set():
    profile = AgentProfile(name="ops", role="Ops", goal="answer", num_ctx=8192)
    backend = MessageCaptureBackend()
    agent = Agent(profile, backend)
    agent.run("task")
    assert backend.last_opts.get("num_ctx") == 8192


def test_agent_omits_num_ctx_when_unset():
    profile = AgentProfile(name="ops", role="Ops", goal="answer")
    backend = MessageCaptureBackend()
    agent = Agent(profile, backend)
    agent.run("task")
    assert "num_ctx" not in backend.last_opts


# ── tools ───────────────────────────────────────────────────────────────────
def test_builtin_tools():
    assert calculator("2 + 2 * 3") == "8"
    assert calculator("(3 + 4) * 2 - 1") == "13"
    assert calculator("import os").startswith("error")   # rejects non-arithmetic
    assert calculator("__import__('os')").startswith("error")
    assert word_count("the quick brown fox") == "4"


def test_first_json_handles_nested_and_selection():
    # Nested braces inside a value must not break extraction (regression for the
    # old flat-regex parser).
    obj = first_json('prefix {"agent": "foo", "task": "explain {x}"} suffix', ("agent", "final"))
    assert obj == {"agent": "foo", "task": "explain {x}"}
    # Skips objects that lack the required keys, returns the first that matches.
    assert first_json('{"other": 1} {"answer": "42"}', ("answer",)) == {"answer": "42"}
    assert first_json("no json here", ("answer",)) is None


def test_contrast_ratio_matches_known_wcag_values():
    # Black on white is the theoretical max (21:1); near-identical grays are near 1:1.
    assert contrast_ratio("000000, FFFFFF") == (
        "contrast ratio 21.00:1 — AA normal text: pass, AA large text: pass, "
        "AAA normal text: pass, AAA large text: pass"
    )
    low = contrast_ratio("777777, 808080")
    assert low.startswith("contrast ratio 1.1")
    assert "AA normal text: fail" in low
    # #2E86AB on white is a known real-world borderline case: passes for large
    # text (>=3:1) but fails the stricter normal-text AA threshold (4.5:1).
    borderline = contrast_ratio("2E86AB, FFFFFF")
    assert "AA normal text: fail" in borderline
    assert "AA large text: pass" in borderline
    # order shouldn't matter — ratio is symmetric
    assert contrast_ratio("FFFFFF, 000000") == contrast_ratio("000000, FFFFFF")
    # 3-digit shorthand hex
    assert contrast_ratio("000, fff") == contrast_ratio("000000, ffffff")


def test_contrast_ratio_rejects_bad_input():
    assert contrast_ratio("not-a-color, FFFFFF").startswith("error")
    assert contrast_ratio("FFFFFF").startswith("error")
    assert contrast_ratio("").startswith("error")


def test_shannon_entropy_ranks_repetitive_below_random():
    repetitive = shannon_entropy("a" * 64)
    english = shannon_entropy("the quick brown fox jumps over the lazy dog " * 3)
    assert repetitive.startswith("entropy: 0.00")
    assert "low" in repetitive
    assert "typical" in english
    # A uniform 256-byte-value spread is the theoretical max (8.0 bits/byte).
    # Round-trip through surrogateescape (the same handler shannon_entropy uses)
    # so all 256 raw byte values survive the str type unchanged — chr(i) for
    # i in range(256) would NOT do this, since UTF-8 multi-byte-encodes
    # codepoints 128-255 into overlapping continuation bytes.
    all_byte_values = bytes(range(256)).decode("utf-8", errors="surrogateescape")
    maximal = shannon_entropy(all_byte_values)
    assert "entropy: 8.00" in maximal
    assert "high" in maximal


def test_shannon_entropy_rejects_empty_input():
    assert shannon_entropy("") == "error: empty input"


def test_defang_iocs_neutralizes_urls_ips_and_domains():
    out = defang_iocs("Beacon at 1.2.3.4 called http://evil.example.com/gate.php")
    assert "1[.]2[.]3[.]4" in out
    assert "hxxp[://]" in out
    assert "evil[.]example[.]com" in out
    assert "http://" not in out
    # https keeps its 's' before the bracketed scheme separator
    assert "hxxps[://]" in defang_iocs("see https://sub.example.co.uk/x")


def test_defang_iocs_does_not_false_positive_on_ordinary_prose():
    # Single-letter TLD-shaped suffixes ('e.g.') and non-IPv4 dotted numbers
    # ('3.12') must survive untouched.
    text = "e.g. this is normal prose, and Python 3.12 is not an IP"
    assert defang_iocs(text) == text


def test_tool_registry_scoping():
    reg = ToolRegistry(["calculator"])
    assert reg.has("calculator")
    assert not reg.has("word_count")  # not in this agent's allowed set
    assert reg.run("calculator", "10 / 2") == "5.0"


# ── dynamic (Tool Shed) tools ────────────────────────────────────────────────
def _mock_public_dns(monkeypatch, ip="93.184.216.34"):
    """Dynamic-tool registration/calls resolve their host via socket.getaddrinfo —
    stub it so these tests never touch the real network, and so the resolved
    address is controllable (see the DNS-rebinding test below)."""
    import cohortex.tools as tools_mod
    box = {"ip": ip}
    monkeypatch.setattr(tools_mod.socket, "getaddrinfo",
                         lambda host, port: [(None, None, None, None, (box["ip"], 0))])
    return box


def test_make_dynamic_tool_builtin_kind_returns_none():
    # Builtins already live in the global registry — nothing to build.
    assert make_dynamic_tool("calculator", "", "builtin") is None


def test_make_dynamic_tool_rejects_missing_url_template():
    import pytest
    with pytest.raises(ValueError):
        make_dynamic_tool("lookup", "", "http", method="GET", url_template="")


def test_make_dynamic_tool_rejects_unsafe_literal_host():
    import pytest
    with pytest.raises(ValueError):
        make_dynamic_tool("lookup", "", "http", method="GET",
                           url_template="http://169.254.169.254/latest/meta-data/")


def test_make_dynamic_tool_rejects_input_controlled_host():
    # Letting the tool's own argument choose the host would make this an open
    # proxy — {input} is only allowed in the path/query.
    import pytest
    with pytest.raises(ValueError):
        make_dynamic_tool("fetch", "", "http", method="GET", url_template="https://{input}/")


def test_dynamic_tool_extra_is_instance_scoped_not_global(monkeypatch):
    from cohortex.tools import _TOOLS

    _mock_public_dns(monkeypatch)
    entry = make_dynamic_tool("lookup", "test tool", "http", method="GET",
                               url_template="https://example.com/{input}")
    reg = ToolRegistry(["lookup"], extra={"lookup": entry})
    assert reg.has("lookup")
    assert "lookup" not in _TOOLS  # never touches the shared global registry

    other_reg = ToolRegistry(["lookup"])  # no extra passed — simulates a concurrent run
    assert not other_reg.has("lookup")


def test_dynamic_http_tool_get_calls_configured_url(monkeypatch):
    _mock_public_dns(monkeypatch)
    calls = []

    class FakeResponse:
        text = "42 degrees"
        def raise_for_status(self):
            pass

    def fake_get(url, params=None, headers=None, timeout=None, follow_redirects=None):
        calls.append((url, params))
        return FakeResponse()

    # _http_tool_fn does a local `import httpx`, which resolves to the same
    # module object in sys.modules — patching it here takes effect there too.
    import httpx as real_httpx
    monkeypatch.setattr(real_httpx, "get", fake_get)

    entry = make_dynamic_tool("weather", "look up weather", "http", method="GET",
                               url_template="https://api.example.com/weather?city={input}")
    result = entry["fn"]("paris")
    assert result == "42 degrees"
    assert calls[0][0] == "https://api.example.com/weather?city=paris"


def test_dynamic_http_tool_revalidates_host_on_every_call_dns_rebinding(monkeypatch):
    # Registration-time resolution sees a public IP; a later call's resolution
    # returns a private one (DNS rebinding) — the call must still be blocked,
    # proving the safety check re-resolves on every call rather than caching.
    box = _mock_public_dns(monkeypatch, ip="93.184.216.34")

    entry = make_dynamic_tool("fetch", "", "http", method="GET",
                               url_template="https://rebind.example.com/status")

    box["ip"] = "127.0.0.1"
    result = entry["fn"]("ignored")
    assert result.startswith("error:")
    assert "not allowed" in result


def test_is_safe_url_blocks_private_and_loopback_and_metadata(monkeypatch):
    from cohortex.tools import _is_safe_url
    assert not _is_safe_url("http://127.0.0.1/")
    assert not _is_safe_url("http://localhost/")
    assert not _is_safe_url("http://10.0.0.5/")
    assert not _is_safe_url("http://169.254.169.254/")
    assert not _is_safe_url("http://metadata.google.internal/")
    assert not _is_safe_url("ftp://example.com/")
    _mock_public_dns(monkeypatch)
    assert _is_safe_url("https://example.com/path")


# ── agent (single) ────────────────────────────────────────────────────────────
def test_agent_single_run():
    a = _agent("solo", ConstBackend("hello world"))
    result = a.run("say hi")
    assert result.agent == "solo" and result.output == "hello world"


def test_agent_react_tool_loop():
    # Model asks for the calculator, then answers with the observation.
    backend = ScriptedBackend([
        '{"tool": "calculator", "input": "6 * 7"}',
        '{"answer": "42"}',
    ])
    a = _agent("mathy", backend, tools=["calculator"])
    result = a.run("what is 6 times 7")
    assert result.output == "42"
    assert result.meta.get("tool_steps") == 2


# ── orchestrator ──────────────────────────────────────────────────────────────
def test_sequential_order_and_handoff():
    b = ConstBackend("STEP")
    crew = Crew("c", [_agent("researcher", b), _agent("writer", b), _agent("editor", b)],
                topology="sequential")
    res = crew.run("task")
    assert [s.agent for s in res.steps] == ["researcher", "writer", "editor"]
    assert res.output == "STEP"


def test_sequential_truncation_applies_max_handoff_chars():
    class EchoUpstream:
        name = "echo"
        model = "echo"
        def chat(self, messages, *, temperature=0.3, **opts):
            user_msg = messages[-1]["content"] if messages else ""
            return "TRUNCATED" if "[truncated]" in user_msg else "NOT_TRUNCATED"

    crew = Crew("c", [_agent("a", ConstBackend("A" * 500)), _agent("b", EchoUpstream())],
                topology="sequential", max_handoff_chars=100)
    res = crew.run("task")
    assert res.output == "TRUNCATED"
    assert len(res.steps[0].output) == 500  # first agent's output is untruncated


def test_sequential_no_truncation_when_under_limit():
    crew = Crew("c", [_agent("a", ConstBackend("short")), _agent("b", ConstBackend("ok"))],
                topology="sequential", max_handoff_chars=100)
    res = crew.run("task")
    assert res.output == "ok"


def test_supervisor_delegates_then_finishes():
    supervisor = Agent(
        AgentProfile(name="supervisor", role="Supervisor", goal="route"),
        ScriptedBackend([
            '{"agent": "mathematician", "task": "2+2"}',
            '{"final": "the answer is 4"}',
        ]),
    )
    math = _agent("mathematician", ConstBackend("4"))
    crew = Crew("c", [math], topology="supervisor", supervisor=supervisor)
    res = crew.run("what is 2 + 2")
    assert res.output == "the answer is 4"
    assert any(s.agent == "mathematician" for s in res.steps)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)
