"""
Unit tests with fake backends — no network, no Ollama, no keys.

Runnable via `pytest` or directly: `python tests/test_framework.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cohortex.agent import Agent
from cohortex.jsonutil import first_json
from cohortex.orchestrator import Crew
from cohortex.profiles import AgentProfile
from cohortex.providers import register
from cohortex.runtime import build_agent
from cohortex.tools import ToolRegistry, calculator, word_count


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


def test_tool_registry_scoping():
    reg = ToolRegistry(["calculator"])
    assert reg.has("calculator")
    assert not reg.has("word_count")  # not in this agent's allowed set
    assert reg.run("calculator", "10 / 2") == "5.0"


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
