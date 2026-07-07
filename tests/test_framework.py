"""
Unit tests with fake backends — no network, no Ollama, no keys.

Runnable via `pytest` or directly: `python tests/test_framework.py`.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cohortex.agent import Agent
from cohortex.orchestrator import Crew
from cohortex.profiles import AgentProfile
from cohortex.tools import ToolRegistry, calculator, word_count


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


# ── tools ───────────────────────────────────────────────────────────────────
def test_builtin_tools():
    assert calculator("2 + 2 * 3") == "8"
    assert calculator("import os") == "error: only numbers and + - * / ( ) are allowed"
    assert word_count("the quick brown fox") == "4"


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
