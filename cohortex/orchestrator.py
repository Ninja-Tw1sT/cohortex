"""
Crew orchestration. Three topologies:

  single      — one agent answers.
  sequential  — agents run in order, each fed the previous one's output
                (adapts the portfolio's researcher → writer → editor example).
  supervisor  — a supervisor agent repeatedly delegates subtasks to named
                specialists, then writes the final answer (the JARVIS pattern).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from cohortex.agent import Agent, AgentResult


@dataclass
class CrewResult:
    crew: str
    output: str
    steps: list[AgentResult] = field(default_factory=list)


class Crew:
    def __init__(self, name: str, agents: list[Agent], topology: str = "sequential",
                 supervisor: Agent | None = None, max_rounds: int = 4):
        self.name = name
        self.agents = agents
        self.topology = topology
        self.supervisor = supervisor
        self.max_rounds = max_rounds
        self._by_name = {a.profile.name: a for a in agents}

    def run(self, task: str) -> CrewResult:
        if self.topology == "single":
            r = self.agents[0].run(task)
            return CrewResult(self.name, r.output, [r])
        if self.topology == "sequential":
            return self._sequential(task)
        if self.topology == "supervisor":
            return self._supervisor(task)
        raise ValueError(f"Unknown topology {self.topology!r}")

    def _sequential(self, task: str) -> CrewResult:
        steps: list[AgentResult] = []
        upstream = ""
        for a in self.agents:
            r = a.run(task, upstream=upstream)
            steps.append(r)
            upstream = r.output
        return CrewResult(self.name, steps[-1].output if steps else "", steps)

    def _supervisor(self, task: str) -> CrewResult:
        if not self.supervisor:
            raise ValueError("supervisor topology requires a supervisor agent")
        roster = "\n".join(
            f"- {n}: {a.profile.role or 'agent'} — {a.profile.goal}"
            for n, a in self._by_name.items()
        )
        steps: list[AgentResult] = []
        notes: list[str] = []
        for _ in range(self.max_rounds):
            sup_prompt = (
                f"User request: {task}\n\n"
                f"Specialists you can delegate to:\n{roster}\n\n"
                f"Work so far:\n{chr(10).join(notes) or '(none yet)'}\n\n"
                'Reply with ONLY one JSON object. To delegate: '
                '{"agent": "<name>", "task": "<subtask>"}. '
                'When you have enough to answer: {"final": "<answer to the user>"}.'
            )
            raw = self.supervisor.run(sup_prompt).output
            action = _find_action(raw, ("agent", "final"))
            if not action or "final" in action:
                final = action["final"] if action and "final" in action else raw
                return CrewResult(self.name, str(final).strip(), steps)
            name, sub = action.get("agent"), action.get("task", task)
            agent = self._by_name.get(name)
            if not agent:
                notes.append(f"[supervisor named unknown agent {name!r}]")
                continue
            r = agent.run(str(sub))
            steps.append(r)
            notes.append(f"{name} → {r.output}")

        # Ran out of rounds — ask the supervisor to synthesize what we have.
        r = self.supervisor.run(
            f"User request: {task}\n\nSpecialist results:\n{chr(10).join(notes)}\n\n"
            "Write the final answer to the user."
        )
        steps.append(r)
        return CrewResult(self.name, r.output, steps)


def _find_action(text: str, keys: tuple[str, ...]) -> dict | None:
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in keys):
            return obj
    return None
