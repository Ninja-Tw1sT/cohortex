"""
Agent = profile + backend + bound vault(s) + tools.

On run(): search bound vaults for context, build a role prompt, call the backend.
If the profile lists tools, run a ReAct loop instead (adapts the portfolio's
example_agent.py — small open models rarely support the native tools API).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from cohortex.jsonutil import first_json
from cohortex.profiles import AgentProfile
from cohortex.prompts import build_messages, build_system
from cohortex.providers import LLMBackend
from cohortex.tools import ToolRegistry
from cohortex.vault import KnowledgeVault


@dataclass
class AgentResult:
    agent: str
    output: str
    raw: str = ""
    meta: dict = field(default_factory=dict)


class Agent:
    def __init__(self, profile: AgentProfile, backend: LLMBackend,
                 vaults: list[KnowledgeVault] | None = None,
                 tools: ToolRegistry | None = None):
        self.profile = profile
        self.backend = backend
        self.vaults = vaults or []
        self.tools = tools

    def _gather_context(self, task: str) -> list[dict]:
        hits: list[dict] = []
        for v in self.vaults:
            hits.extend(v.search(task))
        return hits

    def run(self, task: str, upstream: str = "") -> AgentResult:
        hits = self._gather_context(task)
        if self.tools and self.profile.tools:
            return self._run_react(task, hits, upstream)
        messages = build_messages(self.profile, task, hits, upstream)
        out = self.backend.chat(messages, temperature=self.profile.temperature,
                                max_tokens=self.profile.max_tokens)
        return AgentResult(self.profile.name, out.strip(), out,
                           {"backend": self.backend.name, "context_hits": len(hits)})

    def _run_react(self, task, hits, upstream, max_steps: int = 5) -> AgentResult:
        specs = self.tools.specs()
        tool_desc = "\n".join(f"- {n}: {d}" for n, d in specs.items())
        system = (
            build_system(self.profile)
            + f"\n\nYou can call tools. Available:\n{tool_desc}\n\n"
            + 'Reply with ONLY one JSON object. To use a tool: '
            + '{"tool": "<name>", "input": "<arg>"}. To finish: {"answer": "<final answer>"}.'
        )
        user = build_messages(self.profile, task, hits, upstream)[1]["content"]
        transcript = [{"role": "system", "content": system},
                      {"role": "user", "content": user}]

        for step in range(1, max_steps + 1):
            raw = self.backend.chat(transcript, temperature=self.profile.temperature,
                                    max_tokens=self.profile.max_tokens)
            action = first_json(raw, ("tool", "answer"))
            if not action:
                return AgentResult(self.profile.name, raw.strip(), raw, {"parse_error": True})
            if "answer" in action:
                return AgentResult(self.profile.name, str(action["answer"]).strip(), raw,
                                   {"tool_steps": step, "backend": self.backend.name})
            name, arg = action.get("tool"), str(action.get("input", ""))
            obs = self.tools.run(name, arg) if self.tools.has(name) else f"error: unknown tool {name!r}"
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append({
                "role": "user",
                "content": f'Observation: {obs}. If this answers the task, reply with '
                           f'{{"answer": "{obs}"}} and nothing else.',
            })
        return AgentResult(self.profile.name, "(reached tool-step limit)", "",
                           {"tool_steps": max_steps})
