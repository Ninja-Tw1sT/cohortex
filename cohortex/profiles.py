"""AgentProfile — the config that defines an agent (role, backend, vaults, tools)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from cohortex.config import load_yaml


@dataclass
class AgentProfile:
    name: str
    role: str = ""
    goal: str = ""
    backend: str | None = None      # None → global default backend
    model: str | None = None        # None → backend's default model
    temperature: float = 0.3
    system_prompt: str = ""
    vaults: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentProfile":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_profile(path: str | Path) -> AgentProfile:
    path = Path(path)
    data = load_yaml(path)
    data.setdefault("name", path.stem)
    return AgentProfile.from_dict(data)


def load_profiles_dir(directory: str | Path) -> dict[str, AgentProfile]:
    out: dict[str, AgentProfile] = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        prof = load_profile(p)
        out[prof.name] = prof
    return out
