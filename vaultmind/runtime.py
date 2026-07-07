"""
Runtime — assemble agents and crews from YAML config and run them.

    from vaultmind.runtime import run_crew
    result = run_crew("research_team", "Explain vector databases")
"""
from __future__ import annotations

from vaultmind import config
from vaultmind.agent import Agent
from vaultmind.orchestrator import Crew, CrewResult
from vaultmind.profiles import AgentProfile, load_profiles_dir
from vaultmind.providers import get_backend
from vaultmind.tools import ToolRegistry
from vaultmind.vault import KnowledgeVault

_VAULT_CACHE: dict[str, KnowledgeVault] = {}


def _vault_defs() -> dict:
    p = config.CONFIG_DIR / "vaults.yaml"
    return (config.load_yaml(p).get("vaults", {}) or {}) if p.exists() else {}


def get_vault(name: str, defs: dict | None = None) -> KnowledgeVault:
    if name in _VAULT_CACHE:
        return _VAULT_CACHE[name]
    defs = _vault_defs() if defs is None else defs
    d = defs.get(name, {})
    v = KnowledgeVault(name=name, collection=d.get("collection", name), db_path=d.get("db_path"))
    _VAULT_CACHE[name] = v
    return v


def build_agent(profile: AgentProfile, defs: dict | None = None) -> Agent:
    defs = _vault_defs() if defs is None else defs
    backend = get_backend(profile.backend, profile.model)
    vaults = [get_vault(n, defs) for n in profile.vaults]
    tools = ToolRegistry(profile.tools) if profile.tools else None
    return Agent(profile, backend, vaults, tools)


def load_crew(name: str) -> Crew:
    crew_cfg = config.load_yaml(config.CONFIG_DIR / "crews" / f"{name}.yaml")
    profiles = load_profiles_dir(config.CONFIG_DIR / "agents")
    defs = _vault_defs()
    agents = [build_agent(profiles[n], defs) for n in crew_cfg.get("agents", [])]
    supervisor = None
    if crew_cfg.get("supervisor"):
        supervisor = build_agent(profiles[crew_cfg["supervisor"]], defs)
    return Crew(
        name,
        agents,
        topology=crew_cfg.get("topology", "sequential"),
        supervisor=supervisor,
        max_rounds=crew_cfg.get("max_rounds", 4),
    )


def run_crew(name: str, task: str) -> CrewResult:
    return load_crew(name).run(task)
