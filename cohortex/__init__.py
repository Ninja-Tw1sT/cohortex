"""
Cohortex — a modular, provider-agnostic multi-agent framework.

Agents are defined by config (role/goal/backend/vaults/tools). Backends are
selectable per-agent (local Ollama or cloud APIs) with a global default. Each
agent can bind to one or more knowledge vaults (ChromaDB collections), and
crews run them in single, sequential, or supervisor topologies.

    from cohortex.runtime import run_crew
    result = run_crew("research_team", "Summarize vector databases")
"""

__version__ = "0.1.0"
