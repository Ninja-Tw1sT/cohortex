"""
Single configurable agent bound to a knowledge vault — fully local, no API key.

Shows the three moving parts: a KnowledgeVault (retrieval), an AgentProfile
(role + backend + vault binding), and an Agent that ties them together.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vaultmind.agent import Agent
from vaultmind.profiles import AgentProfile
from vaultmind.providers import get_backend
from vaultmind.vault import KnowledgeVault


def main() -> None:
    # 1. A small in-memory knowledge vault.
    vault = KnowledgeVault("demo_kb", persistent=False)
    vault.add(
        [
            "VaultMind is a modular multi-agent framework with pluggable LLM backends.",
            "A KnowledgeVault wraps a ChromaDB collection and returns top-k context for a query.",
            "An agent's backend is selectable: ollama (local), openai, anthropic, gemini, or grok.",
        ],
        metadatas=[{"title": "overview"}, {"title": "vault"}, {"title": "backends"}],
    )

    # 2. A profile: role, which backend (None = global default = local Ollama), which vaults.
    profile = AgentProfile(
        name="researcher",
        role="Research Analyst",
        goal="answer using only the vault context",
        vaults=["demo_kb"],
        temperature=0.1,
    )

    # 3. Wire it up and run.
    agent = Agent(profile, get_backend(profile.backend, profile.model), vaults=[vault])
    question = "What backends can a VaultMind agent use?"
    result = agent.run(question)

    print(f"Backend: {agent.backend.name}  |  context hits: {result.meta.get('context_hits')}")
    print(f"Q: {question}")
    print(f"A: {result.output}")


if __name__ == "__main__":
    main()
