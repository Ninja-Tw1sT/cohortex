"""
Streaming agent — the same single-agent setup as single_agent.py, but consuming
Agent.run_stream() instead of run() so output prints token-by-token as the
model generates it, rather than all at once when the call finishes.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cohortex.agent import Agent
from cohortex.profiles import AgentProfile
from cohortex.providers import get_backend
from cohortex.vault import KnowledgeVault


def main() -> None:
    vault = KnowledgeVault("demo_kb", persistent=False)
    vault.add(
        [
            "Cohortex is a modular multi-agent framework with pluggable LLM backends.",
            "A KnowledgeVault wraps a ChromaDB collection and returns top-k context for a query.",
            "An agent's backend is selectable: ollama (local), openai, anthropic, gemini, or grok.",
        ],
        metadatas=[{"title": "overview"}, {"title": "vault"}, {"title": "backends"}],
    )

    profile = AgentProfile(
        name="researcher", role="Research Analyst",
        goal="answer using only the vault context", vaults=["demo_kb"], temperature=0.1,
    )
    agent = Agent(profile, get_backend(profile.backend, profile.model), vaults=[vault])

    question = "What backends can a Cohortex agent use?"
    print(f"Q: {question}")
    print("A: ", end="", flush=True)

    result = None
    for event in agent.run_stream(question):
        if event["type"] == "delta":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            result = event["result"]
    print()

    print(f"\nBackend: {agent.backend.name}  |  context hits: {result.meta.get('context_hits')}"
          f"  |  usage: {result.meta.get('usage')}")


if __name__ == "__main__":
    main()
