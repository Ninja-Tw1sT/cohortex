"""
Long-context agent — the structural opposite of single_agent.py's RAG demo.

Instead of embedding + retrieving top-k chunks from a KnowledgeVault, this
loads an entire document verbatim into the agent's context window (via
DocumentSource) and lets the model read all of it — a classic "needle in a
haystack" long-context recall test. No embeddings, no chunking, no ranking:
nothing is left out because nothing was ranked away — the tradeoff is token
cost, not recall.
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cohortex.agent import Agent
from cohortex.docsource import DocumentSource
from cohortex.profiles import AgentProfile
from cohortex.providers import get_backend

NEEDLE = "The Q3 password-rotation exception code is BLUE-42-FALCON."

_FILLER = (
    "Standard incident response begins with triage: confirm the alert, assess "
    "blast radius, and notify the on-call rotation before touching production. "
    "Change requests outside the maintenance window require director sign-off. "
    "Backups are verified nightly and restored to a scratch environment weekly "
    "to confirm recoverability, not just that the job exited zero."
)


def _write_haystack(dir_: pathlib.Path, needle_at_pct: float = 0.6, target_chars: int = 12000) -> None:
    """Build a synthetic multi-section document with one unique fact buried
    partway through — the standard "needle in a haystack" long-context probe."""
    paragraphs: list[str] = []
    chars = 0
    i = 0
    needle_inserted = False
    while chars < target_chars:
        if not needle_inserted and chars >= target_chars * needle_at_pct:
            paragraphs.append(NEEDLE)
            needle_inserted = True
        section = f"Section {i}: {_FILLER}"
        paragraphs.append(section)
        chars += len(section)
        i += 1
    (dir_ / "it_ops_manual.md").write_text("\n\n".join(paragraphs), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        doc_dir = pathlib.Path(tmp)
        _write_haystack(doc_dir)

        source = DocumentSource("it_ops_manual", dir=str(doc_dir))
        print(f"Generated haystack document: {source.char_count():,} chars, needle buried ~60% through.")

        # No `vaults=` here — context_docs bypasses retrieval entirely.
        profile = AgentProfile(
            name="ops_analyst",
            role="IT Operations Analyst",
            goal="answer using only the full document provided — no retrieval, no guessing",
            context_docs=[str(doc_dir)],
            num_ctx=8192,  # bump Ollama's context window past its 2048 default
            temperature=0.0,
        )
        agent = Agent(profile, get_backend(profile.backend, profile.model), doc_sources=[source])

        question = "What is the Q3 password-rotation exception code?"
        result = agent.run(question)

        print(f"Backend: {agent.backend.name}  |  full-context chars sent: "
              f"{result.meta.get('full_context_chars', 0):,}")
        print(f"Q: {question}")
        print(f"A: {result.output}")
        print(f"Needle recalled correctly: {'BLUE-42-FALCON' in result.output}")


if __name__ == "__main__":
    main()
