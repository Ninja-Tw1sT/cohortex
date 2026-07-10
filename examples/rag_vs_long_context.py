"""
RAG vs. long-context — same corpus, same question, two different strategies,
run side by side so the tradeoff is visible instead of asserted.

RAG (KnowledgeVault): chunks the corpus, embeds each chunk, and retrieves only
the top-k most similar chunks — cheap per call, but can miss a fact if its
chunk doesn't embed close to the query.

Long-context (DocumentSource): loads every chunk into the context window, no
ranking, nothing left behind — costlier per call, but recall is bounded only
by the model's ability to read, not by a similarity search.
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cohortex.agent import Agent
from cohortex.docsource import DocumentSource
from cohortex.profiles import AgentProfile
from cohortex.providers import get_backend
from cohortex.vault import KnowledgeVault

NEEDLE = "The Q3 password-rotation exception code is BLUE-42-FALCON."
FILLER_TOPICS = [
    "Backups are verified nightly and restored to a scratch environment weekly.",
    "Change requests outside the maintenance window require director sign-off.",
    "On-call rotation handoffs happen every Monday at 09:00 local time.",
    "New laptops are imaged from the golden image before being shipped to hires.",
    "VPN split-tunneling is disabled by default for all remote employees.",
    "Incident postmortems are published within five business days of resolution.",
]


def _build_paragraphs(n: int = 40) -> list[str]:
    paragraphs = [f"Section {i}: {FILLER_TOPICS[i % len(FILLER_TOPICS)]}" for i in range(n)]
    paragraphs.insert(int(n * 0.6), NEEDLE)
    return paragraphs


def run_rag(paragraphs: list[str], question: str) -> tuple[str, dict]:
    vault = KnowledgeVault("it_ops_rag", persistent=False)
    vault.add(paragraphs, metadatas=[{"title": f"section-{i}"} for i in range(len(paragraphs))])
    profile = AgentProfile(
        name="rag_analyst", role="IT Operations Analyst",
        goal="answer using only the retrieved context", vaults=["it_ops_rag"], temperature=0.0,
    )
    agent = Agent(profile, get_backend(profile.backend, profile.model), vaults=[vault])
    result = agent.run(question)
    return result.output, result.meta


def run_long_context(paragraphs: list[str], question: str) -> tuple[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        doc_dir = pathlib.Path(tmp)
        (doc_dir / "it_ops_manual.md").write_text("\n\n".join(paragraphs), encoding="utf-8")
        source = DocumentSource("it_ops_manual", dir=str(doc_dir))
        profile = AgentProfile(
            name="longctx_analyst", role="IT Operations Analyst",
            goal="answer using only the full document provided",
            context_docs=[str(doc_dir)], num_ctx=8192, temperature=0.0,
        )
        agent = Agent(profile, get_backend(profile.backend, profile.model), doc_sources=[source])
        result = agent.run(question)
        return result.output, result.meta


def main() -> None:
    paragraphs = _build_paragraphs()
    question = "What is the Q3 password-rotation exception code?"

    rag_answer, rag_meta = run_rag(paragraphs, question)
    lc_answer, lc_meta = run_long_context(paragraphs, question)

    print(f"Corpus: {len(paragraphs)} paragraphs, needle buried at ~60%.\nQ: {question}\n")

    print("-- RAG (top-k retrieval) --------------------------")
    print(f"context hits: {rag_meta.get('context_hits')}  |  usage: {rag_meta.get('usage')}")
    print(f"A: {rag_answer}")
    print(f"Needle recalled: {'BLUE-42-FALCON' in rag_answer}\n")

    print("-- Long-context (full document, no retrieval) -----")
    print(f"full-context chars: {lc_meta.get('full_context_chars')}  |  usage: {lc_meta.get('usage')}")
    print(f"A: {lc_answer}")
    print(f"Needle recalled: {'BLUE-42-FALCON' in lc_answer}")


if __name__ == "__main__":
    main()
