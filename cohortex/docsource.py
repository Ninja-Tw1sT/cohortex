"""
DocumentSource — the opposite of a KnowledgeVault: loads whole documents
verbatim (no embeddings, no chunking, no top-k retrieval) so an agent reasons
over the full text using the backend's context window. Trades token cost for
guaranteed recall — nothing is left out because nothing was ranked away.

Same Hit shape as KnowledgeVault.search() (id/document/source/title) so
Agent._gather_context and build_messages handle both uniformly; a
`full_context: True` flag on each hit is what tells build_messages to render
it as "entire file" content instead of "retrieved chunk" content.
"""
from __future__ import annotations

from pathlib import Path

Hit = dict


class DocumentSource:
    def __init__(self, name: str, paths: list[str] | None = None,
                 dir: str | None = None, glob: str = "*.*"):
        self.name = name
        self._paths: list[Path] = []
        if paths:
            self._paths.extend(Path(p) for p in paths)
        if dir:
            self._paths.extend(sorted(Path(dir).glob(glob)))
        if not self._paths:
            raise ValueError(f"DocumentSource {name!r} has no files (checked paths= and dir=).")

    def load(self) -> list[Hit]:
        hits: list[Hit] = []
        for p in self._paths:
            text = p.read_text(encoding="utf-8", errors="replace")
            hits.append({
                "id": f"{self.name}-{p.stem}",
                "document": text,
                "source": str(p),
                "title": p.name,
                "full_context": True,
            })
        return hits

    def char_count(self) -> int:
        return sum(len(h["document"]) for h in self.load())
