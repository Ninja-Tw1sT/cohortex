"""
KnowledgeVault — a named retrieval source backed by a ChromaDB collection.

Bind an agent to one or more vaults; on each run it searches them for context.
A vault can point at a brand-new store or an existing ChromaDB (so you can reuse,
say, ai-workflow's `obsidian_vault` collection).
"""
from __future__ import annotations

from pathlib import Path

from vaultmind import config
from vaultmind.embeddings import Embedder, get_embedder

Hit = dict  # {id, document, source, title, distance}


class KnowledgeVault:
    def __init__(self, name: str, collection: str | None = None,
                 db_path: str | None = None, embedder: Embedder | None = None,
                 persistent: bool = True):
        import chromadb
        from chromadb.config import Settings

        if len(name) < 3:
            raise ValueError("Vault name must be at least 3 characters (ChromaDB rule).")
        self.name = name
        coll_name = collection or name

        if persistent:
            path = str(db_path or config.VAULT_DB_PATH)
            Path(path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False),
            )
        else:
            client = chromadb.EphemeralClient()

        self._col = client.get_or_create_collection(
            coll_name, metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder or get_embedder()

    def add(self, documents: list[str], ids: list[str] | None = None,
            metadatas: list[dict] | None = None) -> None:
        ids = ids or [f"{self.name}-{i}" for i in range(len(documents))]
        embeddings = self._embedder.embed_batch(documents)
        self._col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def count(self) -> int:
        return self._col.count()

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or config.TOP_K_DEFAULT
        if self._col.count() == 0:
            return []
        qv = self._embedder.embed(query)
        res = self._col.query(
            query_embeddings=[qv], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0] or [{}] * len(docs)
        dists = res["distances"][0]
        ids = res["ids"][0]
        hits: list[Hit] = []
        for doc, meta, dist, rid in zip(docs, metas, dists, ids):
            meta = meta or {}
            hits.append({
                "id": rid,
                "document": doc,
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "distance": dist,
            })
        return hits
