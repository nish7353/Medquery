"""Retriever for MedQuery's own knowledge base (built with `python -m knowledge.build all`).

    MEDQUERY_KB_ADAPTER=retrieval.adapters.medquery_kb:MedQueryKnowledgeBase
    MEDQUERY_KB_CONNECTION=            # optional: knowledge-base folder (default backend/data/kb)

Search is semantic: the question is embedded with the same sentence-transformers
model that built the index, the vector index returns the nearest chunks, and each chunk is
re-checked in SQLite so only currently approved documents and sources are used.
"""

from pathlib import Path

from app.lazy_model import LazyModel
from knowledge.build import DB_FILE, DEFAULT_KB_DIR
from knowledge.db import APPROVED_CHUNKS_SQL, connect
from knowledge.index import VectorIndex, sentence_transformer_encoder
from retrieval.base import Passage, RetrievalQuery, RetrievalResult, RetrievalStatus, Retriever

# Vector hits that turn out to be unapproved are dropped, so ask for extra candidates.
OVERFETCH = 4


class MedQueryKnowledgeBase(Retriever):
    name = "medquery_kb"

    def __init__(self, connection: str = "", **options) -> None:
        super().__init__(connection, **options)
        kb_dir = Path(connection) if connection.strip() else DEFAULT_KB_DIR
        self.db_path = kb_dir / DB_FILE
        if not self.db_path.exists():
            raise FileNotFoundError(f"{self.db_path} not found. Run `python -m knowledge.build all` first.")
        self.index = VectorIndex.load(kb_dir / "index")
        self.meta = self.index.meta
        self.encoder = options.get("encoder") or LazyModel(
            "embedding", self.meta["embedding_model"], lambda: sentence_transformer_encoder(self.meta["embedding_model"])
        )
        if isinstance(self.encoder, LazyModel):
            self.encoder.warm_up_in_background()

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        encode = self.encoder.get() if isinstance(self.encoder, LazyModel) else self.encoder
        if encode is None:
            raise RuntimeError(f"Embedding model unavailable: {self.encoder.error}")

        vector = encode([query.normalized_text or query.original_text])[0]
        hits = self.index.search(vector, query.top_k * OVERFETCH)
        if not hits:
            return RetrievalResult(status=RetrievalStatus.NO_RESULTS, adapter=self.name)

        conn = connect(self.db_path, readonly=True)
        try:
            marks = ",".join("?" * len(hits))
            rows = {r["chunk_id"]: r for r in conn.execute(f"{APPROVED_CHUNKS_SQL} AND c.chunk_id IN ({marks})",
                                                           [cid for cid, _ in hits])}
        finally:
            conn.close()

        passages, seen = [], set()
        for cid, score in hits:
            row = rows.get(cid)
            # MedQuAD repeats some NIH pages verbatim; show each passage text once.
            if row is None or row["chunk_text"] in seen:
                continue
            seen.add(row["chunk_text"])
            passages.append(Passage(
                text=row["chunk_text"],
                source_id=row["external_id"],
                source_title=row["title"] if row["dataset"] != "bioasq" else row["source_name"],
                source_url=row["source_url"],
                # Cosine similarity, clipped to 0..1 so MEDQUERY_KB_MIN_RELEVANCE applies directly.
                relevance_score=round(max(0.0, min(1.0, score)), 4),
                metadata={
                    "document_id": row["document_id"], "chunk_id": cid, "dataset": row["dataset"],
                    "publisher": row["publisher"], "topic": row["topic_name"],
                    "content_type": row["content_type"], "section": row["section"],
                },
            ))
            if len(passages) == query.top_k:
                break
        status = RetrievalStatus.OK if passages else RetrievalStatus.NO_RESULTS
        return RetrievalResult(status=status, passages=passages, adapter=self.name)
