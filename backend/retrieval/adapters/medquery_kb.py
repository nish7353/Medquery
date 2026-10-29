"""Retriever for MedQuery's own knowledge base (built with `python -m knowledge.build all`).

    MEDQUERY_KB_ADAPTER=retrieval.adapters.medquery_kb:MedQueryKnowledgeBase
    MEDQUERY_KB_CONNECTION=            # optional: knowledge-base folder (default backend/data/kb)

Search is semantic: the question is embedded with the same BERT-family model that built
the index (normally the pipeline's own BERT query embedding is reused; the adapter only
encodes the question itself if the pipeline used a different model), FAISS returns the
nearest chunks, and each chunk is re-checked in SQLite so only currently approved documents
and sources are used.
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
        # Only needed when the pipeline's encoder differs from the index's model, so it is
        # loaded on first use rather than warmed up: a second background load would hold a
        # duplicate model in memory and race the query encoder's import of transformers.
        self.encoder = options.get("encoder") or LazyModel(
            "embedding", self.meta["embedding_model"], lambda: sentence_transformer_encoder(self.meta["embedding_model"])
        )

    def query_vector(self, query: RetrievalQuery) -> tuple[list[float], str]:
        """The pipeline's BERT embedding when it comes from the index's model, else our own encoding."""
        if (query.embedding and query.embedding_model == self.meta["embedding_model"]
                and len(query.embedding) == self.meta["dimension"]):
            return query.embedding, "bert_query_embedding"
        encode = self.encoder.get() if isinstance(self.encoder, LazyModel) else self.encoder
        if encode is None:
            raise RuntimeError(f"Embedding model unavailable: {self.encoder.error}")
        return encode([query.normalized_text or query.original_text])[0], "adapter_embedding"

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        vector, how = self.query_vector(query)
        message = f"Searched with {how.replace('_', ' ')} ({self.index.backend})"
        hits = self.index.search(vector, query.top_k * OVERFETCH)
        if not hits:
            return RetrievalResult(status=RetrievalStatus.NO_RESULTS, adapter=self.name, message=message)

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
        return RetrievalResult(status=status, passages=passages, adapter=self.name, message=message)
