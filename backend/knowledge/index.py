"""Vector index over approved chunks.

Exact cosine-similarity search with FAISS: every chunk vector is L2-normalised,
so inner product in a FAISS `IndexFlatIP` is the cosine score. If `faiss-cpu`
is not installed, the same computation runs as a numpy matrix-vector product,
which returns identical results. At this size (~80k chunks x 384 dims) a query
takes a few milliseconds either way.

`chunk_ids.npy` maps each vector row to a `document_chunks.chunk_id`;
`meta.json` records which embedding model built the index so queries are
always embedded with the same model.
"""

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from knowledge.db import APPROVED_CHUNKS_SQL, connect

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORS_FILE, IDS_FILE, META_FILE = "vectors.npy", "chunk_ids.npy", "meta.json"

Encoder = Callable[[list[str]], np.ndarray]


def sentence_transformer_encoder(model_name: str, batch_size: int = 64) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype("float32")

    return encode


def embedding_text(row) -> str:
    # Titles of BioASQ rows are just "PubMed abstract <id>", which adds nothing.
    return row["chunk_text"] if row["dataset"] == "bioasq" else f"{row['title']}. {row['chunk_text']}"


def _faiss_index(vectors: np.ndarray):
    try:
        import faiss
    except ImportError:
        logger.info("faiss-cpu not installed; using numpy exact search")
        return None
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype="float32"))
    return index


def _swap_in(tmp: Path, index_dir: Path) -> None:
    """Move freshly written index files over the old ones.

    Files are replaced one at a time (metadata last) rather than renaming the folder,
    because Windows refuses to replace a folder that another process, such as OneDrive
    or a running server, has open.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    for name in (VECTORS_FILE, IDS_FILE, META_FILE):
        os.replace(tmp / name, index_dir / name)
    shutil.rmtree(tmp, ignore_errors=True)


class VectorIndex:
    def __init__(self, vectors: np.ndarray, chunk_ids: np.ndarray, meta: dict, use_faiss: bool = True):
        if len(vectors) != len(chunk_ids):
            raise RuntimeError("Vectors and chunk-id map are out of sync; rebuild the index.")
        self.vectors, self.chunk_ids, self.meta = vectors, chunk_ids, meta
        self.faiss = _faiss_index(vectors) if use_faiss and len(vectors) else None

    @property
    def backend(self) -> str:
        return "faiss" if self.faiss is not None else "numpy"

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_id, cosine score) pairs, best first."""
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        k = min(k, len(self.chunk_ids))
        if k <= 0:
            return []
        if self.faiss is not None:
            scores, rows = self.faiss.search(q, k)
            return [(int(self.chunk_ids[i]), float(s)) for i, s in zip(rows[0], scores[0]) if i >= 0]
        scores = self.vectors @ q[0]
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in top]

    @classmethod
    def load(cls, index_dir: Path, use_faiss: bool = True) -> "VectorIndex":
        meta = json.loads((index_dir / META_FILE).read_text())
        return cls(np.load(index_dir / VECTORS_FILE), np.load(index_dir / IDS_FILE), meta, use_faiss)


def build_index(db_path: Path, index_dir: Path, model_name: str = DEFAULT_EMBEDDING_MODEL,
                encode: Encoder | None = None, batch: int = 512) -> int:
    """Embed every approved chunk and write a fresh index. Returns the vector count."""
    encode = encode or sentence_transformer_encoder(model_name)
    conn = connect(db_path, readonly=True)
    rows = conn.execute(APPROVED_CHUNKS_SQL + " ORDER BY c.chunk_id").fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("No approved chunks to index. Import data first.")

    parts = []
    started = time.time()
    for i in range(0, len(rows), batch):
        parts.append(encode([embedding_text(r) for r in rows[i:i + batch]]))
        done = min(i + batch, len(rows))
        rate = done / max(time.time() - started, 1e-6)
        logger.info("Embedded %d/%d chunks (%.0f/s, ~%.0fs left)", done, len(rows), rate, (len(rows) - done) / rate)
    vectors = np.vstack(parts).astype("float32")

    # Write to a temporary folder, then swap, so a running server never sees a half-written index.
    tmp = index_dir.with_name(index_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / VECTORS_FILE, vectors)
    np.save(tmp / IDS_FILE, np.array([r["chunk_id"] for r in rows], dtype="int64"))
    (tmp / META_FILE).write_text(json.dumps({
        "embedding_model": model_name,
        "dimension": int(vectors.shape[1]),
        "vectors": int(vectors.shape[0]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    _swap_in(tmp, index_dir)
    return int(vectors.shape[0])


def prune_index(index_dir: Path, keep: Callable[[np.ndarray], np.ndarray]) -> int:
    """Drop index rows whose chunk ids fail `keep` (e.g. deleted chunks) without re-embedding."""
    index = VectorIndex.load(index_dir, use_faiss=False)
    mask = keep(index.chunk_ids)
    removed = int((~mask).sum())
    if removed:
        tmp = index_dir.with_name(index_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        np.save(tmp / VECTORS_FILE, index.vectors[mask])
        np.save(tmp / IDS_FILE, index.chunk_ids[mask])
        (tmp / META_FILE).write_text(json.dumps({**index.meta, "vectors": int(mask.sum())}, indent=2))
        _swap_in(tmp, index_dir)
    return removed
