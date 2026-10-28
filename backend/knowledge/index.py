"""Vector index over approved chunks.

Exact cosine-similarity search with numpy: every chunk vector is L2-normalised,
so a matrix-vector product gives the cosine score for all chunks at once. This
is the same computation as a FAISS `IndexFlatIP` and returns identical results.
FAISS is not used because its Windows wheels load an OpenMP runtime that
conflicts with PyTorch's and aborts the process when both are imported.
At this size (~80k chunks x 384 dims) a query takes a few milliseconds.

`chunk_ids.npy` maps each vector row to a `document_chunks.chunk_id`;
`meta.json` records which embedding model built the index so queries are
always embedded with the same model.
"""

import json
import logging
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


class VectorIndex:
    def __init__(self, vectors: np.ndarray, chunk_ids: np.ndarray, meta: dict):
        if len(vectors) != len(chunk_ids):
            raise RuntimeError("Vectors and chunk-id map are out of sync; rebuild the index.")
        self.vectors, self.chunk_ids, self.meta = vectors, chunk_ids, meta

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_id, cosine score) pairs, best first."""
        scores = self.vectors @ np.asarray(query_vector, dtype="float32").reshape(-1)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in top]

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        meta = json.loads((index_dir / META_FILE).read_text())
        return cls(np.load(index_dir / VECTORS_FILE), np.load(index_dir / IDS_FILE), meta)


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
    shutil.rmtree(index_dir, ignore_errors=True)
    tmp.rename(index_dir)
    return int(vectors.shape[0])
