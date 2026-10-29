"""Command-line tool to build and maintain the knowledge base.

Run from the `backend` folder:

    python -m knowledge.build all                  # download, import, index (everything)
    python -m knowledge.build download             # fetch raw datasets into data/kb/raw
    python -m knowledge.build import --sources medlineplus,medquad
    python -m knowledge.build index                # (re)build the vector index
    python -m knowledge.build dedupe               # remove duplicate documents (no re-embedding)
    python -m knowledge.build stats
    python -m knowledge.build review --document 42 --status rejected
    python -m knowledge.build review --dataset bioasq --status pending
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

from app.config import BACKEND_DIR
from knowledge import sources
from knowledge.chunking import chunk_blocks, strip_boilerplate
from knowledge.db import REVIEW_STATUSES, connect, insert_record, remove_duplicates
from knowledge.index import DEFAULT_EMBEDDING_MODEL, build_index, prune_index

logger = logging.getLogger("knowledge.build")

DEFAULT_KB_DIR = BACKEND_DIR / "data" / "kb"
DB_FILE = "medquery_kb.sqlite3"
DATASETS = ("medlineplus", "medquad", "bioasq", "pubmed")


def paths(kb_dir: Path) -> tuple[Path, Path, Path]:
    return kb_dir / DB_FILE, kb_dir / "raw", kb_dir / "index"


def cmd_download(kb_dir: Path, datasets: list[str], pubmed_per_topic: int) -> None:
    _, raw, _ = paths(kb_dir)
    if "medlineplus" in datasets:
        sources.download_medlineplus(raw)
    if "medquad" in datasets:
        sources.download(sources.MEDQUAD_ZIP, raw / "medquad.zip")
    if "bioasq" in datasets:
        sources.download(sources.BIOASQ_PASSAGES, raw / "bioasq_passages.parquet")
        sources.download(sources.BIOASQ_QUESTIONS, raw / "bioasq_test.parquet")
    if "pubmed" in datasets:
        sources.download_pubmed(raw, per_topic=pubmed_per_topic)


def cmd_import(kb_dir: Path, datasets: list[str], status: str) -> None:
    db, raw, _ = paths(kb_dir)
    readers = {
        "medlineplus": lambda: sources.read_medlineplus(raw / "medlineplus.zip"),
        "medquad": lambda: sources.read_medquad(raw / "medquad.zip"),
        "bioasq": lambda: sources.read_bioasq(raw / "bioasq_passages.parquet"),
        "pubmed": lambda: sources.read_pubmed(raw / "pubmed"),
    }
    conn = connect(db)
    for name in datasets:
        added = skipped = rejected = 0
        for record in readers[name]():
            try:
                chunks = chunk_blocks(strip_boilerplate(record.blocks))
                if insert_record(conn, record, chunks, status):
                    added += 1
                else:
                    skipped += 1
            except ValueError as exc:
                rejected += 1
                logger.debug("Rejected record: %s", exc)
            if added and added % 2000 == 0:
                conn.commit()
                logger.info("%s: %d documents imported...", name, added)
        conn.commit()
        logger.info("%s: %d imported, %d already present or duplicate, %d rejected (missing fields)",
                    name, added, skipped, rejected)
    conn.close()


def cmd_dedupe(kb_dir: Path) -> None:
    db, _, index_dir = paths(kb_dir)
    backup = db.with_suffix(".sqlite3.bak")
    shutil.copy2(db, backup)
    logger.info("Backed up the database to %s", backup.name)
    conn = connect(db)
    removed = remove_duplicates(conn, DATASETS)
    live = np.array([r[0] for r in conn.execute("SELECT chunk_id FROM document_chunks")], dtype="int64")
    conn.close()
    logger.info("Removed %d duplicate documents", removed)
    if (index_dir / "meta.json").exists():
        pruned = prune_index(index_dir, lambda ids: np.isin(ids, live))
        logger.info("Removed %d vectors of deleted chunks from the index", pruned)


def cmd_stats(kb_dir: Path) -> None:
    db, _, index_dir = paths(kb_dir)
    conn = connect(db, readonly=True)
    print(f"Database: {db}")
    for row in conn.execute(
        """SELECT s.dataset, d.review_status, COUNT(DISTINCT d.document_id) AS docs, COUNT(c.chunk_id) AS chunks
           FROM medical_documents d JOIN medical_sources s ON s.source_id = d.source_id
           LEFT JOIN document_chunks c ON c.document_id = d.document_id
           GROUP BY s.dataset, d.review_status ORDER BY s.dataset"""
    ):
        print(f"  {row['dataset']:<12} {row['review_status']:<9} {row['docs']:>7} documents {row['chunks']:>8} chunks")
    for table in ("medical_sources", "medical_topics", "medical_claims"):
        print(f"  {table:<22} {conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>7} rows")
    conn.close()
    meta = index_dir / "meta.json"
    print(f"Index: {meta.read_text() if meta.exists() else 'not built'}")


def cmd_review(kb_dir: Path, status: str, document: int | None, dataset: str | None) -> None:
    db, _, _ = paths(kb_dir)
    conn = connect(db)
    if document is not None:
        n = conn.execute("UPDATE medical_documents SET review_status = ?, last_verified = date('now') "
                         "WHERE document_id = ?", (status, document)).rowcount
    else:
        conn.execute("UPDATE medical_sources SET review_status = ?, verified_date = date('now') WHERE dataset = ?",
                     (status, dataset))
        n = conn.execute(
            "UPDATE medical_documents SET review_status = ?, last_verified = date('now') "
            "WHERE source_id IN (SELECT source_id FROM medical_sources WHERE dataset = ?)", (status, dataset)).rowcount
    conn.commit()
    conn.close()
    print(f"{n} document(s) set to {status}.")
    if status == "approved":
        print("Run `python -m knowledge.build index` so newly approved documents become searchable.")
    else:
        print("The running server stops using them immediately (approval is re-checked on every query).")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m knowledge.build", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["all", "download", "import", "dedupe", "index", "stats", "review"])
    p.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    p.add_argument("--sources", default=",".join(DATASETS), help=f"comma-separated subset of {', '.join(DATASETS)}")
    p.add_argument("--status", choices=REVIEW_STATUSES, default="approved",
                   help="import: review status for new documents (default approved = trusted at source level); "
                        "review: the status to set")
    p.add_argument("--pubmed-per-topic", type=int, default=50)
    p.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="sentence-transformers embedding model")
    p.add_argument("--document", type=int, help="review: a document_id")
    p.add_argument("--dataset", choices=DATASETS, help="review: every document from one dataset")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    datasets = [s.strip() for s in args.sources.split(",") if s.strip()]
    if unknown := set(datasets) - set(DATASETS):
        p.error(f"unknown sources: {', '.join(sorted(unknown))}")

    db, _, index_dir = paths(args.kb_dir)
    if args.command in ("all", "download"):
        cmd_download(args.kb_dir, datasets, args.pubmed_per_topic)
    if args.command in ("all", "import"):
        cmd_import(args.kb_dir, datasets, args.status)
    if args.command == "dedupe":
        cmd_dedupe(args.kb_dir)
    if args.command in ("all", "index"):
        n = build_index(db, index_dir, args.model)
        logger.info("Index built: %d vectors in %s", n, index_dir)
    if args.command in ("all", "stats", "import", "dedupe", "index"):
        cmd_stats(args.kb_dir)
    if args.command == "review":
        if (args.document is None) == (args.dataset is None):
            p.error("review needs exactly one of --document or --dataset")
        cmd_review(args.kb_dir, args.status, args.document, args.dataset)


if __name__ == "__main__":
    sys.exit(main())
