"""SQLite schema and write helpers for the knowledge base.

Same five tables as the project design (sources, topics, documents, chunks,
claims). SQLite is the source of truth: the vector index only stores chunk ids,
and every retrieved chunk is re-checked here for approval before it is used.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REVIEW_STATUSES = ("pending", "approved", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS medical_sources (
    source_id        INTEGER PRIMARY KEY,
    source_name      TEXT NOT NULL,
    publisher        TEXT,
    source_url       TEXT NOT NULL UNIQUE,
    dataset          TEXT NOT NULL,
    license          TEXT,
    publication_date TEXT,
    verified_date    TEXT,
    review_status    TEXT NOT NULL DEFAULT 'pending'
                     CHECK (review_status IN ('pending', 'approved', 'rejected'))
);

CREATE TABLE IF NOT EXISTS medical_topics (
    topic_id    INTEGER PRIMARY KEY,
    topic_name  TEXT NOT NULL UNIQUE COLLATE NOCASE,
    category    TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS medical_documents (
    document_id   INTEGER PRIMARY KEY,
    external_id   TEXT NOT NULL UNIQUE,
    topic_id      INTEGER REFERENCES medical_topics(topic_id),
    source_id     INTEGER NOT NULL REFERENCES medical_sources(source_id),
    title         TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_type  TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending'
                  CHECK (review_status IN ('pending', 'approved', 'rejected')),
    last_verified TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id     INTEGER PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES medical_documents(document_id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    section      TEXT,
    chunk_text   TEXT NOT NULL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, chunk_number)
);

CREATE TABLE IF NOT EXISTS medical_claims (
    claim_id      INTEGER PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES medical_documents(document_id) ON DELETE CASCADE,
    claim_text    TEXT NOT NULL,
    evidence_type TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending'
                  CHECK (review_status IN ('pending', 'approved', 'rejected')),
    reviewed_by   TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON medical_documents(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
"""


@dataclass
class Block:
    """A piece of document text: a section heading or a paragraph."""
    kind: str  # "heading" | "para"
    text: str


@dataclass
class Record:
    """One document as produced by an importer, before it is stored."""
    external_id: str
    title: str
    blocks: list[Block]
    source_name: str
    source_url: str
    publisher: str
    dataset: str
    license: str
    content_type: str | None = None
    topic: str | None = None
    topic_category: str | None = None
    topic_description: str | None = None
    publication_date: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def connect(path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _source_id(conn: sqlite3.Connection, r: Record, status: str, today: str) -> int:
    conn.execute(
        """INSERT OR IGNORE INTO medical_sources
               (source_name, publisher, source_url, dataset, license, publication_date, verified_date, review_status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (r.source_name, r.publisher, r.source_url, r.dataset, r.license, r.publication_date, today, status),
    )
    return conn.execute("SELECT source_id FROM medical_sources WHERE source_url = ?", (r.source_url,)).fetchone()[0]


def _topic_id(conn: sqlite3.Connection, r: Record) -> int | None:
    if not r.topic:
        return None
    conn.execute(
        "INSERT OR IGNORE INTO medical_topics (topic_name, category, description) VALUES (?, ?, ?)",
        (r.topic, r.topic_category, r.topic_description),
    )
    return conn.execute("SELECT topic_id FROM medical_topics WHERE topic_name = ?", (r.topic,)).fetchone()[0]


def insert_record(conn: sqlite3.Connection, r: Record, chunks: list[tuple[str | None, str]], status: str) -> bool:
    """Store a document with its chunks. Returns False if it was already imported."""
    if status not in REVIEW_STATUSES:
        raise ValueError(f"invalid review status {status!r}")
    if not r.title.strip() or not r.content.strip() or not r.source_url.strip():
        raise ValueError(f"{r.external_id}: title, content and source URL are required")
    if not chunks:
        raise ValueError(f"{r.external_id}: no searchable text left after cleaning")
    if conn.execute("SELECT 1 FROM medical_documents WHERE external_id = ?", (r.external_id,)).fetchone():
        return False

    today = date.today().isoformat()
    cur = conn.execute(
        """INSERT INTO medical_documents
               (external_id, topic_id, source_id, title, content, content_type, review_status, last_verified)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (r.external_id, _topic_id(conn, r), _source_id(conn, r, status, today), r.title, r.content,
         r.content_type, status, today),
    )
    conn.executemany(
        "INSERT INTO document_chunks (document_id, chunk_number, section, chunk_text) VALUES (?, ?, ?, ?)",
        [(cur.lastrowid, i, section, text) for i, (section, text) in enumerate(chunks)],
    )
    return True


APPROVED_CHUNKS_SQL = """
SELECT c.chunk_id, c.chunk_text, c.section, d.document_id, d.external_id, d.title, d.content_type,
       s.source_name, s.publisher, s.source_url, s.dataset, t.topic_name
FROM document_chunks c
JOIN medical_documents d ON d.document_id = c.document_id
JOIN medical_sources s   ON s.source_id = d.source_id
LEFT JOIN medical_topics t ON t.topic_id = d.topic_id
WHERE d.review_status = 'approved' AND s.review_status = 'approved'
"""
