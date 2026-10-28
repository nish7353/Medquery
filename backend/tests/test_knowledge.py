"""Knowledge-base build + retrieval, using a tiny fake encoder (no model downloads).

All text here is neutral TEST-ONLY placeholder content, not medical information.
"""

import hashlib

import numpy as np
import pytest

from knowledge.build import DB_FILE
from knowledge.chunking import chunk_blocks, html_to_blocks, paragraphs, strip_boilerplate
from knowledge.db import Block, Record, connect, insert_record
from knowledge.index import build_index
from retrieval.adapters.medquery_kb import MedQueryKnowledgeBase
from retrieval.base import RetrievalQuery, RetrievalStatus


def fake_encode(texts):
    """Bag-of-words hashed into 64 dims: texts sharing words get similar vectors."""
    out = np.zeros((len(texts), 64), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def record(ext, title, text, url=None, dataset="testset"):
    return Record(external_id=ext, title=title, blocks=paragraphs(text), source_name=title,
                  source_url=url or f"https://example.org/{ext}", publisher="Test", dataset=dataset, license="test")


def test_html_to_blocks_keeps_headings_and_lists():
    blocks = html_to_blocks("<h3>Alpha?</h3><p>One &amp; two.</p><ul><li>beta</li><li>gamma</li></ul>")
    assert blocks[0] == Block("heading", "Alpha?")
    # A list stays attached to the paragraph that introduces it.
    assert blocks[1] == Block("para", "One & two.\n- beta\n- gamma")


def test_chunks_stay_within_sections_and_overlap():
    words = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_blocks([Block("heading", "Part A"), Block("para", words), Block("heading", "Part B"),
                           Block("para", "Short text.")], target_words=60, overlap_words=10)
    a = [c for s, c in chunks if s == "Part A"]
    assert len(a) > 1 and all(c.startswith("Part A\n") for c in a)
    assert all(len(c.split()) <= 75 for c in a)
    last_sentence_of_first = a[0].rsplit("Sentence", 1)[1]
    assert last_sentence_of_first in a[1], "adjacent chunks should overlap"
    assert chunks[-1] == ("Part B", "Part B\nShort text.")


def test_boilerplate_removed_but_repeated_facts_kept():
    blocks = paragraphs("Much of this information comes from Orphanet, a European rare disease database. "
                        "This condition is inherited in an autosomal dominant pattern, which means one copy is enough.")
    text = strip_boilerplate(blocks)[0].text
    assert "Orphanet" not in text and "autosomal dominant" in text


def test_insert_rejects_missing_fields_and_skips_duplicates(tmp_path):
    conn = connect(tmp_path / DB_FILE)
    r = record("t:1", "Alpha", "Alpha text.")
    assert insert_record(conn, r, chunk_blocks(r.blocks), "approved") is True
    assert insert_record(conn, r, chunk_blocks(r.blocks), "approved") is False
    with pytest.raises(ValueError):
        insert_record(conn, record("t:2", "Beta", "Beta.", url=" "), [(None, "Beta.")], "approved")
    with pytest.raises(ValueError):
        insert_record(conn, record("t:3", "Gamma", "Gamma."), [(None, "Gamma.")], "bogus")


@pytest.fixture
def kb(tmp_path):
    conn = connect(tmp_path / DB_FILE)
    docs = [
        record("t:alpha", "Alpha placeholder", "The alpha widget glows when the lever is pulled."),
        record("t:beta", "Beta placeholder", "The beta gadget hums quietly during the night."),
        record("t:pending", "Pending placeholder", "The alpha widget glows brightly in pending notes."),
    ]
    for r in docs:
        insert_record(conn, r, chunk_blocks(r.blocks), "pending" if r.external_id == "t:pending" else "approved")
    conn.commit()
    conn.close()
    assert build_index(tmp_path / DB_FILE, tmp_path / "index", "fake-model", encode=fake_encode) == 2
    return tmp_path


def query(text):
    return RetrievalQuery(original_text=text, normalized_text=text, top_k=3)


def test_retriever_returns_approved_evidence_with_sources(kb):
    result = MedQueryKnowledgeBase(str(kb), encoder=fake_encode).retrieve(query("when does the alpha widget glow"))
    assert result.status == RetrievalStatus.OK
    top = result.passages[0]
    assert top.source_id == "t:alpha"
    assert top.source_url == "https://example.org/t:alpha"
    assert 0 <= top.relevance_score <= 1
    assert all(p.source_id != "t:pending" for p in result.passages), "pending documents are never indexed"


def test_rejected_document_disappears_without_reindexing(kb):
    conn = connect(kb / DB_FILE)
    conn.execute("UPDATE medical_documents SET review_status = 'rejected' WHERE external_id = 't:alpha'")
    conn.commit()
    conn.close()
    result = MedQueryKnowledgeBase(str(kb), encoder=fake_encode).retrieve(query("alpha widget glow"))
    assert all(p.source_id != "t:alpha" for p in result.passages)


def test_missing_knowledge_base_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        MedQueryKnowledgeBase(str(tmp_path))
