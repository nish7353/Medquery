"""Dataset readers, downloaders and the build CLI, using tiny fixture files.

Fixtures copy each dataset's real file format but contain neutral TEST-ONLY
placeholder text, not medical information. Nothing touches the network.
"""

import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from knowledge import build, sources
from knowledge.db import connect
from retrieval.adapters.medquery_kb import MedQueryKnowledgeBase
from retrieval.base import RetrievalQuery
from tests.test_knowledge import fake_encode

MEDLINEPLUS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<health-topics total="3">
<health-topic title="Alpha Topic" url="https://example.org/alpha.html" id="1" language="English">
<also-called>Alpha alias</also-called>
<full-summary>&lt;h3&gt;What is alpha?&lt;/h3&gt;&lt;p&gt;Alpha is a placeholder topic.&lt;/p&gt;
&lt;p class=""&gt;NIH: Placeholder Institute&lt;/p&gt;</full-summary>
<group url="https://example.org/g" id="9">Placeholder Group</group>
<primary-institute url="https://example.org">Placeholder Institute</primary-institute>
</health-topic>
<health-topic title="Alfa" url="https://example.org/es/alfa.html" id="2" language="Spanish">
<full-summary>&lt;p&gt;Texto de prueba.&lt;/p&gt;</full-summary>
</health-topic>
<health-topic title="Empty" url="https://example.org/empty.html" id="3" language="English">
<full-summary></full-summary>
</health-topic>
</health-topics>"""

MEDQUAD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="0000001" source="TESTSRC" url="https://example.org/beta">
<Focus>Beta Thing</Focus>
<FocusAnnotations><UMLS><SemanticGroup>Placeholders</SemanticGroup></UMLS></FocusAnnotations>
<QAPairs>
<QAPair pid="1"><Question qid="0000001-1" qtype="information">What is (are) Beta Thing ?</Question>
<Answer>Beta thing is a placeholder.

It has a second paragraph.</Answer></QAPair>
<QAPair pid="2"><Question qid="0000001-2" qtype="symptoms">What are the symptoms of Beta Thing ?</Question>
<Answer></Answer></QAPair>
</QAPairs>
</Document>"""

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
<Journal><Title>Journal of Placeholders</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Gamma <i>review</i>.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Gamma background text.</AbstractText>
<AbstractText Label="RESULTS">Gamma results text.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>112</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article>
</MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def raw(tmp_path):
    """A kb folder with one tiny raw file per dataset, as `download` would leave it."""
    raw = tmp_path / "raw"
    (raw / "pubmed").mkdir(parents=True)
    with zipfile.ZipFile(raw / "medlineplus.zip", "w") as z:
        z.writestr("mplus_topics_test.xml", MEDLINEPLUS_XML)
    with zipfile.ZipFile(raw / "medquad.zip", "w") as z:
        z.writestr("MedQuAD-master/5_TEST_QA/0000001.xml", MEDQUAD_XML)
        # Same question id in another sub-collection must not collide.
        z.writestr("MedQuAD-master/6_OTHER_QA/0000001.xml", MEDQUAD_XML.replace("example.org/beta", "example.org/beta2"))
    pq.write_table(pa.table({"passage": ["Delta placeholder abstract.", "nan", ""], "id": [201, 202, 203]}),
                   raw / "bioasq_passages.parquet")
    (raw / "pubmed" / "diabetes.xml").write_text(PUBMED_XML, encoding="utf-8")
    return tmp_path


def test_read_medlineplus_keeps_english_drops_attribution(raw):
    records = list(sources.read_medlineplus(raw / "raw" / "medlineplus.zip"))
    assert [r.external_id for r in records] == ["medlineplus:1"]
    r = records[0]
    assert r.source_url == "https://example.org/alpha.html"
    assert [b.kind for b in r.blocks] == ["heading", "para"]
    assert "NIH" not in r.content
    assert r.topic_category == "Placeholder Group"
    assert r.topic_description == "Also called: Alpha alias"
    assert "Placeholder Institute" in r.publisher


def test_read_medquad_skips_empty_answers_and_qualifies_ids(raw):
    records = list(sources.read_medquad(raw / "raw" / "medquad.zip"))
    assert sorted(r.external_id for r in records) == ["medquad:5:0000001-1", "medquad:6:0000001-1"]
    r = records[0]
    assert r.title == "What is (are) Beta Thing ?"
    assert r.content_type == "information"
    assert r.topic == "Beta Thing" and r.topic_category == "Placeholders"
    assert len(r.blocks) == 2


def test_read_bioasq_skips_missing_passages(raw):
    records = list(sources.read_bioasq(raw / "raw" / "bioasq_passages.parquet"))
    assert [r.external_id for r in records] == ["pubmed:201"]
    assert records[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/201/"


def test_read_pubmed_parses_labelled_abstracts_and_skips_empty(raw):
    records = list(sources.read_pubmed(raw / "raw" / "pubmed"))
    assert [r.external_id for r in records] == ["pubmed:111"]
    r = records[0]
    assert r.title == "Gamma review."
    assert r.topic == "Diabetes"
    assert r.publication_date == "2020"
    assert [b.text for b in r.blocks] == ["Background", "Gamma background text.", "Results", "Gamma results text."]


def test_download_caches_existing_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sources, "_get", lambda url, timeout=60: calls.append(url) or b"data")
    dest = tmp_path / "f.bin"
    sources.download("https://example.org/f", dest)
    sources.download("https://example.org/f", dest)
    assert dest.read_bytes() == b"data" and len(calls) == 1


def test_download_medlineplus_finds_current_file(tmp_path, monkeypatch):
    page = b'<a href="https://medlineplus.gov/xml/mplus_topics_compressed_2026-01-02.zip">zip</a>'
    fetched = []
    monkeypatch.setattr(sources, "_get", lambda url, timeout=60: fetched.append(url) or (page if url.endswith("xml.html") else b"zip"))
    sources.download_medlineplus(tmp_path)
    assert fetched[-1].endswith("mplus_topics_compressed_2026-01-02.zip")
    monkeypatch.setattr(sources, "_get", lambda url, timeout=60: b"no links here")
    with pytest.raises(RuntimeError):
        sources.download_medlineplus(tmp_path / "other")


def test_download_pubmed_searches_then_fetches_each_topic(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "PUBMED_TOPICS", {"Alpha topic": "Alpha", "Empty topic": "Empty"})
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)

    def fake_get(url, timeout=60):
        if "esearch" in url:
            ids = ["1", "2"] if "Alpha" in url else []
            return json.dumps({"esearchresult": {"idlist": ids}}).encode()
        assert "id=1%2C2" in url
        return b"<PubmedArticleSet/>"

    monkeypatch.setattr(sources, "_get", fake_get)
    out = sources.download_pubmed(tmp_path, per_topic=2)
    assert (out / "alpha_topic.xml").read_bytes() == b"<PubmedArticleSet/>"
    assert not (out / "empty_topic.xml").exists()


def test_build_cli_import_stats_review_index(raw, monkeypatch, capsys):
    kb = str(raw)
    real_build_index = build.build_index
    monkeypatch.setattr(build, "build_index", lambda db, out, model: real_build_index(db, out, model, encode=fake_encode))

    build.main(["import", "--kb-dir", kb])
    out = capsys.readouterr().out
    for dataset in ("medlineplus", "medquad", "bioasq", "pubmed"):
        assert dataset in out
    conn = connect(raw / build.DB_FILE, readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM medical_documents").fetchone()[0] == 5
    conn.close()

    build.main(["import", "--kb-dir", kb, "--sources", "bioasq"])  # re-import is a no-op
    build.main(["index", "--kb-dir", kb])
    assert (raw / "index" / "vectors.npy").exists()

    build.main(["review", "--kb-dir", kb, "--dataset", "bioasq", "--status", "rejected"])
    assert "1 document(s) set to rejected" in capsys.readouterr().out
    kb_retriever = MedQueryKnowledgeBase(kb, encoder=fake_encode)
    hits = kb_retriever.retrieve(RetrievalQuery(original_text="delta", normalized_text="delta placeholder abstract", top_k=5))
    assert all(p.metadata["dataset"] != "bioasq" for p in hits.passages)

    build.main(["review", "--kb-dir", kb, "--document", "1", "--status", "approved"])
    assert "knowledge.build index" in capsys.readouterr().out


def test_build_cli_rejects_bad_arguments(raw):
    with pytest.raises(SystemExit):
        build.main(["import", "--kb-dir", str(raw), "--sources", "mimic"])
    with pytest.raises(SystemExit):
        build.main(["review", "--kb-dir", str(raw)])


def test_retriever_shows_duplicate_passages_once(raw):
    build.main(["import", "--kb-dir", str(raw), "--sources", "medquad"])
    build.build_index(raw / build.DB_FILE, raw / "index", "fake", encode=fake_encode)
    hits = MedQueryKnowledgeBase(str(raw), encoder=fake_encode).retrieve(
        RetrievalQuery(original_text="beta", normalized_text="what is beta thing placeholder", top_k=5))
    texts = [p.text for p in hits.passages]
    assert len(texts) == len(set(texts)) and texts
