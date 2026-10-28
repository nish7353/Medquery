"""Downloaders and importers for the public datasets in the knowledge base.

Each `read_*` function yields `Record`s from a downloaded file and never
invents content: records without text or a source URL are skipped.

| Dataset      | What it contributes                                  | Licence           |
|--------------|------------------------------------------------------|-------------------|
| MedlinePlus  | NLM health-topic summaries (conditions, treatments)  | Public domain     |
| MedQuAD      | Q&A pairs from NIH sites (symptoms, treatment, ...)  | CC BY 4.0         |
| BioASQ (mini)| PubMed abstracts from the BioASQ task 11b corpus     | CC BY 2.5         |
| PubMed       | Review-article abstracts fetched via NCBI E-utilities| NLM terms of use  |
"""

import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

from knowledge.chunking import clean, html_to_blocks, paragraphs
from knowledge.db import Block, Record

logger = logging.getLogger(__name__)

USER_AGENT = "MedQuery-student-project/1.0 (knowledge-base builder)"
MEDLINEPLUS_INDEX = "https://medlineplus.gov/xml.html"
MEDQUAD_ZIP = "https://codeload.github.com/abachaa/MedQuAD/zip/refs/heads/master"
BIOASQ_PASSAGES = "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq/resolve/main/data/passages.parquet/part.0.parquet"
BIOASQ_QUESTIONS = "https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq/resolve/main/data/test.parquet/part.0.parquet"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Topics from the project plan, mapped to the MeSH heading used for the PubMed search.
PUBMED_TOPICS = {
    "Diabetes": "Diabetes Mellitus",
    "Hypertension": "Hypertension",
    "Asthma": "Asthma",
    "Dengue": "Dengue",
    "Malaria": "Malaria",
    "Tuberculosis": "Tuberculosis",
    "Anemia": "Anemia",
    "Migraine": "Migraine Disorders",
    "Kidney disease": "Renal Insufficiency, Chronic",
    "Heart disease": "Heart Diseases",
}


def _get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(_get(url, timeout=300))
    tmp.replace(dest)
    return dest


def download_medlineplus(raw_dir: Path) -> Path:
    page = _get(MEDLINEPLUS_INDEX).decode("utf-8", "replace")
    m = re.search(r'href="(https://medlineplus\.gov/xml/mplus_topics_compressed_[\d-]+\.zip)"', page)
    if not m:
        raise RuntimeError("Could not find the MedlinePlus health-topics XML link on xml.html")
    return download(m.group(1), raw_dir / "medlineplus.zip")


# ---------------------------------------------------------------- MedlinePlus
def read_medlineplus(path: Path) -> Iterator[Record]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    for t in root.iter("health-topic"):
        if t.get("language") != "English":
            continue
        summary = t.findtext("full-summary") or ""
        # Drop the trailing "NIH: <institute>" attribution paragraph.
        blocks = [b for b in html_to_blocks(summary) if not re.match(r"^(NIH|CDC|FDA)[:\s]", b.text)]
        if not blocks:
            continue
        title = t.get("title", "").strip()
        groups = [g.text for g in t.findall("group") if g.text]
        aliases = [a.text for a in t.findall("also-called") if a.text]
        institute = t.findtext("primary-institute")
        yield Record(
            external_id=f"medlineplus:{t.get('id')}",
            title=title,
            blocks=blocks,
            source_name=f"MedlinePlus: {title}",
            source_url=t.get("url", ""),
            publisher="U.S. National Library of Medicine (MedlinePlus)" + (f"; {institute}" if institute else ""),
            dataset="medlineplus",
            license="Public domain (U.S. Government work)",
            content_type="overview",
            topic=title,
            topic_category=groups[0] if groups else None,
            topic_description=("Also called: " + ", ".join(aliases)) if aliases else None,
        )


# ---------------------------------------------------------------- MedQuAD
def read_medquad(path: Path) -> Iterator[Record]:
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".xml")):
            doc = ET.fromstring(z.read(name))
            collection = name.split("/")[1].split("_")[0]
            url = (doc.get("url") or "").strip()
            focus = clean(doc.findtext("Focus") or "")
            group = doc.findtext(".//SemanticGroup")
            if not url:
                continue
            for pair in doc.iter("QAPair"):
                q = pair.find("Question")
                answer = pair.findtext("Answer") or ""
                if q is None or not answer.strip():
                    continue  # three MedQuAD subsets ship without answers (copyright)
                yield Record(
                    # Question ids restart in every sub-collection, so qualify them.
                    external_id=f"medquad:{collection}:{q.get('qid')}",
                    title=clean(q.text or ""),
                    blocks=paragraphs(answer),
                    source_name=f"{doc.get('source')}: {focus}" if focus else doc.get("source", "MedQuAD"),
                    source_url=url,
                    publisher=f"{doc.get('source')} (via MedQuAD)",
                    dataset="medquad",
                    license="CC BY 4.0 (MedQuAD, Ben Abacha & Demner-Fushman 2019)",
                    content_type=q.get("qtype"),
                    topic=focus or None,
                    topic_category=group,
                )


# ---------------------------------------------------------------- BioASQ
def read_bioasq(path: Path) -> Iterator[Record]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "passage"])
    for pmid, text in zip(table.column("id").to_pylist(), table.column("passage").to_pylist()):
        if not text or text.strip().lower() == "nan":
            continue
        yield Record(
            external_id=f"pubmed:{pmid}",
            title=f"PubMed abstract {pmid}",
            blocks=paragraphs(text),
            source_name=f"PubMed {pmid}",
            source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            publisher="PubMed (via BioASQ task 11b)",
            dataset="bioasq",
            license="CC BY 2.5 (rag-mini-bioasq)",
            content_type="research_abstract",
        )


# ---------------------------------------------------------------- PubMed
def _pubmed_query(mesh: str, since: int) -> str:
    return (f'"{mesh}"[MeSH Major Topic] AND hasabstract AND english[lang] '
            f'AND (review[pt] OR systematic review[pt]) AND ("{since}"[dp] : "3000"[dp])')


def download_pubmed(raw_dir: Path, per_topic: int = 50, since: int = 2015) -> Path:
    """Fetch review abstracts for each project topic and cache the raw XML."""
    out = raw_dir / "pubmed"
    out.mkdir(parents=True, exist_ok=True)
    for topic, mesh in PUBMED_TOPICS.items():
        dest = out / f"{re.sub(r'[^a-z0-9]+', '_', topic.lower())}.xml"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        params = {"db": "pubmed", "term": _pubmed_query(mesh, since), "retmax": per_topic,
                  "sort": "relevance", "retmode": "json", "tool": "medquery"}
        ids = json.loads(_get(f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(params)}"))["esearchresult"]["idlist"]
        time.sleep(0.4)  # NCBI allows 3 requests/second without an API key
        if not ids:
            logger.warning("PubMed returned no articles for %s", topic)
            continue
        fetch = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "medquery"}
        xml = _get(f"{EUTILS}/efetch.fcgi?{urllib.parse.urlencode(fetch)}", timeout=120)
        dest.write_bytes(xml)
        logger.info("PubMed: %d abstracts for %s", len(ids), topic)
        time.sleep(0.4)
    return out


def read_pubmed(folder: Path) -> Iterator[Record]:
    names = {re.sub(r"[^a-z0-9]+", "_", t.lower()): t for t in PUBMED_TOPICS}
    for path in sorted(folder.glob("*.xml")):
        topic = names.get(path.stem)
        for art in ET.parse(path).getroot().iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title_el = art.find(".//ArticleTitle")
            title = clean("".join(title_el.itertext())) if title_el is not None else ""
            blocks: list[Block] = []
            for ab in art.findall(".//Abstract/AbstractText"):
                text = clean("".join(ab.itertext()))
                if not text:
                    continue
                if ab.get("Label"):
                    blocks.append(Block("heading", ab.get("Label").capitalize()))
                blocks.append(Block("para", text))
            if not pmid or not title or not blocks:
                continue
            year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4] or None
            journal = art.findtext(".//Journal/Title")
            yield Record(
                external_id=f"pubmed:{pmid}",
                title=title,
                blocks=blocks,
                source_name=f"{title} ({journal}, {year})" if journal else title,
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                publisher=journal or "PubMed",
                dataset="pubmed",
                license="Abstract text © the publisher; used under NLM terms for research",
                content_type="review_abstract",
                topic=topic,
                publication_date=year,
            )
