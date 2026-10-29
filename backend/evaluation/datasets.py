"""Evaluation datasets and their fixed dev/test splits.

No model is trained in MedQuery, so there is no training split. Each question set
is split once, deterministically, into:

  * dev  - used to choose models and thresholds;
  * test - held out, used only to report final numbers.

A question's split is decided by a hash of its id, so it never changes between runs
or machines and needs no split files.

| Set        | Used for                         | Gold labels                         |
|------------|----------------------------------|-------------------------------------|
| BioASQ     | retrieval, end-to-end answers    | relevant PubMed ids, ideal answer   |
| COVID-QA   | answer extraction (EM / F1)      | exact answer span in a paper        |
| Scope      | answering vs. refusing           | in-domain or off-topic, hand-labelled |
"""

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import BACKEND_DIR

EVAL_DIR = BACKEND_DIR / "data" / "eval"
KB_RAW_DIR = BACKEND_DIR / "data" / "kb" / "raw"
SCOPE_FILE = Path(__file__).parent / "data" / "scope_questions.json"
COVID_QA_URL = "https://huggingface.co/datasets/deepset/covid_qa_deepset/resolve/main/covid_qa_deepset/train-00000-of-00001.parquet"
SPLITS = ("dev", "test")


def split_of(dataset: str, item_id: str) -> str:
    digest = hashlib.sha1(f"{dataset}:{item_id}".encode()).digest()
    return "dev" if digest[0] % 2 == 0 else "test"


@dataclass
class EvalItem:
    id: str
    question: str
    answers: list[str] = field(default_factory=list)
    context: str = ""                 # gold passage (reader evaluation)
    gold_ids: set[str] = field(default_factory=set)  # gold source ids (retrieval)
    in_scope: bool = True


def _take(items: list[EvalItem], split: str, dataset: str, limit: int | None) -> list[EvalItem]:
    chosen = [it for it in items if split == "all" or split_of(dataset, it.id) == split]
    return chosen[:limit] if limit else chosen


def load_bioasq(split: str, limit: int | None = None, path: Path | None = None) -> list[EvalItem]:
    import pyarrow.parquet as pq

    path = path or KB_RAW_DIR / "bioasq_test.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m knowledge.build download --sources bioasq`.")
    rows = pq.read_table(path).to_pylist()
    items = [
        EvalItem(
            id=str(r["id"]),
            question=r["question"],
            answers=[r["answer"]] if r.get("answer") else [],
            gold_ids={f"pubmed:{pmid}" for pmid in ast.literal_eval(r["relevant_passage_ids"])},
        )
        for r in rows
    ]
    return _take(items, split, "bioasq", limit)


def _gold_paragraph(context: str, start: int) -> tuple[str, int]:
    """The paragraph of `context` that contains character `start`, and the offset inside it."""
    begin = context.rfind("\n", 0, start) + 1
    end = context.find("\n", start)
    return context[begin:end if end != -1 else len(context)], start - begin


def load_covid_qa(split: str, limit: int | None = None, path: Path | None = None) -> list[EvalItem]:
    import pyarrow.parquet as pq

    path = path or EVAL_DIR / "covid_qa.parquet"
    if not path.exists():
        from knowledge.sources import download

        download(COVID_QA_URL, path)
    items = []
    for r in pq.read_table(path).to_pylist():
        texts, starts = r["answers"]["text"], r["answers"]["answer_start"]
        if r["is_impossible"] or not texts:
            continue
        paragraph, offset = _gold_paragraph(r["context"], starts[0])
        # Answers that run across a paragraph break cannot be extracted from one paragraph.
        if paragraph[offset:offset + len(texts[0])] != texts[0]:
            continue
        items.append(EvalItem(id=str(r["id"]), question=r["question"], answers=texts, context=paragraph))
    return _take(items, split, "covid_qa", limit)


def load_scope() -> list[EvalItem]:
    data = json.loads(SCOPE_FILE.read_text(encoding="utf-8"))
    return [EvalItem(id=f"{kind}-{i}", question=q, in_scope=kind == "in_scope")
            for kind in ("in_scope", "off_topic") for i, q in enumerate(data[kind])]
