"""Measure MedQuery: retrieval accuracy, answer Exact Match / F1, and answer-or-refuse behaviour.

Run from the `backend` folder (the knowledge base must be built and enabled in `.env`):

    python -m evaluation.evaluate retrieval  --split test            # BioASQ: Hit@k, Recall@k, MRR@10
    python -m evaluation.evaluate reader     --split test            # COVID-QA: EM and F1 on the gold paragraph
    python -m evaluation.evaluate end-to-end --split test --limit 300   # BioASQ through the whole pipeline
    python -m evaluation.evaluate scope                              # answer in-domain, refuse off-topic
    python -m evaluation.evaluate reader --split dev --model dmis-lab/biobert-base-cased-v1.1-squad

Choose models and thresholds on `--split dev`; report `--split test`. Every run writes
`evaluation/results/<task>-<split>[-<model>].json` (summary and per-question rows) and a
`...-failures.csv` listing the questions the system got wrong, for error analysis.
"""

import argparse
import csv
import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from answer_processing.generators import ExtractiveQAGenerator
from answer_processing.models import AnswerStatus
from app.config import Settings, get_settings
from app.container import build_services
from evaluation import datasets
from evaluation.metrics import best_over_golds, exact_match, f1_score, hit_at_k, recall_at_k, reciprocal_rank
from query_processing.understanding import QueryUnderstanding
from retrieval.base import Passage, RetrievalQuery

logger = logging.getLogger("evaluation")

RESULTS_DIR = Path(__file__).parent / "results"
ANSWERED = {AnswerStatus.ANSWERED, AnswerStatus.LOW_CONFIDENCE}


def _mean(values) -> float:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else 0.0


def _progress(i: int, total: int, started: float) -> None:
    if i % 50 == 0 or i == total:
        rate = i / max(time.time() - started, 1e-6)
        logger.info("%d/%d questions (%.1f/s, ~%.0fs left)", i, total, rate, (total - i) / rate)


def _settings(model: str | None) -> Settings:
    base = get_settings()
    return base.model_copy(update={"qa_model_name": model}) if model else base


def _query(understanding: QueryUnderstanding, question: str, top_k: int) -> RetrievalQuery:
    analysis = understanding.analyze(question)
    return RetrievalQuery(
        original_text=analysis.pre.clean_text,
        normalized_text=analysis.pre.normalized_text,
        keywords=analysis.pre.keywords,
        question_type=analysis.pre.question_type,
        embedding=analysis.encoded.embedding if analysis.encoded else None,
        embedding_model=understanding.encoder.model.model_name,
        top_k=top_k,
    )


def _documents(passages: list[Passage]) -> list[str]:
    """Source ids in rank order, each document once (a document can return several chunks)."""
    return list(dict.fromkeys(p.source_id for p in passages))


# ---------------------------------------------------------------- tasks
def eval_retrieval(split: str, limit: int | None, model: str | None) -> tuple[dict, list[dict]]:
    services = build_services(get_settings())
    if not services.retrieval.is_configured:
        raise SystemExit("No knowledge base is configured; set MEDQUERY_KB_ADAPTER in .env.")
    understanding = QueryUnderstanding(services.encoder)
    adapter = services.retrieval.retriever
    items = datasets.load_bioasq(split, limit)
    rows, started = [], time.time()
    for i, item in enumerate(items, 1):
        result = adapter.retrieve(_query(understanding, item.question, top_k=20))
        ranked = _documents(result.passages)
        rows.append({
            "id": item.id, "question": item.question, "gold": sorted(item.gold_ids), "retrieved": ranked[:10],
            "hit@1": hit_at_k(ranked, item.gold_ids, 1), "hit@5": hit_at_k(ranked, item.gold_ids, 5),
            "recall@5": recall_at_k(ranked, item.gold_ids, 5), "recall@10": recall_at_k(ranked, item.gold_ids, 10),
            "rr@10": reciprocal_rank(ranked, item.gold_ids, 10),
            "top_score": result.passages[0].relevance_score if result.passages else None,
            "search": result.message,
        })
        _progress(i, len(items), started)
    summary = {
        "questions": len(rows),
        "Hit@1": _mean(r["hit@1"] for r in rows),
        "Hit@5": _mean(r["hit@5"] for r in rows),
        "Recall@5": _mean(r["recall@5"] for r in rows),
        "Recall@10": _mean(r["recall@10"] for r in rows),
        "MRR@10": _mean(r["rr@10"] for r in rows),
        "embedding_model": adapter.meta["embedding_model"],
        "search": rows[0]["search"] if rows else None,
    }
    return summary, rows


def eval_reader(split: str, limit: int | None, model: str | None) -> tuple[dict, list[dict]]:
    settings = _settings(model)
    generator = ExtractiveQAGenerator(settings.qa_model_name, settings.qa_max_answer_tokens, settings.hf_offline)
    if not generator.available:
        raise SystemExit(f"QA model {settings.qa_model_name} could not be loaded: {generator.model.error}")
    items = datasets.load_covid_qa(split, limit)
    rows, started = [], time.time()
    for i, item in enumerate(items, 1):
        out = generator.generate(item.question, [Passage(text=item.context, source_id="gold")])
        prediction = out.text if out else ""
        rows.append({
            "id": item.id, "question": item.question, "gold": item.answers[0], "prediction": prediction,
            "confidence": round(out.confidence, 4) if out else None,
            "em": best_over_golds(exact_match, prediction, item.answers),
            "f1": round(best_over_golds(f1_score, prediction, item.answers), 4),
        })
        _progress(i, len(items), started)
    summary = {
        "questions": len(rows),
        "ExactMatch": _mean(r["em"] for r in rows),
        "F1": _mean(r["f1"] for r in rows),
        "no_answer_rate": _mean(float(not r["prediction"]) for r in rows),
        "qa_model": settings.qa_model_name,
    }
    return summary, rows


def eval_end_to_end(split: str, limit: int | None, model: str | None) -> tuple[dict, list[dict]]:
    settings = _settings(model)
    services = build_services(settings)
    items = datasets.load_bioasq(split, limit)
    rows, started = [], time.time()
    for i, item in enumerate(items, 1):
        r = services.pipeline.run(item.question)
        answered = r.status in ANSWERED
        source = r.evidence[r.answer.evidence_index].source_id if r.answer else None
        rows.append({
            "id": item.id, "question": item.question, "status": r.status.value,
            "prediction": r.answer.text if r.answer else "", "gold": item.answers[0] if item.answers else "",
            "answer_source": source, "gold_ids": sorted(item.gold_ids),
            "evidence_hit": float(bool(item.gold_ids.intersection(e.source_id for e in r.evidence))),
            "answer_from_gold": float(source in item.gold_ids) if answered else 0.0,
            "f1_vs_ideal": round(best_over_golds(f1_score, r.answer.text, item.answers), 4) if answered else 0.0,
        })
        _progress(i, len(items), started)
    answered_rows = [r for r in rows if r["status"] in {s.value for s in ANSWERED}]
    summary = {
        "questions": len(rows),
        "answered_rate": _mean(float(r["status"] in {s.value for s in ANSWERED}) for r in rows),
        "evidence_hit_rate": _mean(r["evidence_hit"] for r in rows),
        "answer_from_gold_passage": _mean(r["answer_from_gold"] for r in answered_rows),
        "F1_vs_ideal_answer": _mean(r["f1_vs_ideal"] for r in answered_rows),
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
        "qa_model": settings.qa_model_name,
        "min_relevance": settings.kb_min_relevance,
    }
    return summary, rows


def eval_scope(split: str, limit: int | None, model: str | None) -> tuple[dict, list[dict]]:
    settings = get_settings()
    services = build_services(settings)
    understanding = QueryUnderstanding(services.encoder)
    adapter = services.retrieval.retriever
    rows = []
    for item in datasets.load_scope():
        result = adapter.retrieve(_query(understanding, item.question, top_k=1))
        top = result.passages[0].relevance_score if result.passages else 0.0
        rows.append({"id": item.id, "question": item.question, "in_scope": item.in_scope, "top_score": top,
                     "answers": top >= settings.kb_min_relevance})
    rows_in = [r for r in rows if r["in_scope"]]
    rows_off = [r for r in rows if not r["in_scope"]]

    def balanced_accuracy(t: float) -> float:
        return (_mean(r["top_score"] >= t for r in rows_in) + _mean(r["top_score"] < t for r in rows_off)) / 2

    sweep = {f"{t / 100:.2f}": round(balanced_accuracy(t / 100), 4) for t in range(25, 71, 5)}
    best = max(sweep, key=lambda t: (sweep[t], -abs(float(t) - settings.kb_min_relevance)))
    summary = {
        "questions": len(rows),
        "min_relevance": settings.kb_min_relevance,
        "in_scope_answered": _mean(r["answers"] for r in rows_in),
        "off_topic_refused": _mean(not r["answers"] for r in rows_off),
        "in_scope_top_score": [min(r["top_score"] for r in rows_in), max(r["top_score"] for r in rows_in)],
        "off_topic_top_score": [min(r["top_score"] for r in rows_off), max(r["top_score"] for r in rows_off)],
        "threshold_sweep_balanced_accuracy": sweep,
        "best_threshold": float(best),
    }
    for r in rows:
        r["correct"] = r["answers"] == r["in_scope"]
    return summary, rows


TASKS = {"retrieval": eval_retrieval, "reader": eval_reader, "end-to-end": eval_end_to_end, "scope": eval_scope}
FAILED = {
    "retrieval": lambda r: r["hit@5"] == 0,
    "reader": lambda r: r["em"] == 0,
    "end-to-end": lambda r: r["answer_from_gold"] == 0,
    "scope": lambda r: not r["correct"],
}


def write_results(task: str, split: str, model: str | None, summary: dict, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    slug = f"-{re.sub(r'[^A-Za-z0-9.]+', '_', model.split('/')[-1])}" if model else ""
    stem = f"{task}-{split}{slug}" if task != "scope" else "scope"
    out = RESULTS_DIR / f"{stem}.json"
    out.write_text(json.dumps({
        "task": task, "split": split, "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary, "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    failures = [r for r in rows if FAILED[task](r)]
    with open(RESULTS_DIR / f"{stem}-failures.csv", "w", newline="", encoding="utf-8") as f:
        if failures:
            w = csv.DictWriter(f, fieldnames=list(failures[0]))
            w.writeheader()
            w.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in failures)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m evaluation.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=list(TASKS))
    p.add_argument("--split", choices=[*datasets.SPLITS, "all"], default="dev")
    p.add_argument("--limit", type=int, help="evaluate only the first N questions of the split")
    p.add_argument("--model", help="QA model to evaluate instead of MEDQUERY_QA_MODEL_NAME (reader, end-to-end)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("medquery").setLevel(logging.WARNING)
    logging.getLogger("answer_processing.pipeline").setLevel(logging.WARNING)
    summary, rows = TASKS[args.task](args.split, args.limit, args.model)
    out = write_results(args.task, args.split, args.model, summary, rows)
    print(json.dumps(summary, indent=2))
    print(f"Results: {out}")


if __name__ == "__main__":
    sys.exit(main())
