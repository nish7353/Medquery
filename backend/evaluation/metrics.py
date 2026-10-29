"""Evaluation metrics.

Exact Match and F1 follow the official SQuAD evaluation script: answers are
lower-cased and stripped of punctuation, articles and extra spaces before they
are compared, and F1 is computed over the bag of remaining tokens. When a
question has several gold answers, the best-matching one counts.
"""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    text = "".join(ch for ch in str(text).lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(metric, prediction: str, golds: list[str]) -> float:
    return max((metric(prediction, g) for g in golds), default=0.0)


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """1 if any gold item is in the top k, else 0."""
    return float(any(r in gold for r in ranked[:k]))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Fraction of the gold items found in the top k."""
    return len(gold.intersection(ranked[:k])) / len(gold) if gold else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str], k: int) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in gold:
            return 1.0 / i
    return 0.0
