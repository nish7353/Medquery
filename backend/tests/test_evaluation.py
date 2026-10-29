"""Evaluation metrics and splits (no models or datasets needed)."""

from evaluation.datasets import _gold_paragraph, load_scope, split_of
from evaluation.metrics import exact_match, f1_score, hit_at_k, normalize_answer, recall_at_k, reciprocal_rank


def test_normalize_follows_squad_rules():
    assert normalize_answer("The  Quick, brown fox!") == "quick brown fox"


def test_exact_match_and_f1():
    assert exact_match("the alpha", "Alpha.") == 1.0
    assert exact_match("alpha beta", "alpha") == 0.0
    assert f1_score("alpha beta", "alpha") == 2 * (0.5 * 1.0) / 1.5
    assert f1_score("gamma", "alpha") == 0.0
    assert f1_score("", "") == 1.0


def test_ranking_metrics():
    ranked, gold = ["a", "b", "c", "d"], {"c", "z"}
    assert hit_at_k(ranked, gold, 2) == 0.0 and hit_at_k(ranked, gold, 3) == 1.0
    assert recall_at_k(ranked, gold, 4) == 0.5
    assert reciprocal_rank(ranked, gold, 10) == 1 / 3
    assert reciprocal_rank(ranked, gold, 2) == 0.0


def test_splits_are_deterministic_and_balanced():
    splits = [split_of("set", str(i)) for i in range(2000)]
    assert splits == [split_of("set", str(i)) for i in range(2000)]
    assert 0.45 < splits.count("dev") / len(splits) < 0.55


def test_gold_paragraph_contains_answer():
    context = "Title\n\nFirst para.\nSecond para has the ANSWER here.\nThird."
    start = context.index("ANSWER")
    para, offset = _gold_paragraph(context, start)
    assert para == "Second para has the ANSWER here." and para[offset:offset + 6] == "ANSWER"


def test_scope_set_is_balanced():
    items = load_scope()
    assert sum(i.in_scope for i in items) == sum(not i.in_scope for i in items) >= 20
