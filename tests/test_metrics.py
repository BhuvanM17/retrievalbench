import math

import pytest

from retrievalbench.metrics import evaluate


def test_multi_relevant_metrics_use_full_relevance_denominator_and_ideal_dcg():
    # q1: hits at 2 and 3 of 3 relevant; q2: only hit at 1 of 1 relevant.
    # At 3 q1 DCG = 1/log2(3) + 1/2; ideal = 1 + 1/log2(3) + 1/2.
    report = evaluate(
        {"q1": {"a", "b", "c"}, "q2": {"z"}},
        {"q1": ["x", "b", "a"], "q2": ["z", "y"]},
        [1, 3],
    )
    q1 = report["per_query"]["q1"]["3"]
    assert q1["recall"] == pytest.approx(2 / 3)
    assert q1["mrr"] == 0.5
    assert q1["ndcg"] == pytest.approx(0.5307212739772434)
    assert report["aggregate"]["1"] == {"recall": 0.5, "mrr": 0.5, "ndcg": 0.5}
    assert report["aggregate"]["3"]["recall"] == pytest.approx(5 / 6)
    assert report["aggregate"]["3"]["mrr"] == 0.75
    assert report["query_count"] == 2


def test_missing_query_is_zero_and_short_run_uses_ideal_relevance():
    result = evaluate({"a": {"d1", "d2"}, "b": {"d3"}}, {"a": ["d1"]}, [5])
    assert result["per_query"]["b"]["5"] == {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    assert result["per_query"]["a"]["5"]["ndcg"] == pytest.approx(0.6131471927654584)
    assert result["aggregate"]["5"]["recall"] == 0.25


@pytest.mark.parametrize("ks", [[], [0], [-1], [True], [1.5], [1, 1]])
def test_invalid_cutoffs_are_rejected(ks):
    with pytest.raises(ValueError):
        evaluate({"q": {"d"}}, {}, ks)


@pytest.mark.parametrize("qrels", [{}, {"q": set()}])
def test_empty_judgments_rejected(qrels):
    with pytest.raises(ValueError):
        evaluate(qrels, {}, [1])


@pytest.mark.parametrize("run", [{"unknown": ["d"]}, {"q": ["d", "d"]}])
def test_invalid_run_rejected(run):
    with pytest.raises(ValueError):
        evaluate({"q": {"d"}}, run, [1])


def test_perfect_run_and_no_hits_are_bounded():
    result = evaluate({"q": {"a", "b"}}, {"q": ["a", "b"]}, [1, 2, 10])
    assert result["per_query"]["q"]["1"]["recall"] == 0.5
    assert result["per_query"]["q"]["2"] == {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0}
    assert all(math.isfinite(x) for x in result["aggregate"]["10"].values())
