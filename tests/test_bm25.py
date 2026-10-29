import pytest

from retrievalbench.bm25 import BM25, tokenize


def test_bm25_score_matches_hand_computed_robertson_positive_idf():
    # N=2, df(cat)=1, dl=avgdl=2, tf=2: ln(2)*(2*2.5)/(2+1.5).
    index = BM25({"b": "cat cat", "a": "dog dog"})
    result = index.search("CAT", 2)
    assert result[0][0] == "b"
    assert result[0][1] == pytest.approx(0.990210257942779)
    assert result[1] == ("a", 0.0)


def test_ties_and_query_term_repetitions_are_deterministic():
    index = BM25({"z": "cat", "a": "cat", "m": "dog"})
    assert [doc for doc, _ in index.search("cat cat", 3)] == ["a", "z", "m"]
    assert index.search("cat cat", 3) == index.search("cat", 3)
    assert index.search("", 10) == [("a", 0.0), ("m", 0.0), ("z", 0.0)]


def test_unicode_tokens_and_empty_document():
    assert tokenize("Café's PYTHON_3 — 你好!") == ["café", "s", "python_3", "你好"]
    assert BM25({"empty": "", "x": "cat"}).search("cat", 1)[0][0] == "x"
    assert BM25({"b": "", "a": ""}).search("cat", 2) == [("a", 0.0), ("b", 0.0)]


@pytest.mark.parametrize(
    "kwargs", [{"k1": 0}, {"k1": float("nan")}, {"b": -0.1}, {"b": 1.1}, {"b": float("inf")}]
)
def test_invalid_bm25_parameters(kwargs):
    with pytest.raises(ValueError):
        BM25({"a": "cat"}, **kwargs)


def test_empty_corpus_and_invalid_limit_rejected():
    with pytest.raises(ValueError):
        BM25({})
    with pytest.raises(ValueError):
        BM25({"a": "cat"}).search("cat", 0)


def test_very_large_finite_k1_does_not_overflow_scores():
    result = BM25({"a": "cat " * 10, "b": "dog " * 10}, k1=1e308).search("cat", 2)
    # As k1 grows with normalized length 1, the saturation factor approaches tf=10.
    assert result[0] == pytest.approx(("a", 6.931471805599453))
