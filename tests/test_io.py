import json

import pytest

from retrievalbench.io import load_documents, load_queries, load_run


def jsonl(tmp_path, rows):
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_documents_and_queries_load(tmp_path):
    assert load_documents(jsonl(tmp_path, [{"doc_id": "a", "text": "Cat"}])) == {"a": "Cat"}
    queries, qrels = load_queries(
        jsonl(tmp_path, [{"query_id": "q", "text": "Cat", "relevant_doc_ids": ["a", "b"]}])
    )
    assert queries == {"q": "Cat"}
    assert qrels == {"q": {"a", "b"}}


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"doc_id": "a", "text": "x"}] * 2,
        [{"doc_id": "", "text": "x"}],
        [{"doc_id": "a", "text": 1}],
        [{"doc_id": "a", "text": "x", "typo": 1}],
        [None],
    ],
)
def test_document_schema_validation(tmp_path, rows):
    with pytest.raises(ValueError):
        load_documents(jsonl(tmp_path, rows))


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"query_id": "q", "text": "x", "relevant_doc_ids": []}],
        [{"query_id": "q", "text": "x", "relevant_doc_ids": ["a", "a"]}],
        [{"query_id": "q", "text": "x", "relevant_doc_ids": ["a"]}] * 2,
        [{"query_id": "q", "text": "x", "relevant_doc_ids": "a"}],
        [{"query_id": "q", "text": "x"}],
        [{"query_id": "q", "text": "x", "relevant_doc_ids": [None]}],
    ],
)
def test_query_schema_validation(tmp_path, rows):
    with pytest.raises(ValueError):
        load_queries(jsonl(tmp_path, rows))


def test_malformed_json_has_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"doc_id":"a","text":"ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_documents(path)


def test_run_sorts_explicit_ranks_ignoring_scores(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("query_id,doc_id,rank,score\nq,b,2,100\nq,a,1,-10\n", encoding="utf-8")
    assert load_run(path, {"q"}) == {"q": ["a", "b"]}


@pytest.mark.parametrize(
    "content",
    [
        "query_id,doc_id,rank\nq,a,1\nq,b,1\n",
        "query_id,doc_id,rank\nq,a,1\nq,a,2\n",
        "query_id,doc_id,rank\nq,a,2\n",
        "query_id,doc_id,rank\nq,a,0\n",
        "query_id,doc_id,rank\nq,a,1.5\n",
        "query_id,doc_id,rank\nother,a,1\n",
        "query_id,doc_id,rank,score\nq,a,1,nan\n",
        "query_id,doc_id,rank,typo\nq,a,1,2\n",
        "query_id,doc_id\nq,a\n",
        "query_id,doc_id,rank\nq,,1\n",
        "query_id,doc_id,rank\nq,a,1,extra\n",
    ],
)
def test_run_schema_validation(tmp_path, content):
    path = tmp_path / "run.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_run(path, {"q"})


def test_empty_run_with_header_is_valid(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("query_id,doc_id,rank\n", encoding="utf-8")
    assert load_run(path, {"q"}) == {}


def test_repeated_json_keys_are_rejected_instead_of_silently_overwriting(tmp_path):
    path = tmp_path / "ambiguous.jsonl"
    path.write_text('{"doc_id":"a","doc_id":"b","text":"Cat"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_documents(path)
