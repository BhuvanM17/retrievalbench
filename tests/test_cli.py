import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(*arguments):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "retrievalbench", *map(str, arguments)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=ROOT,
        check=False,
    )


@pytest.fixture
def inputs(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        '{"doc_id":"a","text":"cat cat"}\n{"doc_id":"b","text":"dog dog"}\n',
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query_id":"q","text":"cat","relevant_doc_ids":["a"]}\n'
        '{"query_id":"r","text":"dog","relevant_doc_ids":["b"]}\n',
        encoding="utf-8",
    )
    return docs, queries


def test_baseline_roundtrip_and_comparison_with_missing_queries(inputs, tmp_path):
    docs, queries = inputs
    output = tmp_path / "baseline"
    baseline = cli(
        "baseline", "--documents", docs, "--queries", queries, "--k", 1, 2, "--out", output
    )
    assert baseline.returncode == 0, baseline.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["schema_version"] == 1
    assert report["runs"]["bm25"]["aggregate"]["1"] == {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0}
    assert report["runs"]["bm25"]["retriever"]["type"] == "bm25"
    assert len(report["inputs"]["queries_sha256"]) == 64
    with (output / "per_query.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {row["query_id"] for row in rows} == {"q", "r"}
    other = tmp_path / "external.csv"
    other.write_text("query_id,doc_id,rank\nq,b,1\nq,a,2\n", encoding="utf-8")
    comparison = tmp_path / "comparison"
    evaluated = cli(
        "evaluate",
        "--queries",
        queries,
        "--run",
        f"lexical={output / 'run.csv'}",
        "--run",
        f"external={other}",
        "--k",
        1,
        2,
        "--out",
        comparison,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    result = json.loads((comparison / "report.json").read_text())
    assert result["runs"]["external"]["aggregate"]["1"]["recall"] == 0.0
    assert result["runs"]["external"]["aggregate"]["2"]["recall"] == 0.5
    assert result["runs"]["external"]["aggregate"]["2"]["mrr"] == 0.25
    assert result["runs"]["lexical"]["aggregate"] == report["runs"]["bm25"]["aggregate"]
    assert "external" in evaluated.stdout


def test_outputs_are_byte_deterministic(inputs, tmp_path):
    docs, queries = inputs
    for name in ("first", "second"):
        result = cli(
            "baseline", "--documents", docs, "--queries", queries, "--out", tmp_path / name
        )
        assert result.returncode == 0, result.stderr
    for filename in ("report.json", "per_query.csv", "run.csv"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


@pytest.mark.parametrize(
    "options",
    [
        ["--k", "0"],
        ["--k", "2", "2"],
        ["--limit", "0"],
        ["--k1", "nan"],
        ["--b", "1.2"],
    ],
)
def test_baseline_rejects_invalid_options_without_creating_report(inputs, tmp_path, options):
    docs, queries = inputs
    output = tmp_path / "out"
    result = cli("baseline", "--documents", docs, "--queries", queries, "--out", output, *options)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_baseline_rejects_relevance_ids_missing_from_corpus(inputs, tmp_path):
    docs, queries = inputs
    queries.write_text('{"query_id":"q","text":"x","relevant_doc_ids":["unknown"]}\n')
    result = cli("baseline", "--documents", docs, "--queries", queries, "--out", tmp_path / "out")
    assert result.returncode == 2
    assert "unknown" in result.stderr


def test_evaluate_rejects_duplicate_names_and_malformed_csv(inputs, tmp_path):
    _, queries = inputs
    run = tmp_path / "bad.csv"
    run.write_text("query_id,doc_id,rank\nq,a,1\nq,b,1\n")
    result = cli("evaluate", "--queries", queries, "--run", f"x={run}", "--out", tmp_path / "out")
    assert result.returncode == 2
    assert "duplicate rank" in result.stderr
    run.write_text("query_id,doc_id,rank\nq,a,1\n")
    result = cli(
        "evaluate",
        "--queries",
        queries,
        "--run",
        f"x={run}",
        "--run",
        f"x={run}",
        "--out",
        tmp_path / "out",
    )
    assert result.returncode == 2
    assert "duplicate run name" in result.stderr


def test_missing_file_is_friendly_and_help_available(inputs, tmp_path):
    _, queries = inputs
    result = cli(
        "evaluate", "--queries", queries, "--run", "x=missing.csv", "--out", tmp_path / "out"
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert cli("--help").returncode == 0
