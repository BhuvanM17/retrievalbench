"""Strict UTF-8 input readers and deterministic report writers."""

import csv
import json
import math
from pathlib import Path


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonblank string without surrounding whitespace")
    return value


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _jsonl(path: Path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_fields)
                if not isinstance(row, dict):
                    raise ValueError("record must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
            yield line_no, row


def _load_records(path: Path, kind: str) -> dict:
    id_field = "doc_id" if kind == "document" else "query_id"
    fields = {id_field, "text"}
    if kind == "query":
        fields.add("relevant_doc_ids")
    records = {}
    for line_no, row in _jsonl(path):
        try:
            if set(row) != fields:
                raise ValueError(f"expected fields: {', '.join(sorted(fields))}")
            record_id = _identifier(row[id_field], id_field)
            if record_id in records:
                raise ValueError(f"duplicate {id_field}: {record_id}")
            if not isinstance(row["text"], str):
                raise ValueError("text must be a string")
            if kind == "query":
                relevant = row["relevant_doc_ids"]
                if not isinstance(relevant, list) or not relevant:
                    raise ValueError("relevant_doc_ids must be a nonempty list")
                for doc in relevant:
                    _identifier(doc, "relevant document ID")
                if len(set(relevant)) != len(relevant):
                    raise ValueError("duplicate relevant document ID")
            records[record_id] = row
        except ValueError as exc:
            raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no {kind} records")
    return records


def load_documents(path: Path) -> dict[str, str]:
    """Read unique document IDs and text; blank text is allowed."""
    return {key: row["text"] for key, row in _load_records(path, "document").items()}


def load_queries(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Read queries and binary judgments, rejecting zero-relevance queries."""
    rows = _load_records(path, "query")
    return (
        {key: row["text"] for key, row in rows.items()},
        {key: set(row["relevant_doc_ids"]) for key, row in rows.items()},
    )


def load_run(path: Path, query_ids: set[str]) -> dict[str, list[str]]:
    """Read CSV ranks, requiring contiguous 1-based ranks per query."""
    ranked = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if set(headers) not in (
            {"query_id", "doc_id", "rank"},
            {"query_id", "doc_id", "rank", "score"},
        ):
            raise ValueError(f"{path}: expected query_id,doc_id,rank and optional score columns")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate CSV headers")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong number of CSV fields")
                query_id = _identifier(row["query_id"], "query_id")
                doc_id = _identifier(row["doc_id"], "doc_id")
                if query_id not in query_ids:
                    raise ValueError(f"unknown query ID: {query_id}")
                rank = int(row["rank"])
                if rank <= 0:
                    raise ValueError("rank must be a positive integer")
                if "score" in row and not math.isfinite(float(row["score"])):
                    raise ValueError("score must be finite")
                entries = ranked.setdefault(query_id, {})
                if rank in entries:
                    raise ValueError(f"duplicate rank for query: {query_id}")
                if doc_id in entries.values():
                    raise ValueError(f"duplicate document for query: {query_id}")
                entries[rank] = doc_id
            except ValueError as exc:
                raise ValueError(f"{path}: line {reader.line_num}: {exc}") from exc
    runs = {}
    for query_id, entries in ranked.items():
        if sorted(entries) != list(range(1, len(entries) + 1)):
            raise ValueError(f"{path}: ranks for {query_id} must be contiguous starting at 1")
        runs[query_id] = [entries[rank] for rank in sorted(entries)]
    return runs


def write_run(path: Path, results: dict[str, list[tuple[str, float]]]) -> None:
    """Export full-precision BM25 scores and authoritative integer ranks."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_id", "doc_id", "rank", "score"])
        for query_id in sorted(results):
            for rank, (doc_id, score) in enumerate(results[query_id], 1):
                writer.writerow([query_id, doc_id, rank, repr(score)])


def write_reports(directory: Path, report: dict) -> None:
    """Write machine-readable JSON and one CSV row per run/query/cutoff."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (directory / "per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "query_id", "k", "recall", "mrr", "ndcg"])
        for name, result in sorted(report["runs"].items()):
            for query_id, cutoffs in result["per_query"].items():
                for k, metrics in cutoffs.items():
                    writer.writerow(
                        [name, query_id, k, *(metrics[m] for m in ("recall", "mrr", "ndcg"))]
                    )
