"""Command line entry points for baseline retrieval and run comparison."""

import argparse
import csv
import hashlib
from pathlib import Path

from retrievalbench import __version__
from retrievalbench.bm25 import BM25
from retrievalbench.io import load_documents, load_queries, load_run, write_reports, write_run
from retrievalbench.metrics import evaluate, validate_cutoffs


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ranked retrieval offline, or generate a lexical BM25 baseline."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline", help="retrieve with in-memory BM25 and evaluate")
    comparison = commands.add_parser("evaluate", help="evaluate one or more exported CSV runs")
    for command in (baseline, comparison):
        command.add_argument("--queries", required=True, type=Path, help="query/judgment JSONL")
        command.add_argument("--k", nargs="+", type=int, default=[1, 3, 5], help="distinct cutoffs")
        command.add_argument("--out", required=True, type=Path, help="report output directory")
    baseline.add_argument("--documents", required=True, type=Path, help="document JSONL")
    baseline.add_argument("--limit", type=int, help="retrieval depth (default: largest cutoff)")
    baseline.add_argument("--k1", type=float, default=1.5, help="BM25 term saturation")
    baseline.add_argument("--b", type=float, default=0.75, help="BM25 length normalization")
    comparison.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="named run; repeat to compare retrievers on the same queries",
    )
    return parser


def _build_report(args: argparse.Namespace) -> tuple[dict, dict | None]:
    cutoffs = validate_cutoffs(args.k)
    queries, qrels = load_queries(args.queries)
    report = {
        "schema_version": 1,
        "tool_version": __version__,
        "evaluation": {
            "cutoffs": cutoffs,
            "relevance": "binary",
            "aggregation": "macro_average_over_all_queries",
            "missing_query_policy": "zero",
            "unjudged_document_policy": "nonrelevant",
            "zero_relevance_policy": "reject",
        },
        "inputs": {"queries_sha256": _sha256(args.queries)},
        "runs": {},
    }
    results = None
    if args.command == "baseline":
        limit = args.limit if args.limit is not None else max(cutoffs)
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        documents = load_documents(args.documents)
        missing = set().union(*qrels.values()) - documents.keys()
        if missing:
            raise ValueError(f"relevant IDs missing from corpus: {', '.join(sorted(missing))}")
        index = BM25(documents, k1=args.k1, b=args.b)
        results = {query_id: index.search(text, limit) for query_id, text in queries.items()}
        run = {query_id: [doc_id for doc_id, _ in ranked] for query_id, ranked in results.items()}
        result = evaluate(qrels, run, cutoffs)
        result["retriever"] = {
            "type": "bm25",
            "k1": args.k1,
            "b": args.b,
            "limit": limit,
            "tokenizer": "unicode_word_casefold",
            "query_term_frequency": "binary",
            "document_count": len(documents),
            "documents_sha256": _sha256(args.documents),
        }
        report["runs"]["bm25"] = result
    else:
        for specification in args.run:
            name, separator, filename = specification.partition("=")
            if not separator or not name.strip() or name != name.strip() or not filename:
                raise ValueError("--run must be NAME=CSV with a nonblank name and path")
            if name in report["runs"]:
                raise ValueError(f"duplicate run name: {name}")
            path = Path(filename)
            result = evaluate(qrels, load_run(path, set(queries)), cutoffs)
            result["source_sha256"] = _sha256(path)
            report["runs"][name] = result
    return report, results


def main(argv: list[str] | None = None) -> int:
    """Return zero on success; argparse reports input and filesystem errors as exit 2."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report, results = _build_report(args)
        write_reports(args.out, report)
        if results is not None:
            write_run(args.out / "run.csv", results)
    except (ValueError, OSError, UnicodeError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"{'run':<20} {'k':>5} {'Recall@K':>10} {'MRR@K':>10} {'nDCG@K':>10}")
    for name, run in sorted(report["runs"].items()):
        for cutoff in report["evaluation"]["cutoffs"]:
            metrics = run["aggregate"][str(cutoff)]
            print(
                f"{name:<20} {cutoff:>5} {metrics['recall']:>10.4f} "
                f"{metrics['mrr']:>10.4f} {metrics['ndcg']:>10.4f}"
            )
    print(f"Reports: {args.out / 'report.json'} and {args.out / 'per_query.csv'}")
    return 0
