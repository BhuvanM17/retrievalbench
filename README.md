# RetrievalBench

[![CI](https://github.com/BhuvanM17/retrievalbench/actions/workflows/ci.yml/badge.svg)](https://github.com/BhuvanM17/retrievalbench/actions/workflows/ci.yml)

**Measure what your RAG retriever actually finds.** RetrievalBench evaluates ranked document results with Recall@K, MRR@K and binary nDCG@K. Compare exports from a vector database, hybrid search service or your own retriever on the same relevance judgments.

The included **BM25 lexical baseline** gives you a reproducible starting point. It uses word matching, with no embeddings, vector index or LLM. Evaluation and retrieval run locally with **Python 3.12+ and zero runtime dependencies**; no API key, model download or network connection is needed after installation.

This is a portfolio project by BhuvanM, created with AI assistance. No employer deployment is claimed.

## Quick start

Clone the repository and run these commands from its root. You can use the virtual environment's Python directly, so activation and PowerShell execution-policy changes are unnecessary.

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m retrievalbench baseline --documents examples/documents.jsonl --queries examples/queries.jsonl --k 1 3 5 --out reports/bm25
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m retrievalbench baseline --documents examples/documents.jsonl --queries examples/queries.jsonl --k 1 3 5 --out reports/bm25
```

The installed `retrievalbench` command and `python -m retrievalbench` are equivalent. Use `--help`, `baseline --help`, or `evaluate --help` to inspect options.

The example contains **16 synthetic documents and 8 author-written queries** about software systems. Running the command above produces this measured result:

| K | Recall@K | MRR@K | nDCG@K |
|---:|---:|---:|---:|
| 1 | 0.6250 | 0.8750 | 0.8750 |
| 3 | 1.0000 | 0.9375 | 0.9617 |
| 5 | 1.0000 | 0.9375 | 0.9617 |

These are toy-data results, not evidence of production retrieval quality or improvements over another system. For example, `q05` retrieves a BM25 description first, while its judgments identify the evaluation documents at ranks 2 and 3. Its Recall@1 is zero and MRR@3 is 0.5. This is visible in the checked-in [per-query CSV](examples/baseline/per_query.csv).

The actual generated artifacts are checked into [examples/baseline](examples/baseline): [report.json](examples/baseline/report.json), [per_query.csv](examples/baseline/per_query.csv) and [run.csv](examples/baseline/run.csv). Regenerate them with the quick-start command using `--out examples/baseline`.

## Evaluate your retriever

Export each query's ordered document IDs to a CSV file:

```csv
query_id,doc_id,rank,score
q01,d03,1,0.92
q01,d01,2,0.84
q02,d04,1,0.88
```

Then compare any number of named runs. Run names must be unique; quote the entire `NAME=PATH` argument when a path contains spaces. Replace `your-export.csv` with your actual export:

```bash
python -m retrievalbench evaluate --queries examples/queries.jsonl --run bm25=examples/baseline/run.csv --run candidate=your-export.csv --k 1 3 5 --out reports/comparison
```

Only ranks determine metrics. Scores are optional and ignored for ordering; scores from different systems need not be on a common scale. The evaluator does not call your retriever, generate embeddings, or require document text. Use identical document IDs and the same query judgments across runs.

For an executable comparison using only files in this repository, generate a second BM25 configuration and compare both exports:

```bash
python -m retrievalbench baseline --documents examples/documents.jsonl --queries examples/queries.jsonl --b 0 --out reports/no-length-normalization
python -m retrievalbench evaluate --queries examples/queries.jsonl --run default=examples/baseline/run.csv --run b0=reports/no-length-normalization/run.csv --out reports/comparison
```

This changes a baseline parameter; it does not imply a quality improvement. The report preserves both runs' metrics rather than declaring a winner.

## Input contracts

All input is UTF-8; a leading UTF-8 BOM is accepted. IDs are case-sensitive nonblank strings without surrounding whitespace. JSONL blank lines are skipped. Fields must match the schemas below exactly so misspelled fields fail visibly.

### Documents: `documents.jsonl`

One object per document:

```json
{"doc_id":"d01","text":"Retry transient HTTP errors with exponential backoff."}
```

`doc_id` must be unique across the corpus; `text` must be a string. Empty text is allowed. Baseline retrieval requires a nonempty corpus and verifies every relevant ID exists in it.

### Queries and judgments: `queries.jsonl`

One object per query, combining query text with binary relevance judgments:

```json
{"query_id":"q01","text":"retry network errors","relevant_doc_ids":["d01","d03"]}
```

`query_id` must be unique. `text` must be a string; empty text is allowed. `relevant_doc_ids` must be a nonempty list of unique IDs. All listed documents are equally relevant. Unlisted documents count as nonrelevant, including IDs absent from the corpus when evaluating external runs.

Queries with **zero relevant documents are rejected**, because recall and normalized DCG have no defined denominator. Empty query files are also rejected. Remove such queries or create judgments before evaluating; the tool never silently drops them. Duplicate JSON object fields, document IDs, query IDs and repeated relevance IDs are errors.

### Ranked runs: `run.csv`

| Column | Required | Contract |
|---|---|---|
| `query_id` | Yes | Must exist in the query file |
| `doc_id` | Yes | Unique within that query's run |
| `rank` | Yes | Positive integer; contiguous from 1 within each query |
| `score` | No | Finite numeric value when the column is present |

Column order and row order may vary. Extra or duplicate columns, unknown query IDs, duplicate ranks, duplicate document IDs within a query, and gaps in ranks are rejected. A document may appear for different queries. Resolve score ties into explicit ranks before exporting. The evaluator never silently deduplicates or reranks your results.

A query omitted from a run receives **zero for all metrics** and stays in the aggregate denominator. A header-only CSV is a valid empty run. Short runs are evaluated as supplied; missing results do not shrink the ideal ranking or relevant-document count. Cutoffs must be distinct positive integers.

## Metric definitions

For query `q`, let `R` be all its judged relevant IDs and `r_i` be 1 when the result at rank `i` is relevant, otherwise 0. Missing positions have `r_i = 0`.

| Metric | Definition | What it measures |
|---|---|---|
| Recall@K | `sum(r_i, i=1..K) / len(R)` | Coverage of all relevant documents |
| MRR@K | `1 / first_relevant_rank` if the first relevant rank is at most K, else 0 | How soon a useful result appears |
| Binary nDCG@K | `DCG@K / IDCG@K`, where `DCG = sum(r_i / log2(i+1))` and `IDCG = sum(1 / log2(i+1), i=1..min(K,len(R)))` | Relevant results near the top |

All three lie between 0 and 1; higher is better. Reports contain per-query values and a **macro average over every query**, giving each query equal weight regardless of its number of relevant documents. The per-query `mrr` value is reciprocal rank; the aggregate is its mean.

A hand-checkable fixture: if three documents are relevant and the run's first three positions are `[irrelevant, relevant, relevant]`, Recall@3 = `2/3`, MRR@3 = `1/2`, and binary nDCG@3 ≈ `0.530721274`. The tests independently check this case, perfect results, no hits, multiple relevant IDs and truncated runs.

## Reports and reproducibility

Every successful command writes:

- `report.json`: schema version, tool version, evaluation policies, input SHA-256 hashes, aggregate metrics and per-query metrics for each run. Cutoff keys are strings (`"1"`, `"3"`) in JSON objects.
- `per_query.csv`: one row per `(run, query_id, k)`, with `recall`, `mrr` and `ndcg` columns.
- `run.csv`: baseline command only; the generated ranks and full-precision BM25 scores, reusable by `evaluate`.

Baseline reports also record `k1`, `b`, retrieval depth, tokenizer policy and corpus hash. Imported runs record their source CSV hash. Reports omit timestamps and machine-specific paths. Sorted query IDs, sorted query terms and document-ID tie-breaking keep results deterministic for identical input bytes and configuration. Hashes identify source bytes, so changing line endings changes the hash even if the parsed content is equivalent.

`--out` creates a directory and replaces these output filenames if they already exist. Use a fresh directory for each experiment you want to retain. Inputs are validated before report creation; filesystem failures return an error and may leave partial output. Exit code is 0 on success and 2 for invalid input or filesystem errors.

## BM25 baseline details

Tokenization applies Unicode `casefold()` followed by Python's Unicode `\w+` matching. Digits and underscores remain part of tokens; apostrophes split them. There is no stemming, stopword removal, phrase matching or language-specific segmentation. Each distinct query term contributes once.

For each matching query term, the score contribution is:

```text
IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
score = IDF * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_length))
```

Defaults are `k1=1.5`, `b=0.75`. `k1` must be finite and positive; `b` must be finite and between 0 and 1. Use `--limit` to set retrieval depth; the default is the largest K. The baseline returns up to `min(limit, corpus_size)` documents, including documents with zero lexical score. Equal scores sort by ascending document ID. An empty or unmatched query therefore returns document IDs in sorted order. An all-empty corpus is supported and produces zero scores.

## Architecture

The CLI composes four small modules:

| Module | Responsibility |
|---|---|
| [`cli.py`](src/retrievalbench/cli.py) | Arguments, provenance, named-run comparison and terminal summary |
| [`io.py`](src/retrievalbench/io.py) | Strict JSONL/CSV validation and report serialization |
| [`bm25.py`](src/retrievalbench/bm25.py) | In-memory inverted postings and deterministic lexical ranking |
| [`metrics.py`](src/retrievalbench/metrics.py) | Pure evaluation functions with no filesystem or retrieval dependency |

The metric evaluator can also be used from Python:

```python
from retrievalbench.metrics import evaluate

result = evaluate(
    qrels={"q1": {"d1", "d2"}},
    run={"q1": ["d3", "d2"]},
    cutoffs=[1, 2],
)
print(result["aggregate"]["2"])  # recall=0.5, mrr=0.5, ndcg≈0.38685
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Development tools are pinned in `pyproject.toml`; runtime dependencies are empty. Tests cover independently computed metrics and BM25 scores, strict input rejection, stable tie ordering, empty and missing runs, CLI subprocess round trips and deterministic artifacts. GitHub Actions runs tests and lint on Python 3.12 and 3.13, plus an installed-command smoke test.

## Scope and limitations

This measures the retrieval stage of RAG. It does not measure answer correctness, faithfulness, generation quality, latency, statistical significance or cost. Binary judgments cannot express graded relevance, and incomplete judgments treat potentially useful unjudged documents as nonrelevant. Compare systems on the same queries, document IDs and judgment policy; use a held-out, representative evaluation set for meaningful decisions.

The bundled synthetic data is an illustrative fixture with subjective judgments. It is deliberately small and shares vocabulary with its queries. Its scores do not generalize to real corpora.

The baseline keeps documents, postings and scores in memory and sorts the whole corpus per query (`O(D log D)` sorting, plus matching postings). Evaluation loads complete runs into memory. This is an inspectable baseline and an offline export evaluator, not a production search service or an approximate nearest-neighbor engine. Exports from embedding and hybrid retrievers work through the same CSV contract.

## License

[MIT](LICENSE), copyright © 2026 BhuvanM.
