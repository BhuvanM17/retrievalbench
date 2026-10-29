"""Binary relevance metrics; each query has equal weight in macro averages."""

import math
from collections.abc import Mapping, Sequence, Set


def validate_cutoffs(cutoffs: Sequence[int]) -> list[int]:
    """Return sorted, distinct positive integer cutoffs or raise ValueError."""
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs):
        raise ValueError("cutoffs must be positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be distinct")
    return sorted(cutoffs)


def evaluate(
    qrels: Mapping[str, Set[str]], run: Mapping[str, Sequence[str]], cutoffs: Sequence[int]
) -> dict:
    """Evaluate all judged queries, including missing runs as zero-scoring queries.

    Unjudged documents count as nonrelevant. Queries without any relevant document
    are rejected because recall and normalized DCG would be undefined.
    """
    ks = validate_cutoffs(cutoffs)
    if not qrels or any(not relevant for relevant in qrels.values()):
        raise ValueError("every query must have at least one relevant document")
    if set(run) - set(qrels):
        raise ValueError("run contains unknown query IDs")
    if any(len(docs) != len(set(docs)) for docs in run.values()):
        raise ValueError("run contains duplicate document IDs within a query")
    per_query = {}
    for query_id in sorted(qrels):
        relevant = qrels[query_id]
        ranked = run.get(query_id, [])
        by_cutoff = {}
        for k in ks:
            hit_ranks = [rank for rank, doc in enumerate(ranked[:k], 1) if doc in relevant]
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            by_cutoff[str(k)] = {
                "recall": len(hit_ranks) / len(relevant),
                "mrr": 1 / hit_ranks[0] if hit_ranks else 0.0,
                "ndcg": sum(1 / math.log2(rank + 1) for rank in hit_ranks) / ideal,
            }
        per_query[query_id] = by_cutoff
    aggregate = {
        str(k): {
            metric: sum(row[str(k)][metric] for row in per_query.values()) / len(qrels)
            for metric in ("recall", "mrr", "ndcg")
        }
        for k in ks
    }
    return {"query_count": len(qrels), "aggregate": aggregate, "per_query": per_query}
