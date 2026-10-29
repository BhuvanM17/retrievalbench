"""Small in-memory BM25 baseline. No embeddings, models, or network calls."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping


def tokenize(text: str) -> list[str]:
    """Unicode word tokens, casefolded; underscores and digits are preserved."""
    return re.findall(r"\w+", text.casefold())


class BM25:
    """BM25 with positive IDF and one contribution per distinct query term."""

    def __init__(self, documents: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("corpus must contain at least one document")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and greater than zero")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between zero and one")
        self.k1, self.b = k1, b
        self.lengths = {}
        self.postings = defaultdict(dict)
        for doc_id, text in documents.items():
            counts = Counter(tokenize(text))
            self.lengths[doc_id] = sum(counts.values())
            for term, frequency in counts.items():
                self.postings[term][doc_id] = frequency
        self.average_length = sum(self.lengths.values()) / len(documents)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Rank all docs, including zero scores; ties sort by document ID."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        scores = dict.fromkeys(self.lengths, 0.0)
        for term in sorted(set(tokenize(query))):
            postings = self.postings.get(term, {})
            idf = math.log(1 + (len(self.lengths) - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc_id, frequency in postings.items():
                normalization = 1 - self.b + self.b * self.lengths[doc_id] / self.average_length
                if self.k1 >= 1:
                    # Divide by k1 before multiplying: finite large k1 must not overflow.
                    saturation = (
                        frequency * (1 + 1 / self.k1) / (frequency / self.k1 + normalization)
                    )
                else:
                    saturation = frequency * (self.k1 + 1) / (frequency + self.k1 * normalization)
                scores[doc_id] += idf * saturation
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
