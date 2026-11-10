"""BM25, rank fusion and retrieval metrics. Numpy and PyStemmer only."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
import Stemmer

WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then "
    "there these they this to was will with".split()
)
_stemmer = Stemmer.Stemmer("english")


def tokenize(text: str) -> list[str]:
    return _stemmer.stemWords([w for w in WORD.findall(text.lower()) if w not in STOP])


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 0.9, b: float = 0.4):
        self.n = len(documents)
        lengths = np.array([len(d) for d in documents], dtype=np.float32)
        norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1))
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for doc_id, document in enumerate(documents):
            for term, tf in Counter(document).items():
                postings[term][0].append(doc_id)
                postings[term][1].append(tf)
        self.index = {}
        for term, (ids, tfs) in postings.items():
            ids_array, tf_array = np.array(ids), np.array(tfs, dtype=np.float32)
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            self.index[term] = (ids_array, idf * tf_array * (k1 + 1) / (tf_array + norm[ids_array]))

    def scores(self, query: list[str]) -> np.ndarray:
        totals = np.zeros(self.n, dtype=np.float32)
        for term in query:
            if term in self.index:
                ids, weights = self.index[term]
                totals[ids] += weights
        return totals


def top(scores: np.ndarray, depth: int) -> list[int]:
    depth = min(depth, scores.size)
    candidates = np.argpartition(-scores, depth - 1)[:depth]
    return candidates[np.argsort(-scores[candidates], kind="stable")].tolist()


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: sum 1/(k + rank) over each list. Scores never need comparing."""
    fused: Counter[int] = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            fused[doc] += 1 / (k + rank)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def convex(a: dict[int, float], b: dict[int, float], alpha: float = 0.5) -> list[int]:
    """Min-max normalise two score maps and take a weighted sum."""

    def normalise(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        low, high = min(scores.values()), max(scores.values())
        span = (high - low) or 1.0
        return {d: (s - low) / span for d, s in scores.items()}

    na, nb = normalise(a), normalise(b)
    combined = {d: alpha * na.get(d, 0) + (1 - alpha) * nb.get(d, 0) for d in set(na) | set(nb)}
    return [doc for doc, _ in sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))]


def ndcg(ranking: list[int], relevant: dict[int, int], k: int = 10) -> float:
    gain = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    best = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return gain / best if best else 0.0


def recall(ranking: list[int], relevant: dict[int, int], k: int = 100) -> float:
    wanted = {d for d, r in relevant.items() if r > 0}
    return len(wanted & set(ranking[:k])) / len(wanted) if wanted else 0.0


def reciprocal_rank(ranking: list[int], relevant: dict[int, int]) -> float:
    return next((1 / (i + 1) for i, d in enumerate(ranking) if relevant.get(d, 0) > 0), 0.0)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng, rounds: int = 10000):
    """95% interval for mean(b - a), resampling queries."""
    differences = b - a
    picks = rng.integers(0, differences.size, (rounds, differences.size))
    means = differences[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
