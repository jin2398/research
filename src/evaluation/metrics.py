"""Offline metrics shared by search and recommendation evaluation.

Definitions (binary relevance; `relevant` is the set of items the user actually
interacted with in the test period):

- HitRate@k  = 1 if any of the top-k items is relevant, else 0 (averaged over users)
- Recall@k   = |top-k ∩ relevant| / |relevant|
- NDCG@k     = DCG@k / IDCG@k, DCG@k = Σ_{i=1..k} rel_i / log2(i + 1),
               IDCG@k = Σ_{i=1..min(k, |relevant|)} 1 / log2(i + 1)
- MRR@k      = 1 / rank of the first relevant item in the top-k (0 if none)
- Coverage   = |unique items recommended to all users| / |catalog|
- AUC        = P(score of a random positive > score of a random negative), ties = 0.5
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import Any


def _top(recommended: Sequence[Hashable], k: int) -> Sequence[Hashable]:
    if k < 1:
        raise ValueError("k must be positive")
    return recommended[:k]


def hit_rate_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    return float(any(item in relevant for item in _top(recommended, k)))


def recall_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(_top(recommended, k)) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float:
    if not relevant:
        return 0.0
    seen: set[Hashable] = set()
    dcg = 0.0
    for i, item in enumerate(_top(recommended, k)):
        if item in relevant and item not in seen:      # a duplicate never counts twice
            dcg += 1.0 / math.log2(i + 2)
        seen.add(item)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal


def reciprocal_rank_at_k(recommended: Sequence[Hashable], relevant: Collection[Hashable],
                         k: int) -> float:
    for i, item in enumerate(_top(recommended, k)):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def coverage(recommendation_lists: Iterable[Sequence[Hashable]], catalog_size: int,
             k: int | None = None) -> float:
    if catalog_size < 1:
        raise ValueError("catalog_size must be positive")
    unique = {item for items in recommendation_lists for item in (items[:k] if k else items)}
    return len(unique) / catalog_size


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC AUC via rank statistics (Mann-Whitney U), O(n log n), ties averaged."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores differ in length")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(1 for y in labels if y)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC needs both positive and negative labels")
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_rankings(recommendations: Mapping[Hashable, Sequence[Hashable]],
                      truth: Mapping[Hashable, Collection[Hashable]], k: int,
                      catalog_size: int | None = None) -> dict[str, Any]:
    """Average per-user metrics over users present in `truth` (missing list = empty list)."""
    users = [u for u in truth if truth[u]]
    if not users:
        raise ValueError("truth has no user with relevant items")
    lists = [list(recommendations.get(u, ())) for u in users]
    rel = [truth[u] for u in users]
    n = len(users)
    result: dict[str, Any] = {
        "users": n, "k": k,
        f"hit_rate_at_{k}": sum(hit_rate_at_k(r, t, k) for r, t in zip(lists, rel)) / n,
        f"ndcg_at_{k}": sum(ndcg_at_k(r, t, k) for r, t in zip(lists, rel)) / n,
        f"recall_at_{k}": sum(recall_at_k(r, t, k) for r, t in zip(lists, rel)) / n,
        f"mrr_at_{k}": sum(reciprocal_rank_at_k(r, t, k) for r, t in zip(lists, rel)) / n,
    }
    if catalog_size:
        result["coverage"] = coverage(lists, catalog_size, k)
    return {key: round(v, 4) if isinstance(v, float) else v for key, v in result.items()}
