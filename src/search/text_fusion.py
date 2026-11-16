"""Experimental text ranking over BM25 and CLIP candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .bm25 import tokenize
from .catalog import CatalogItem


@dataclass(frozen=True, slots=True)
class FusionSettings:
    depth: int = 50
    clip_weight: float = 0.2
    rank_constant: int = 20
    match_attributes: bool = False


class AttributeMatcher:
    """Recognize unambiguous literal catalog phrases in the request text."""

    def __init__(self, products: Sequence[CatalogItem]) -> None:
        fields = ("category_small", "color", "pattern")
        self.values = {
            item.product_id: tuple(
                getattr(item, field).strip().casefold() for field in fields
            )
            for item in products
        }
        self.phrases: list[dict[tuple[str, ...], set[str]]] = []
        for values in zip(*self.values.values()):
            phrases: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for value in set(values):
                if tokens := tuple(tokenize(value)):
                    phrases[tokens].add(value)
            self.phrases.append(phrases)
        self.max_lengths = [
            max(map(len, phrases), default=0) for phrases in self.phrases
        ]

    def constraints(self, query: str) -> dict[int, str]:
        tokens = tuple(tokenize(query))
        used: set[int] = set()
        result = {}
        for field, (phrases, max_length) in enumerate(
            zip(self.phrases, self.max_lengths)
        ):
            for length in range(min(max_length, len(tokens)), 0, -1):
                hits = [
                    (start, value)
                    for start in range(len(tokens) - length + 1)
                    if not used.intersection(range(start, start + length))
                    for value in phrases.get(tokens[start : start + length], ())
                ]
                if hits:
                    values = {value for _, value in hits}
                    if len(values) == 1:
                        result[field] = hits[0][1]
                        for start, _ in hits:
                            used.update(range(start, start + length))
                    break
        return result

    def penalties(self, query: str, candidates: Sequence[str]) -> dict[str, int]:
        constraints = self.constraints(query)
        if not constraints:
            return {}
        return {
            product_id: sum(
                self.values[product_id][field] != value
                for field, value in constraints.items()
            )
            for product_id in candidates
        }


def fuse_rankings(
    bm25: Sequence[str],
    clip: Sequence[str],
    settings: FusionSettings,
    penalties: dict[str, int] | None = None,
    *,
    top_k: int = 10,
) -> list[str]:
    """Combine rankings on a common rank scale; keep all candidates eligible."""
    return [
        pid
        for pid, _ in fuse_rankings_scored(bm25, clip, settings, penalties, top_k=top_k)
    ]


def fuse_rankings_scored(
    bm25: Sequence[str],
    clip: Sequence[str],
    settings: FusionSettings,
    penalties: dict[str, int] | None = None,
    *,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Same order as fuse_rankings, with a 0~1 score per product.

    score = fused reciprocal-rank score * (rank_constant + 1), i.e. 1.0 for an item ranked
    first by both retrievers. With attribute matching, each unmatched attribute halves it,
    and the score is capped by the previous item's so it never increases down the list.
    """
    if (
        settings.depth < 1
        or not 0 <= settings.clip_weight <= 1
        or settings.rank_constant < 1
        or not 1 <= top_k <= settings.depth
    ):
        raise ValueError("Invalid fusion settings or top_k")
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in (
        (bm25, 1 - settings.clip_weight),
        (clip, settings.clip_weight),
    ):
        if weight:
            for rank, product_id in enumerate(ranking[: settings.depth], 1):
                scores[product_id] += weight / (settings.rank_constant + rank)
    scale = settings.rank_constant + 1
    if settings.match_attributes and penalties:
        order = sorted(scores, key=lambda pid: (penalties[pid], -scores[pid]))[:top_k]
        result, cap = [], 1.0
        for pid in order:
            cap = min(cap, min(1.0, scores[pid] * scale) / (2 ** penalties[pid]))
            result.append((pid, round(cap, 5)))
        return result
    order = sorted(scores, key=lambda pid: -scores[pid])[:top_k]
    return [(pid, round(min(1.0, scores[pid] * scale), 5)) for pid in order]
