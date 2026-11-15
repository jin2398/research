"""Stage 2 정렬 전략 (전략 패턴)

추천 알고리즘을 코드 수정 없이 바꾸기 위한 모듈이다.
Stage 1(Two-Tower 후보 300개)과 Stage 3(Re-ranking)는 그대로 두고,
후보를 어떤 기준으로 정렬할지만 교체한다.

선택 방법 (우선순위 순)
  1. 환경변수  RANKING_STRATEGY=newest
  2. config/stage12.yaml  →  serving.ranking_strategy: newest

전략 추가 방법: RankingStrategy를 상속한 클래스를 만들고 @register 를 붙이면 끝.

모든 전략의 반환 형식: [(product_id, score)]  (score 0~1, 높은 순)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import RecommendationService
    from .stage12_data import UserContext

DEFAULT_STRATEGY = "deepfm"
ENV_VAR = "RANKING_STRATEGY"

_REGISTRY: dict[str, type["RankingStrategy"]] = {}


def register(cls: type["RankingStrategy"]) -> type["RankingStrategy"]:
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_strategy(name: str) -> "RankingStrategy":
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"알 수 없는 ranking_strategy '{name}'. 사용 가능: {available()}") from None


def resolve_name(raw_config: dict[str, Any], override: str | None = None) -> str:
    return (override or os.getenv(ENV_VAR)
            or (raw_config.get("serving") or {}).get("ranking_strategy")
            or DEFAULT_STRATEGY)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full(len(values), 0.5)
    return (values - lo) / (hi - lo)


def _sorted_pairs(svc: "RecommendationService", rows: np.ndarray,
                  scores: np.ndarray) -> list[tuple[str, float]]:
    order = np.argsort(-scores, kind="stable")
    ids = svc.items.ids
    return [(str(ids[rows[i]]), float(scores[i])) for i in order]


class RankingStrategy(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def rank(self, svc: "RecommendationService", ctx: "UserContext", rows: np.ndarray,
             two_tower_scores: np.ndarray, now_ts: float) -> list[tuple[str, float]]:
        """rows: Stage 1 후보의 item row index. two_tower_scores: 같은 순서의 내적 점수."""


@register
class DeepFMStrategy(RankingStrategy):
    name = "deepfm"
    description = "DeepFM이 예측한 클릭 확률(CTR) 순 (기본값)"

    def rank(self, svc, ctx, rows, two_tower_scores, now_ts):
        ranked = svc.ranker.rank(ctx, rows, two_tower_scores, now_ts)
        ids = svc.items.ids
        return [(str(ids[r]), float(score)) for r, score, _, _ in ranked]


@register
class SimilarityStrategy(RankingStrategy):
    name = "similarity"
    description = "Two-Tower 사용자-상품 벡터 유사도 순"

    def rank(self, svc, ctx, rows, two_tower_scores, now_ts):
        return _sorted_pairs(svc, rows, _minmax(two_tower_scores))


@register
class NewestStrategy(RankingStrategy):
    name = "newest"
    description = "신상품 지수(newness_value, 클수록 최신) 순"

    def rank(self, svc, ctx, rows, two_tower_scores, now_ts):
        newness = svc.item_newness()
        return _sorted_pairs(svc, rows, newness[rows])


@register
class PopularityStrategy(RankingStrategy):
    name = "popularity"
    description = "전체 로그 인기도 순"

    def rank(self, svc, ctx, rows, two_tower_scores, now_ts):
        pop = np.asarray(svc.popularity, dtype=np.float64)[rows]
        top = float(pop.max()) if len(pop) else 0.0
        return _sorted_pairs(svc, rows, pop / top if top > 0 else pop)
