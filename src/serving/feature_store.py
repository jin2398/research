"""Redis Feature Store: 사용자 실시간 행동 피처

키 구조 (user_id별)
  user:{id}:views     최근 조회 상품 (최신순, 중복 제거, 최대 100개, 30일)
  user:{id}:clicks    최근 클릭/장바구니/구매 상품 (최신순, 중복 제거, 최대 100개, 30일)
  user:{id}:session   현재 세션의 모든 상호작용 (최신순, 중복 허용, 최대 20개, 30분)
                      → 추천 API가 recent_clicks로 넘겨 세션 기반 추천에 즉시 반영
  user:{id}:explore   직전 추천에서 보여준 MAB 탐색 상품 (30분)
                      → 클릭되면 보상 1, 다음 추천 때까지 클릭 없으면 보상 0
"""

from __future__ import annotations

import math
import os
import time

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=3,
    socket_timeout=3,
)

RECORD_TTL = 60 * 60 * 24 * 30  # 행동 기록 보관: 30일
SESSION_TTL = 60 * 30  # 세션 만료: 마지막 행동 후 30분
MAX_HISTORY = 100  # 사용자별 최근 기록 개수
SESSION_LENGTH = 20  # 세션 클릭 개수 (config/rerank.yaml session.window 와 동일)
LATENCY_THRESHOLD_MS = 10.0  # Feature Store 조회 목표

EVENT_TYPES = ("view", "click", "cart", "purchase")
_HISTORY_KEY = {
    "view": "views",
    "click": "clicks",
    "cart": "clicks",
    "purchase": "clicks",
}


def _key(user_id: str, name: str) -> str:
    return f"user:{user_id}:{name}"


def check_connection() -> bool:
    return bool(redis_client.ping())


# ------------------------------------------------------------------ write
def record_event(user_id: str, product_id: str, event_type: str) -> None:
    """행동 1건 저장: 기록 리스트 + 세션 리스트 갱신 (Redis 왕복 1회)"""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"event_type은 {EVENT_TYPES} 중 하나여야 합니다.")
    history = _key(user_id, _HISTORY_KEY[event_type])
    session = _key(user_id, "session")

    with redis_client.pipeline() as pipe:
        pipe.lrem(history, 0, product_id)
        pipe.lpush(history, product_id)
        pipe.ltrim(history, 0, MAX_HISTORY - 1)
        pipe.expire(history, RECORD_TTL)
        pipe.lpush(session, product_id)
        pipe.ltrim(session, 0, SESSION_LENGTH - 1)
        pipe.expire(session, SESSION_TTL)
        pipe.execute()


def record_view(user_id: str, product_id: str) -> None:
    record_event(user_id, product_id, "view")


def record_click(user_id: str, product_id: str) -> None:
    record_event(user_id, product_id, "click")


def pop_exploration_hit(user_id: str, product_id: str) -> bool:
    """클릭한 상품이 직전 추천의 탐색 상품이면 목록에서 빼고 True"""
    return bool(redis_client.srem(_key(user_id, "explore"), product_id))


def replace_exploration(user_id: str, product_ids: list[str]) -> list[str]:
    """새 탐색 상품으로 교체하고, 클릭되지 않은 채 남아 있던 이전 탐색 상품을 반환"""
    key = _key(user_id, "explore")
    with redis_client.pipeline() as pipe:
        pipe.smembers(key)
        pipe.delete(key)
        if product_ids:
            pipe.sadd(key, *product_ids)
            pipe.expire(key, SESSION_TTL)
        previous = pipe.execute()[0]
    return sorted(previous)


# ------------------------------------------------------------------ read
def get_features(user_id: str) -> dict:
    """추천에 필요한 사용자 피처 전체 (Redis 왕복 1회)"""
    with redis_client.pipeline(transaction=False) as pipe:
        pipe.lrange(_key(user_id, "views"), 0, MAX_HISTORY - 1)
        pipe.lrange(_key(user_id, "clicks"), 0, MAX_HISTORY - 1)
        pipe.lrange(_key(user_id, "session"), 0, SESSION_LENGTH - 1)
        views, clicks, session = pipe.execute()

    return {
        "user_id": user_id,
        "recent_views": views,
        "recent_clicks": clicks,
        # 추천 모델 입력 형식: 오래된 것 → 최신
        "session_clicks": list(reversed(session)),
    }


def get_history(user_id: str) -> dict:
    features = get_features(user_id)
    return {k: features[k] for k in ("user_id", "recent_views", "recent_clicks")}


# ------------------------------------------------------------------ latency
def _percentile(sorted_values: list[float], p: float) -> float:
    """nearest-rank 방식 백분위수"""
    rank = math.ceil(p / 100 * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def measure_read_latency(
    user_id: str, repeat: int = 100, threshold_ms: float = LATENCY_THRESHOLD_MS
) -> dict:
    """추천 API가 쓰는 피처 조회(get_features)를 1회씩 재서 p95로 목표 달성 판정"""
    if repeat < 1:
        raise ValueError("repeat는 1 이상이어야 합니다.")

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        get_features(user_id)
        timings.append((time.perf_counter() - start) * 1000)

    timings.sort()
    p95 = _percentile(timings, 95)
    return {
        "repeat": repeat,
        "threshold_ms": threshold_ms,
        "average_ms": round(sum(timings) / repeat, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(_percentile(timings, 99), 3),
        "max_ms": round(timings[-1], 3),
        "passed": p95 <= threshold_ms,
    }
