"""통합 API 테스트용 fixture: 실제 Redis·모델 없이 응답 규격을 검사한다."""

import os

import fakeredis
import pytest

os.environ.setdefault("PRELOAD_MODELS", "0")      # 테스트에서는 모델을 미리 로딩하지 않음


class FakeStrategy:
    name = "deepfm"


class FakeRecommender:
    """RecommendationService와 같은 응답 형식을 내는 가짜 추천 서비스"""

    items = {f"P{i:03d}" for i in range(1, 31)}
    strategy = FakeStrategy()

    def __init__(self):
        self.calls = []
        self.feedback = []

    def has_item(self, product_id):
        return product_id in self.items

    def recommend(self, user_id, top_n=10, recent_clicks=()):
        self.calls.append((user_id, top_n, list(recent_clicks)))
        cold = not recent_clicks
        recs = []
        for i in range(top_n):
            explore = i == top_n - 1 and top_n > 1
            recs.append({"product_id": f"P{i + 1:03d}", "score": round(0.9 - i * 0.05, 3),
                         "reason": "mab_exploration" if explore else ("popular" if cold else "ranking"),
                         "is_exploration": explore})
        return {"user_id": user_id, "recommendations": recs,
                "pipeline_latency": {"candidate_ms": 2.0, "ranking_ms": 1.5,
                                     "reranking_ms": 3.0, "total_ms": 6.5},
                "session_context": {"recent_clicks": list(recent_clicks)[-5:],
                                    "session_interest": "상의" if recent_clicks else None},
                "cold_start": cold, "candidate_count": 300, "ranking_strategy": "deepfm"}

    def record_feedback(self, product_id, clicked):
        self.feedback.append((product_id, clicked))


def fake_search(*, query="", image=None, top_k=10):
    kind = "hybrid" if query and image else ("image" if image else "text")
    results = [{"product_id": f"P{i:03d}", "name": f"상품 {i}", "price": 10000 * i,
                "score": round(0.95 - i * 0.01, 3), "category": "상의"}
               for i in range(1, top_k + 1)]
    return {"search_type": kind, "results": results, "latency_ms": 1.0,
            "total_count": len(results)}


@pytest.fixture
def services(monkeypatch):
    from src.serving import api, feature_store as fs, search_bridge

    monkeypatch.setattr(fs, "redis_client", fakeredis.FakeRedis(decode_responses=True))
    monkeypatch.setattr(search_bridge, "execute_search", fake_search)
    fake = FakeRecommender()
    monkeypatch.setattr(api.services, "recommender", fake)
    return fake


@pytest.fixture
def client(services):
    from fastapi.testclient import TestClient

    from src.serving.api import app

    return TestClient(app)
