"""API 응답 필수 필드 규격 검증 (미션 '필수 응답 필드' 표 기준)"""

import io

from PIL import Image

SEARCH_FIELDS = {"search_type", "results", "latency_ms", "total_count"}
SEARCH_RESULT_FIELDS = {"product_id", "name", "score", "price"}
RECOMMEND_FIELDS = {"user_id", "recommendations", "pipeline_latency", "session_context"}
RECOMMEND_ITEM_FIELDS = {"product_id", "score", "reason", "is_exploration"}
LATENCY_FIELDS = {"candidate_ms", "ranking_ms", "reranking_ms", "total_ms"}
SESSION_FIELDS = {"recent_clicks", "session_interest"}


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def assert_search_response(body, search_type, top_k):
    assert SEARCH_FIELDS <= body.keys()
    assert body["search_type"] == search_type
    assert isinstance(body["latency_ms"], float) and body["latency_ms"] >= 0
    assert isinstance(body["total_count"], int)
    assert body["total_count"] == len(body["results"]) <= top_k
    for item in body["results"]:
        assert SEARCH_RESULT_FIELDS <= item.keys()
        assert isinstance(item["product_id"], str) and item["product_id"]
        assert isinstance(item["name"], str)
        assert isinstance(item["score"], float)
        assert isinstance(item["price"], (int, float))


# ---------- /api/search ----------

def test_text_search_required_fields_default_top_k(client):
    res = client.post("/api/search", json={"query_text": "검정 오버핏 후드티"})
    assert res.status_code == 200
    assert_search_response(res.json(), "text", top_k=10)
    assert len(res.json()["results"]) == 10            # 기본값 K=10


def test_text_search_accepts_legacy_query_field(client):
    res = client.post("/api/search", json={"query": "베이지 코트", "mode": "text", "top_k": 3})
    assert res.status_code == 200
    assert_search_response(res.json(), "text", top_k=3)


def test_image_search_required_fields(client):
    res = client.post("/api/search", data={"top_k": "5"},
                      files={"file": ("q.png", _png(), "image/png")})
    assert res.status_code == 200
    assert_search_response(res.json(), "image", top_k=5)


def test_hybrid_search_required_fields(client):
    res = client.post("/api/search", data={"query_text": "후드티", "top_k": "5"},
                      files={"file": ("q.png", _png(), "image/png")})
    assert res.status_code == 200
    assert_search_response(res.json(), "hybrid", top_k=5)


def test_search_rejects_invalid_requests(client):
    assert client.post("/api/search", json={"query_text": ""}).status_code == 422
    assert client.post("/api/search", json={}).status_code == 422
    assert client.post("/api/search", json={"query_text": "a", "top_k": 0}).status_code == 422
    bad_image = client.post("/api/search", files={"file": ("x.png", b"not image", "image/png")})
    assert bad_image.status_code == 422
    no_image = client.post("/api/search", data={"mode": "hybrid", "query_text": "coat"})
    assert no_image.status_code == 422


# ---------- /api/recommend ----------

def assert_recommend_response(body, user_id, top_n):
    assert RECOMMEND_FIELDS <= body.keys()
    assert body["user_id"] == user_id
    assert len(body["recommendations"]) == top_n
    for item in body["recommendations"]:
        assert RECOMMEND_ITEM_FIELDS <= item.keys()
        assert isinstance(item["is_exploration"], bool)
        assert isinstance(item["reason"], str) and item["reason"]
    assert LATENCY_FIELDS <= body["pipeline_latency"].keys()
    assert SESSION_FIELDS <= body["session_context"].keys()
    assert isinstance(body["session_context"]["recent_clicks"], list)


def test_recommend_required_fields_default_top_n(client):
    res = client.get("/api/recommend", params={"user_id": "U1234"})
    assert res.status_code == 200
    assert_recommend_response(res.json(), "U1234", top_n=10)   # 기본값 N=10
    assert any(r["is_exploration"] for r in res.json()["recommendations"])


def test_recommend_uses_session_clicks_from_feature_store(client, services):
    client.post("/api/events", json={"user_id": "U9", "product_id": "P005", "event_type": "view"})
    client.post("/api/events", json={"user_id": "U9", "product_id": "P007", "event_type": "click"})

    body = client.get("/api/recommend", params={"user_id": "U9", "top_n": 5}).json()
    assert services.calls[-1] == ("U9", 5, ["P005", "P007"])    # 오래된 것 → 최신
    assert body["session_context"]["recent_clicks"] == ["P005", "P007"]
    assert "feature_store_ms" in body["pipeline_latency"]


def test_exploration_feedback_reward_and_timeout(client, services):
    first = client.get("/api/recommend", params={"user_id": "U7", "top_n": 4}).json()
    explored = [r["product_id"] for r in first["recommendations"] if r["is_exploration"]]
    assert explored == ["P004"]

    res = client.post("/api/events", json={"user_id": "U7", "product_id": "P004",
                                           "event_type": "click"})
    assert res.json()["exploration_reward"] is True
    assert services.feedback == [("P004", True)]

    client.get("/api/recommend", params={"user_id": "U7", "top_n": 4})   # P004 다시 노출
    client.get("/api/recommend", params={"user_id": "U7", "top_n": 4})   # 클릭 없음 → 보상 0
    assert services.feedback[-1] == ("P004", False)


def test_recommend_rejects_invalid_request(client):
    assert client.get("/api/recommend").status_code == 422
    assert client.get("/api/recommend", params={"user_id": "u", "top_n": 0}).status_code == 422
    assert client.get("/api/recommend", params={"user_id": "u", "top_n": 101}).status_code == 422


# ---------- 이벤트 / 기록 ----------

def test_event_saved_and_history_returned(client):
    res = client.post("/api/events/view", json={"user_id": "u2", "product_id": "P002"})
    assert res.status_code == 200
    assert res.json()["status"] == "saved"
    history = client.get("/api/users/u2/history").json()
    assert history == {"user_id": "u2", "recent_views": ["P002"], "recent_clicks": []}


def test_event_unknown_product_returns_404(client):
    res = client.post("/api/events", json={"user_id": "u", "product_id": "NOPE",
                                           "event_type": "click"})
    assert res.status_code == 404


def test_feature_store_latency_endpoint(client):
    body = client.get("/api/feature-store/latency", params={"repeat": 20}).json()
    assert body["repeat"] == 20
    assert body["threshold_ms"] == 10.0
    assert isinstance(body["passed"], bool)
