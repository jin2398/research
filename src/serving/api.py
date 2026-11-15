"""RE:SEARCH 통합 API 서버 (FastAPI)

  POST /api/search                 멀티모달 검색 (텍스트 / 이미지 / 텍스트+이미지)
  GET  /api/recommend              개인화 추천 (Stage 1→2→3 + Redis 세션 피처)
  POST /api/events                 행동 로그 수집 → Redis Feature Store
  GET  /api/feature-store/latency  Feature Store 조회 속도 검증 (p95 ≤ 10ms)

실행: uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
환경변수
  REDIS_URL         Redis 주소 (기본 redis://localhost:6379/0)
  PRELOAD_MODELS    1이면 서버 시작 시 검색·추천 모델을 미리 로딩 (기본 1)
  RANKING_STRATEGY  추천 정렬 전략 (config/stage12.yaml serving.ranking_strategy 보다 우선)
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from redis.exceptions import RedisError

from src.serving import feature_store as fs
from src.serving import search_bridge
from src.serving.schemas import (
    ActionRequest,
    EventRequest,
    EventResponse,
    HistoryResponse,
    LatencyResponse,
    RecommendResponse,
    SearchRequest,
    SearchResponse,
)

ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
CLICK_EVENTS = {"click", "cart", "purchase"}


def _log(message: str) -> None:
    print(f"[api-server] {message}", flush=True)


class Services:
    """무거운 모델 객체를 프로세스당 한 번만 만든다. 테스트에서는 가짜 객체로 교체한다."""

    def __init__(self) -> None:
        self.recommender: Any = None
        self.recommender_error: str | None = None
        self._lock = threading.Lock()          # 추천 + MAB 피드백은 순서대로 처리

    def load_recommender(self) -> Any:
        if self.recommender is None:
            with self._lock:
                if self.recommender is None:
                    from src.recommendation.pipeline import RecommendationService

                    svc = RecommendationService()
                    self.recommender = svc
                    _log(f"Two-Tower + FAISS 후보 생성 모델 로딩 완료 ({svc.items.n_items:,} 상품)")
                    _log(f"DeepFM Ranking 모델 로딩 완료 (정렬 전략: {svc.strategy.name})")
                    _log(f"Stage 3 Re-ranking 로딩 완료 (총 {svc.load_s}s)")
        return self.recommender

    def require_recommender(self) -> Any:
        try:
            return self.load_recommender()
        except Exception as exc:  # noqa: BLE001 - 모델 파일 누락 등은 503으로 알린다
            self.recommender_error = str(exc)
            raise HTTPException(503, f"추천 모델을 로드할 수 없습니다: {exc}") from exc


services = Services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    os.chdir(ROOT)          # config 파일의 상대 경로(artifacts/, data/) 기준
    if os.getenv("PRELOAD_MODELS", "1") == "1":
        try:
            engine = search_bridge._get_engine()
            search_bridge._ensure_rankers(engine)          # 텍스트 검색용 BM25 색인
            search_bridge.execute_search(query="warm up", top_k=10)   # 첫 요청 지연 제거
            _log(f"CLIP + FAISS 검색 인덱스 로딩 완료 ({len(engine.products):,} 상품)")
        except Exception as exc:  # noqa: BLE001
            _log(f"검색 모델 로딩 실패 (검색 API는 503 응답): {exc}")
        try:
            services.load_recommender()
        except Exception as exc:  # noqa: BLE001
            services.recommender_error = str(exc)
            _log(f"추천 모델 로딩 실패 (추천 API는 503 응답): {exc}")
    _log("http://localhost:8000 에서 API 서버 시작 (문서: /docs)")
    yield


app = FastAPI(title="RE:SEARCH Multimodal Search & Recommendation API",
              version="1.0.0", lifespan=lifespan)
# 로컬 개발용: Streamlit iframe(쇼핑 화면)이 브라우저에서 직접 호출한다.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def _redis_unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)


# ------------------------------------------------------------------ 상태
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health/redis")
def redis_health():
    try:
        fs.check_connection()
        return {"status": "ok", "redis": "connected"}
    except RedisError:
        raise _redis_unavailable("Redis connection failed")


@app.get("/api/status")
def api_status():
    svc = services.recommender
    return {
        "search": search_bridge.index_status(),
        "recommendation": {
            "model_loaded": svc is not None,
            "ranking_strategy": getattr(getattr(svc, "strategy", None), "name", None),
            "error": services.recommender_error,
        },
    }


# 쇼핑 화면이 쓰는 검색 상태 / 상품 이미지 엔드포인트는 기존 구현을 그대로 노출
app.add_api_route("/api/search/status", search_bridge.status, methods=["GET"])
app.add_api_route("/api/search/images/{product_id}", search_bridge.product_image,
                  methods=["GET"])


# ------------------------------------------------------------------ 검색
def _save_upload(payload: bytes) -> Path:
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "이미지는 10MB 이하여야 합니다.")
    try:
        with Image.open(io.BytesIO(payload)) as img:
            img.verify()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as output:
            path = Path(output.name)
        with Image.open(io.BytesIO(payload)) as img:
            img.convert("RGB").save(path, format="JPEG", quality=90)
        return path
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, "올바른 이미지 파일(JPEG/PNG/WEBP)이 아닙니다.") from exc


async def _parse_search(request: Request) -> tuple[str, bytes | None, int]:
    """JSON(텍스트) 또는 multipart(file + 선택적 query_text) → (텍스트, 이미지, top_k)"""
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        text = str(form.get("query_text") or form.get("query") or "").strip()
        try:
            top_k = int(form.get("top_k") or 10)
        except ValueError:
            raise HTTPException(422, "top_k는 정수여야 합니다.")
        if not 1 <= top_k <= 100:
            raise HTTPException(422, "top_k는 1~100 사이여야 합니다.")
        upload = form.get("file") or form.get("image")
        payload = await upload.read(MAX_IMAGE_BYTES + 1) if hasattr(upload, "read") else None
        mode = str(form.get("mode") or "")
        if mode in ("image", "hybrid") and not payload:
            raise HTTPException(422, "이미지 검색에는 file 필드가 필요합니다.")
        if mode == "hybrid" and not text:
            raise HTTPException(422, "하이브리드 검색에는 query_text도 필요합니다.")
        if mode == "image":
            text = ""
        if not text and not payload:
            raise HTTPException(422, "query_text 또는 file이 필요합니다.")
        return text, payload, top_k

    try:
        body = SearchRequest.model_validate(await request.json())
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, "검색 JSON은 {query_text, top_k} 형식이어야 합니다. "
                                 "이미지는 multipart/form-data로 보내세요.") from exc
    if not body.text:
        raise HTTPException(422, "query_text가 비어 있습니다.")
    return body.text, None, body.top_k


@app.post("/api/search", response_model=SearchResponse)
async def search(request: Request):
    started = time.perf_counter()
    text, payload, top_k = await _parse_search(request)
    image = _save_upload(payload) if payload else None
    try:
        result = search_bridge.execute_search(query=text, image=image, top_k=top_k)
    finally:
        if image is not None:
            image.unlink(missing_ok=True)
    result["engine_latency_ms"] = result.get("latency_ms")
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["total_count"] = len(result["results"])
    return result


# ------------------------------------------------------------------ 추천
@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(user_id: str = Query(min_length=1, max_length=64),
              top_n: int = Query(default=10, ge=1, le=100)):
    started = time.perf_counter()
    svc = services.require_recommender()

    feature_ok = True
    t_fs = time.perf_counter()
    try:
        session_clicks = fs.get_features(user_id)["session_clicks"]
    except RedisError:
        # Redis 장애 시에도 추천은 내보낸다 (오프라인 스냅샷만 사용)
        session_clicks, feature_ok = [], False
    feature_ms = (time.perf_counter() - t_fs) * 1000

    with services._lock:
        result = svc.recommend(user_id, top_n, recent_clicks=session_clicks)
        explored = [r["product_id"] for r in result["recommendations"] if r["is_exploration"]]
        if feature_ok:
            try:
                # 지난번 탐색 상품 중 클릭되지 않은 것 = 보상 0
                for product_id in fs.replace_exploration(user_id, explored):
                    svc.record_feedback(product_id, False)
            except RedisError:
                feature_ok = False

    latency = result["pipeline_latency"]
    latency["feature_store_ms"] = round(feature_ms, 3)
    latency["api_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["feature_store"] = "ok" if feature_ok else "unavailable"
    return result


# ------------------------------------------------------------------ 이벤트
def _save_event(user_id: str, product_id: str, event_type: str) -> EventResponse:
    svc = services.recommender
    if svc is not None and not svc.has_item(product_id):
        raise HTTPException(404, f"Product not found: {product_id}")
    try:
        fs.record_event(user_id, product_id, event_type)
        rewarded = event_type in CLICK_EVENTS and fs.pop_exploration_hit(user_id, product_id)
    except RedisError:
        raise _redis_unavailable(f"Could not save {event_type} event")
    if rewarded and svc is not None:
        with services._lock:
            svc.record_feedback(product_id, True)      # 탐색 상품 클릭 = 보상 1
    return EventResponse(status="saved", action=event_type, exploration_reward=rewarded)


@app.post("/api/events", response_model=EventResponse)
def save_event(request: EventRequest):
    return _save_event(request.user_id, request.product_id, request.event_type)


@app.post("/api/events/view", response_model=EventResponse)
def save_view(request: ActionRequest):
    return _save_event(request.user_id, request.product_id, "view")


@app.post("/api/events/click", response_model=EventResponse)
def save_click(request: ActionRequest):
    return _save_event(request.user_id, request.product_id, "click")


# ------------------------------------------------------------------ Feature Store
@app.get("/api/users/{user_id}/history", response_model=HistoryResponse)
def user_history(user_id: str):
    try:
        return fs.get_history(user_id)
    except RedisError:
        raise _redis_unavailable("Could not retrieve user history")


@app.get("/api/feature-store/latency", response_model=LatencyResponse)
def feature_store_latency(user_id: str = "test_user",
                          repeat: int = Query(default=100, ge=1, le=1000)):
    try:
        return fs.measure_read_latency(user_id, repeat)
    except RedisError:
        raise _redis_unavailable("Could not measure Redis latency")
