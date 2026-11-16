"""API 요청/응답 규격 (미션 '필수 응답 필드' 기준)

response_model로 지정하면 FastAPI가 응답을 내보내기 전에 필수 필드와 타입을 검사한다.
규격에 맞지 않는 응답은 클라이언트로 나가지 않고 500 에러가 된다.
필수 필드 외의 추가 필드(image_url, cold_start 등)는 extra="allow"로 그대로 전달한다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


# ------------------------------------------------------------------ 검색
class SearchRequest(BaseModel):
    """JSON 텍스트 검색. query_text(미션 규격)와 query(기존 UI) 둘 다 받는다."""

    query_text: str | None = Field(default=None, max_length=300)
    query: str | None = Field(default=None, max_length=300)
    mode: Literal["text"] = "text"
    top_k: int = Field(default=10, ge=1, le=100)

    @property
    def text(self) -> str:
        return (self.query_text or self.query or "").strip()


class SearchResult(_Open):
    product_id: str = Field(min_length=1)
    name: str
    score: float = Field(ge=-1.0, le=1.0)
    price: float = Field(ge=0)


class SearchResponse(_Open):
    search_type: Literal["text", "image", "hybrid"]
    results: list[SearchResult]
    latency_ms: float = Field(ge=0)
    total_count: int = Field(ge=0)


# ------------------------------------------------------------------ 추천
class Recommendation(_Open):
    product_id: str = Field(min_length=1)
    score: float
    reason: str = Field(min_length=1)
    is_exploration: bool


class PipelineLatency(_Open):
    candidate_ms: float = Field(ge=0)
    ranking_ms: float = Field(ge=0)
    reranking_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class SessionContext(_Open):
    recent_clicks: list[str]
    session_interest: str | None


class RecommendResponse(_Open):
    user_id: str
    recommendations: list[Recommendation]
    pipeline_latency: PipelineLatency
    session_context: SessionContext


# ------------------------------------------------------------------ 이벤트 / Feature Store
class EventRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    product_id: str = Field(min_length=1, max_length=64)
    event_type: Literal["view", "click", "cart", "purchase"] = "view"


class ActionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    product_id: str = Field(min_length=1, max_length=64)


class EventResponse(BaseModel):
    status: str
    action: str
    exploration_reward: bool = False


class HistoryResponse(BaseModel):
    user_id: str
    recent_views: list[str]
    recent_clicks: list[str]


class LatencyResponse(BaseModel):
    repeat: int
    threshold_ms: float
    average_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    passed: bool
