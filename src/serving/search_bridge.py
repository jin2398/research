"""RE:SEARCH local API bridge for the team-provided CLIP HNSW index.

Launch from the project root: python -m uvicorn src.serving.search_bridge:app --port 8000
No index is rebuilt and no fabricated recommendation values are supplied.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "artifacts" / "search" / "clip_hnsw_full"
REQUIRED_INDEX_FILES = ("manifest.json", "products.jsonl", "products.index")
MODEL_CONFIG = ROOT / "config" / "search_full.yaml"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(title="RE:SEARCH 검색 브리지", version="1.0")
# localhost-only development API, supports opaque-origin Streamlit iframes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
_lock = threading.RLock()
_engine: Any = None
_bm25: Any = None
_attribute_matcher: Any = None
_catalog: Any = None
_lookup: dict[str, Any] | None = None


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=20, ge=1, le=50)
    mode: str = Field(default="text")


def index_status() -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_INDEX_FILES if not (INDEX_DIR / name).is_file()
    ]
    manifest = None
    try:
        if (INDEX_DIR / "manifest.json").is_file():
            manifest = json.loads(
                (INDEX_DIR / "manifest.json").read_text(encoding="utf-8")
            )
    except (OSError, ValueError):
        missing.append("manifest.json (형식 오류)")
    files_present = not missing
    if files_present and manifest:
        if manifest.get("model_name") != "openai/clip-vit-base-patch32":
            missing.append("모델 이름 불일치")
        if manifest.get("dimension") != 512:
            missing.append("임베딩 차원 불일치")
    if not MODEL_CONFIG.is_file():
        missing.append("config/search_full.yaml")
    for src in ("clip_faiss.py", "query_text.py", "text_fusion.py"):
        if not (ROOT / "src" / "search" / src).is_file():
            missing.append("src/search/" + src)
    return {
        "index_ready": not missing,
        "model_loaded": _engine is not None,
        "indexed_products": manifest.get("indexed_products") if manifest else None,
        "missing": sorted(set(missing)),
        "note": (
            "모델은 첫 검색 때 로드됩니다. 최초 실행 시 모델 가중치 다운로드가 필요할 수 있습니다."
            if not missing
            else "두 ZIP의 파일을 README_SEARCH_SETUP.md 안내 경로로 넣어주세요."
        ),
    }


@app.get("/api/search/status")
def status() -> dict[str, Any]:
    return index_status()


def _get_engine() -> Any:
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        missing = index_status()["missing"]
        if missing:
            raise HTTPException(503, "검색 파일 누락/설정 오류: " + ", ".join(missing))
        try:
            from src.search.clip_faiss import ClipFaissSearch, SearchConfig

            cfg = SearchConfig.from_yaml(MODEL_CONFIG)
            # User-supplied index was built with exactly this checkpoint.
            # By default, first run is allowed to download weights.
            cfg = replace(
                cfg, model_local_files_only=os.getenv("SEARCH_OFFLINE_ONLY") == "1"
            )
            if cfg.model_name != "openai/clip-vit-base-patch32":
                raise ValueError(
                    "인덱스는 openai/clip-vit-base-patch32 모델과만 호환됩니다."
                )
            _engine = ClipFaissSearch(cfg)
        except Exception as exc:
            # No silent demo fallback when real index search fails.
            raise HTTPException(
                503, f"CLIP 검색 모델을 로드할 수 없습니다: {exc}"
            ) from exc
        return _engine


def _ensure_rankers(engine: Any) -> tuple[Any, Any, Any]:
    global _bm25, _attribute_matcher, _catalog
    with _lock:
        if _bm25 is None:
            from src.search.bm25 import BM25Index
            from src.search.catalog import CatalogItem
            from src.search.text_fusion import AttributeMatcher

            _catalog = [
                CatalogItem(
                    product_id=str(r["product_id"]),
                    name=str(r.get("name", "")),
                    price=int(r.get("price") or 0),
                    category_small=str(r.get("category_small") or ""),
                    color=str(r.get("color") or ""),
                    pattern=str(r.get("pattern") or ""),
                    image_path=Path(""),
                )
                for r in engine.products
            ]
            _bm25 = BM25Index(_catalog)
            _attribute_matcher = AttributeMatcher(_catalog)
    return _bm25, _attribute_matcher, _catalog


def _to_product(row: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    pid = str(row["product_id"])
    return {
        "product_id": pid,
        "id": pid,
        "name": str(row.get("name") or pid),
        "price": row.get("price"),
        "category": row.get("category_small") or "기타",
        "category_small": row.get("category_small") or "",
        "color": row.get("color") or "",
        "pattern": row.get("pattern") or "",
        "score": round(float(score), 5) if score is not None else None,
        "image_url": f"http://127.0.0.1:8000/api/search/images/{pid}",
        "badge": "CLIP 검색",
        "rating": None,
    }


def execute_search(
    query: str = "", image: Path | None = None, top_k: int = 20
) -> dict[str, Any]:
    global _lookup
    engine = _get_engine()
    if _lookup is None:
        with _lock:
            if _lookup is None:
                _lookup = {str(item["product_id"]): item for item in engine.products}
    by_id = _lookup
    query = query.strip()
    if not query and image is None:
        raise HTTPException(422, "검색어 또는 이미지가 필요합니다.")
    try:
        # Text: use the supplied attribute-fusion settings, with the local
        # BM25 retriever built from the indexed metadata (not the original
        # team's BM25 implementation). This is NOT the original evaluation run.
        if query and image is None:
            from src.search.query_text import normalize_query_text
            from src.search.text_fusion import FusionSettings, fuse_rankings_scored

            bm25, matcher, _ = _ensure_rankers(engine)
            q = normalize_query_text(query)
            setting = FusionSettings(
                depth=100, clip_weight=0.15, rank_constant=60, match_attributes=True
            )
            candidates = bm25.search(q, setting.depth)
            clip = engine.search(query_text=query, top_k=setting.depth)
            clip_ids = [str(item["product_id"]) for item in clip["results"]]
            penalties = matcher.penalties(q, list(dict.fromkeys(candidates + clip_ids)))
            ordered = fuse_rankings_scored(
                candidates,
                clip_ids,
                setting,
                penalties,
                top_k=min(top_k, setting.depth),
            )
            products = [
                _to_product(by_id[pid], score) for pid, score in ordered if pid in by_id
            ]
            method = "attribute_fusion_local"
            latency = clip["latency_ms"]
        else:
            raw = engine.search(
                query_text=query or None,
                query_image_path=str(image) if image else None,
                top_k=top_k,
            )
            products = [
                _to_product(by_id[str(item["product_id"])], item["score"])
                for item in raw["results"]
                if str(item["product_id"]) in by_id
            ]
            method = "clip_hybrid" if query else "clip_image"
            latency = raw["latency_ms"]
        return {
            "search_type": "hybrid"
            if query and image
            else ("image" if image else "text"),
            "method": method,
            "total_count": len(products),
            "indexed_products": len(engine.products),
            "results": products,
            "latency_ms": latency,
            "notice": "실제 인덱스 검색 결과입니다. 상품 이미지는 별도 이미지 폴더가 필요합니다.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"검색 처리 중 오류: {exc}") from exc


@app.post("/api/search")
async def search(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()
    if (
        "multipart/form-data" in content_type
        or "application/x-www-form-urlencoded" in content_type
    ):
        form = await request.form()
        mode = str(form.get("mode") or "image")
        q = str(form.get("query") or "").strip()
        try:
            top_k = int(form.get("top_k") or 20)
        except ValueError:
            raise HTTPException(422, "top_k는 정수여야 합니다.")
        if not 1 <= top_k <= 50:
            raise HTTPException(422, "top_k는 1~50 사이여야 합니다.")
        file = form.get("file")
        if not file or not hasattr(file, "read"):
            raise HTTPException(422, "이미지 검색에는 file 필드가 필요합니다.")
        if mode == "hybrid" and not q:
            raise HTTPException(422, "하이브리드 모드에는 검색어도 필요합니다.")
        payload = await file.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "이미지는 10MB 이하여야 합니다.")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as output:
            p = Path(output.name)
        try:
            import io

            with Image.open(io.BytesIO(payload)) as img:
                img.verify()
            with Image.open(io.BytesIO(payload)) as img:
                img.convert("RGB").save(p, format="JPEG", quality=90)
            return execute_search(
                query=q if mode == "hybrid" else "", image=p, top_k=top_k
            )
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                422, "올바른 이미지 파일(JPEG/PNG/WEBP)이 아닙니다."
            ) from exc
        finally:
            p.unlink(missing_ok=True)
    try:
        body = TextSearchRequest.model_validate(await request.json())
    except Exception as exc:
        raise HTTPException(
            422, "검색 JSON은 query, mode, top_k 형식을 사용해야 합니다."
        ) from exc
    if body.mode != "text":
        raise HTTPException(422, "이미지 검색은 multipart/form-data로 요청하세요.")
    return execute_search(query=body.query, top_k=body.top_k)


@app.get("/api/search/images/{product_id}")
def product_image(product_id: str) -> Any:
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,40}", product_id):
        raise HTTPException(400, "올바르지 않은 상품 ID")
    root = Path(os.getenv("SEARCH_IMAGE_ROOT") or ROOT / "data" / "product_images")
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = root / (product_id + ext)
        if path.is_file():
            return FileResponse(path)
    return FileResponse(
        ROOT / "src" / "dashboard" / "assets" / "no_image.jpg", media_type="image/jpeg"
    )
