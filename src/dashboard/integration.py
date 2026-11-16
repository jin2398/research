"""Read team A search results and synthetic-session data without executing ML models.

Only provided JSON reports are treated as existing experimental evidence.
Absent recommendation model output stays missing; there are no fabricated metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEARCH_REPORTS = {
    "bm25_full": "docs/search_baseline.json",
    "bm25_pilot": "docs/search_baseline_pilot.json",
    "clip_pilot": "docs/search_clip_pilot.json",
    "latency_pilot": "docs/search_latency_pilot.json",
    "manifest": "artifacts/search/clip_hnsw/manifest.json",
}


def read_json(root: Path, relative_path: str) -> dict[str, Any] | None:
    """Return JSON object or None when no report is available."""
    path = root / relative_path
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise ValueError(f"읽을 수 없는 JSON 파일: {relative_path}: {err}") from err
    if not isinstance(value, dict):
        raise ValueError(f"JSON 최상위는 객체여야 합니다: {relative_path}")
    return value


def load_search_reports(root: Path) -> dict[str, dict[str, Any] | None]:
    return {key: read_json(root, path) for key, path in SEARCH_REPORTS.items()}


def load_session_report(root: Path) -> dict[str, Any] | None:
    return read_json(root, "docs/split_report.json")


def load_recommendation_metrics(root: Path) -> dict[str, float] | None:
    """Optional real evaluation output provided by Stage 1/2 or Stage 3 teams.

    Expected JSON keys: recall_at_300, auc, hit_rate_at_50, ndcg_at_50, coverage.
    Missing keys remain unknown and are not defaulted to demo numbers.
    """
    data = read_json(root, "data/recommendation_metrics.json")
    if data is None:
        return None
    output = {}
    for key in ("recall_at_300", "auc", "hit_rate_at_50", "ndcg_at_50", "coverage"):
        if key in data:
            value = float(data[key])
            if not 0 <= value <= 1:
                raise ValueError(f"{key} 값은 0~1 범위여야 합니다")
            output[key] = value
    return output


def search_compare_rows(reports: dict, *, covered_only: bool = False) -> list[dict]:
    """Comparable pilot index (1000 items): both methods on the same query subset."""
    rows = []
    for key, label in [
        ("bm25_pilot", "BM25 (1천 개)"),
        ("clip_pilot", "CLIP (1천 개)"),
    ]:
        report = reports.get(key)
        if not report:
            continue
        metrics = (
            report.get("covered_query_metrics" if covered_only else "metrics") or {}
        )
        rows.append(
            {
                "방법": label,
                "MRR@10": metrics.get("mrr"),
                "NDCG@10": metrics.get("ndcg_at_10"),
                "Recall@10": metrics.get("recall_at_10"),
                "평가 질의 수": metrics.get("queries"),
            }
        )
    return rows
