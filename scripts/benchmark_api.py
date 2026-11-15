"""API 응답 시간 검증 (목표: 검색·추천 p95 ≤ 200ms)

클라이언트에서 HTTP 요청 전체 시간을 잰다 (네트워크 + 서버 처리 전부 포함).
측정 전에 각 엔드포인트를 10번 호출해 워밍업한다.

  python scripts/benchmark_api.py --base-url http://localhost:8000 --calls 100 --save
  → docs/api_latency.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["black hoodie", "beige coat", "검정 오버핏 후드티", "denim jeans", "white sneakers",
           "floral dress", "leather bag", "striped shirt", "wool sweater", "summer skirt"]
IMAGES = sorted((ROOT / "src" / "dashboard" / "assets").glob("product_*.jpg"))
THRESHOLD_MS = 200.0


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100 * len(ordered)), 1) - 1]


def timed(call) -> float:
    start = time.perf_counter()
    response = call()
    elapsed = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    base, http = args.base_url.rstrip("/"), requests.Session()
    images = [path.read_bytes() for path in IMAGES]

    cases = {
        "search_text": lambda i: http.post(f"{base}/api/search",
                                           json={"query_text": QUERIES[i % len(QUERIES)],
                                                 "top_k": 10}),
        "search_image": lambda i: http.post(f"{base}/api/search", data={"top_k": "10"},
                                            files={"file": ("q.jpg", images[i % len(images)],
                                                            "image/jpeg")}),
        "search_hybrid": lambda i: http.post(f"{base}/api/search",
                                             data={"top_k": "10",
                                                   "query_text": QUERIES[i % len(QUERIES)]},
                                             files={"file": ("q.jpg", images[i % len(images)],
                                                             "image/jpeg")}),
        "recommend": lambda i: http.get(f"{base}/api/recommend",
                                        params={"user_id": f"U{(i % 8000) + 1:05d}",
                                                "top_n": 10}),
        "recommend_cold_start": lambda i: http.get(f"{base}/api/recommend",
                                                   params={"user_id": f"NEW_{i}", "top_n": 10}),
    }

    results = {}
    for name, call in cases.items():
        for i in range(10):
            timed(lambda: call(i))
        samples = [timed(lambda: call(i)) for i in range(args.calls)]
        p95 = percentile(samples, 95)
        results[name] = {"p50_ms": round(percentile(samples, 50), 1), "p95_ms": round(p95, 1),
                         "max_ms": round(max(samples), 1), "passed": p95 <= THRESHOLD_MS}
        mark = "PASS" if p95 <= THRESHOLD_MS else "FAIL"
        print(f"{name:22s} p50 {results[name]['p50_ms']:7.1f}ms  p95 {p95:7.1f}ms  [{mark}]")

    report = {"measured_at": datetime.now().isoformat(timespec="seconds"),
              "base_url": base, "calls": args.calls, "warmup": 10, "threshold_ms": THRESHOLD_MS,
              "client": f"{platform.system()} {platform.machine()}",
              "note": "HTTP 요청 전체 시간 (클라이언트 측정). 서버 사양에 따라 달라진다.",
              "results": results}
    if args.save:
        path = ROOT / "docs" / "api_latency.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ {path}")


if __name__ == "__main__":
    main()
