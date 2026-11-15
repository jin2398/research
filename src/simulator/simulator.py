"""행동 로그 시뮬레이터

가상 사용자(페르소나)가 세션 단위로 검색 → 추천 확인 → 조회 → 장바구니 → 구매를 하며
1) API(/api/search, /api/recommend, /api/events)를 실제로 호출하고
2) 같은 행동을 data/simulated_logs.csv 에 append 한다 (학습 로그와 같은 컬럼).

추천 결과에서 상품을 고르면 click 이벤트로 보내므로, 탐색 상품(is_exploration)을
클릭하면 MAB 보상이 기록되고 세션 클릭이 다음 추천에 바로 반영된다.
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Persona, SimulatorConfig

LOG_COLUMNS = ["event_id", "timestamp", "user_id", "session_id", "event_type", "product_id",
               "query_product_id", "search_query", "persona", "preferred_category",
               "category_bucket", "category_large", "category_middle", "category_small"]


def _log(message: str) -> None:
    print(f"[simulator] {message}", flush=True)


# ------------------------------------------------------------------ 카탈로그
class Catalog:
    def __init__(self, config: SimulatorConfig) -> None:
        features = pd.read_csv(config.catalog_path, dtype={"article_id": str},
                               usecols=["article_id", "category_bucket", "price", "newness_value"])
        meta = pd.read_csv(config.products_path, dtype={"article_id": str},
                           usecols=["article_id", "category_large", "category_middle",
                                    "category_small"], encoding="utf-8-sig")
        df = features.merge(meta, on="article_id", how="left").fillna("")
        df = df[df["category_bucket"].isin(config.categories)].reset_index(drop=True)
        if df.empty:
            raise ValueError("config.categories 에 해당하는 상품이 카탈로그에 없습니다.")
        self.df = df
        self.ids = df["article_id"].to_numpy()
        self.price = df["price"].to_numpy(dtype=float)
        self.newness = df["newness_value"].to_numpy(dtype=float)
        self.bucket = df["category_bucket"].to_numpy()
        self.rows = {pid: i for i, pid in enumerate(self.ids)}
        self._weights: dict[tuple[str, str | None], tuple[np.ndarray, np.ndarray]] = {}

    def candidates(self, persona: Persona, bucket: str | None) -> tuple[np.ndarray, np.ndarray]:
        """(row 배열, 확률) — 페르소나 가격 상한과 신상품 선호를 반영"""
        key = (persona.name, bucket)
        if key not in self._weights:
            mask = np.ones(len(self.ids), dtype=bool) if bucket is None else self.bucket == bucket
            if persona.price_max is not None:
                cheap = mask & (self.price <= persona.price_max)
                mask = cheap if cheap.any() else mask
            rows = np.flatnonzero(mask)
            weights = 1.0 + persona.new_item_bias * self.newness[rows]
            self._weights[key] = (rows, weights / weights.sum())
        return self._weights[key]

    def info(self, product_id: str) -> dict[str, str]:
        row = self.rows.get(product_id)
        if row is None:
            return {"category_bucket": "", "category_large": "", "category_middle": "",
                    "category_small": ""}
        r = self.df.iloc[row]
        return {k: str(r[k]) for k in ("category_bucket", "category_large", "category_middle",
                                       "category_small")}


# ------------------------------------------------------------------ API
class ApiClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        import requests

        self.base, self.timeout = base_url, timeout
        self.http = requests.Session()
        self.errors = 0

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.http.request(method, self.base + path, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - 시뮬레이터는 API 오류에도 계속 돈다
            self.errors += 1
            if self.errors <= 5 or self.errors % 100 == 0:
                _log(f"API 오류 #{self.errors} {method} {path}: {exc}")
            return None

    def wait_until_ready(self, timeout_s: float = 600) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = self._call("GET", "/api/status")
            if status and status.get("recommendation", {}).get("model_loaded"):
                return True
            time.sleep(3)
        return False

    def search(self, text: str) -> list[dict]:
        data = self._call("POST", "/api/search", json={"query_text": text, "top_k": 10})
        return (data or {}).get("results", [])

    def recommend(self, user_id: str) -> list[dict]:
        data = self._call("GET", "/api/recommend", params={"user_id": user_id, "top_n": 10})
        return (data or {}).get("recommendations", [])

    def event(self, user_id: str, product_id: str, event_type: str) -> None:
        self._call("POST", "/api/events", json={"user_id": user_id, "product_id": product_id,
                                                "event_type": event_type})


# ------------------------------------------------------------------ 시뮬레이터
@dataclass
class SimUser:
    user_id: str
    persona: Persona
    preferred: str
    sessions: int = 0


class Simulator:
    def __init__(self, config: SimulatorConfig, api: ApiClient | None = None,
                 catalog: Catalog | None = None) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.np_rng = np.random.default_rng(config.random_seed)
        self.catalog = catalog or Catalog(config)
        self.api = api if api is not None else (
            ApiClient(config.api_base_url) if config.send_to_api else None)
        personas = list(config.personas)
        buckets = list(config.categories)
        self.users = [
            SimUser(f"{config.user_id_prefix}{i:05d}",
                    self.rng.choices(personas, weights=[p.ratio for p in personas])[0],
                    self.rng.choices(buckets, weights=list(config.categories.values()))[0])
            for i in range(1, config.users + 1)
        ]
        self.events = 0
        self.counts: dict[str, int] = {}
        self._writer: Any = None
        self._file: Any = None

    # ---------------------------------------------------------- 로그
    def _open_log(self) -> None:
        path = self.config.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.is_file() or path.stat().st_size == 0
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=LOG_COLUMNS)
        if is_new:
            self._writer.writeheader()

    def _write(self, user: SimUser, session_id: str, event_type: str, product_id: str = "",
               search_query: str = "") -> None:
        self.events += 1
        self.counts[event_type] = self.counts.get(event_type, 0) + 1
        row = {"event_id": f"SIM{int(time.time() * 1000)}{self.events:08d}",
               "timestamp": datetime.now().isoformat(timespec="seconds"),
               "user_id": user.user_id, "session_id": session_id, "event_type": event_type,
               "product_id": product_id, "query_product_id": "", "search_query": search_query,
               "persona": user.persona.name, "preferred_category": user.preferred,
               **self.catalog.info(product_id)}
        self._writer.writerow(row)
        self._file.flush()
        self._throttle()

    def _throttle(self) -> None:
        if self.config.events_per_second > 0:
            time.sleep(1.0 / self.config.events_per_second)

    def _done(self) -> bool:
        return self.config.max_events is not None and self.events >= self.config.max_events

    # ---------------------------------------------------------- 행동
    def pick_from_catalog(self, user: SimUser) -> str:
        loyal = self.rng.random() < user.persona.category_loyalty
        rows, probs = self.catalog.candidates(user.persona, user.preferred if loyal else None)
        return str(self.catalog.ids[self.np_rng.choice(rows, p=probs)])

    def pick_from_recommendations(self, user: SimUser, recs: list[dict]) -> str | None:
        explore = [r for r in recs if r.get("is_exploration")]
        if explore and self.rng.random() < user.persona.explore_click:
            return str(self.rng.choice(explore)["product_id"])
        regular = [r for r in recs if not r.get("is_exploration")]
        liked = [r for r in regular
                 if self.catalog.info(str(r["product_id"]))["category_bucket"] == user.preferred]
        pool = liked or regular
        return str(self.rng.choice(pool[:5])["product_id"]) if pool else None

    def run_session(self, user: SimUser) -> None:
        user.sessions += 1
        session_id = f"{user.user_id}-S{user.sessions:05d}"
        cfg, api = self.config, self.api
        steps = self.rng.randint(cfg.session_min, cfg.session_max)
        queue: list[str] = []

        if self.rng.random() < cfg.search_probability:
            seed_item = self.pick_from_catalog(user)
            query = self.catalog.info(seed_item)["category_small"] or user.preferred
            self._write(user, session_id, "search", search_query=query)
            if api is not None:
                queue = [str(r["product_id"]) for r in api.search(query)[:3]]

        for _ in range(steps):
            if self._done():
                return
            product_id, from_rec = None, False
            if queue:
                product_id = queue.pop(0)
            elif api is not None and self.rng.random() < cfg.recommend_probability:
                product_id = self.pick_from_recommendations(user, api.recommend(user.user_id))
                from_rec = product_id is not None
            product_id = product_id or self.pick_from_catalog(user)

            self._write(user, session_id, "view", product_id)
            if api is not None:
                api.event(user.user_id, product_id, "click" if from_rec else "view")
            if self.rng.random() < user.persona.cart:
                self._write(user, session_id, "cart", product_id)
                if api is not None:
                    api.event(user.user_id, product_id, "cart")
                if self.rng.random() < user.persona.purchase:
                    self._write(user, session_id, "purchase", product_id)
                    if api is not None:
                        api.event(user.user_id, product_id, "purchase")

    def run(self) -> dict[str, Any]:
        cfg = self.config
        mix = {p.name: round(sum(u.persona is p for u in self.users) / len(self.users), 3)
               for p in cfg.personas}
        _log(f"사용자 {len(self.users)}명, 상품 {len(self.catalog.ids):,}개, 페르소나 {mix}")
        if self.api is not None:
            _log(f"API 대기 중: {cfg.api_base_url}")
            if not self.api.wait_until_ready():
                _log("API가 준비되지 않아 CSV 로그만 생성합니다.")
                self.api = None
        self._open_log()
        _log(f"행동 로그 생성 시작... → {cfg.log_path}")
        started, next_report = time.time(), 500
        try:
            while not self._done():
                self.run_session(self.rng.choice(self.users))
                if self.events >= next_report:
                    _log(f"누적 이벤트 {self.events:,}건 {self.counts}")
                    next_report += 500
        except KeyboardInterrupt:
            _log("중지 요청")
        finally:
            self._file.close()
        summary = {"events": self.events, "counts": self.counts,
                   "api_errors": self.api.errors if self.api else None,
                   "seconds": round(time.time() - started, 1)}
        _log(f"종료: {summary}")
        return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RE:SEARCH 행동 로그 시뮬레이터")
    parser.add_argument("--config", type=Path, default=Path("config/simulator.yaml"))
    parser.add_argument("--max-events", type=int, default=None, help="설정값 대신 사용")
    parser.add_argument("--no-api", action="store_true", help="API 호출 없이 CSV만 생성")
    args = parser.parse_args()
    config = SimulatorConfig.from_yaml(args.config)
    overrides: dict[str, Any] = {}
    if args.max_events is not None:
        overrides["max_events"] = args.max_events
    if args.no_api:
        overrides["send_to_api"] = False
    if overrides:
        from dataclasses import replace

        config = replace(config, **overrides)
    Simulator(config).run()
