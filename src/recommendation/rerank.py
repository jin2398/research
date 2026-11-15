"""Stage 3 re-ranking: diversity, cold start, new arrivals, MAB slots, session interest.

Input is the Stage 2 ranking ``[(product_id, score), ...]`` plus the user's
session state from the feature store. Output follows the recommendation API
fields: ``recommendations[]`` (product_id, score, reason, is_exploration),
``session_context`` and ``reranking_ms``.

Smoke run on synthetic data (no other team module needed)::

    python -m src.recommendation.rerank demo --config config/rerank.yaml
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .bandit import CategoryBandit
from .popularity import Event, PopularityIndex, parse_time

try:  # Korean display names from the search module, if present
    from ..search.query_text import CATEGORIES as _CATEGORY_NAMES
except ImportError:  # pragma: no cover - search module not copied in
    _CATEGORY_NAMES = {}


@dataclass(frozen=True, slots=True)
class RerankConfig:
    products_path: Path
    catalog_path: Path | None
    catalog_id_column: str
    events_path: Path
    train_ratio: float | None
    clip_artifacts_dir: Path | None
    product_id_column: str
    event_product_id_column: str
    category_column: str
    created_at_column: str
    user_id_column: str
    event_type_column: str
    timestamp_column: str
    event_weights: dict[str, float]
    max_same_category_run: int
    cold_start_min_events: int
    trending_share: float
    trending_window_days: int
    trending_min_recent: float
    new_item_days: int
    new_item_slots: int
    new_item_positions: tuple[int, ...]
    created_at_fallback: str | None
    exploration_slots: int
    exploration_positions: tuple[int, ...]
    exploration_pool_size: int
    bandit_algorithm: str
    epsilon: float
    ucb_c: float
    bandit_store: dict[str, Any]
    session_weight: float
    session_reason_threshold: float
    session_window: int
    exclude_recent_clicks: bool
    recent_click_display: int
    encoder_config: Path | None
    random_seed: int
    score_normalization: str = "minmax"   # minmax | rank (how Stage 2 scores map to [0, 1])

    @classmethod
    def from_yaml(cls, path: Path) -> "RerankConfig":
        import yaml

        v = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(v, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        data, cols = v["data"], v["columns"]
        cold, new, explore, session = v["cold_start"], v["new_items"], v["exploration"], v["session"]
        clip_dir = data.get("clip_artifacts_dir")
        train_ratio = data.get("train_ratio")
        encoder_config = session.get("encoder_config")
        config = cls(
            products_path=Path(data["products_path"]),
            catalog_path=Path(data["catalog_path"]) if data.get("catalog_path") else None,
            catalog_id_column=str(cols.get("catalog_id", cols["product_id"])),
            events_path=Path(data["events_path"]),
            train_ratio=float(train_ratio) if train_ratio is not None else None,
            clip_artifacts_dir=Path(clip_dir) if clip_dir else None,
            product_id_column=str(cols["product_id"]),
            event_product_id_column=str(cols.get("event_product_id", cols["product_id"])),
            category_column=str(cols["category"]),
            created_at_column=str(cols["created_at"]),
            user_id_column=str(cols["user_id"]),
            event_type_column=str(cols["event_type"]),
            timestamp_column=str(cols["timestamp"]),
            event_weights={str(k): float(w) for k, w in v["event_weights"].items()},
            max_same_category_run=int(v["diversity"]["max_same_category_run"]),
            cold_start_min_events=int(cold["min_events"]),
            trending_share=float(cold["trending_share"]),
            trending_window_days=int(cold["trending_window_days"]),
            trending_min_recent=float(cold["trending_min_recent"]),
            new_item_days=int(new["days"]),
            new_item_slots=int(new["slots"]),
            new_item_positions=tuple(int(p) for p in new["positions"]),
            created_at_fallback=new.get("created_at_fallback"),
            exploration_slots=int(explore["slots"]),
            exploration_positions=tuple(int(p) for p in explore["positions"]),
            exploration_pool_size=int(explore["pool_size"]),
            bandit_algorithm=str(explore["algorithm"]),
            epsilon=float(explore["epsilon"]),
            ucb_c=float(explore["ucb_c"]),
            bandit_store=dict(explore.get("store") or {"backend": "none"}),
            session_weight=float(session["weight"]),
            session_reason_threshold=float(session["reason_threshold"]),
            session_window=int(session["window"]),
            exclude_recent_clicks=bool(session["exclude_recent_clicks"]),
            recent_click_display=int(session["recent_click_display"]),
            encoder_config=Path(encoder_config) if encoder_config else None,
            random_seed=int(v["random_seed"]),
            score_normalization=str(session.get("score_normalization", "minmax")),
        )
        if config.score_normalization not in ("minmax", "rank"):
            raise ValueError("score_normalization must be minmax or rank")
        if config.max_same_category_run < 1:
            raise ValueError("max_same_category_run must be at least 1")
        if not 1 <= config.exploration_slots <= 2:
            raise ValueError("exploration slots must be 1 or 2 (mission requirement)")
        if (len(config.exploration_positions) < config.exploration_slots
                or len(config.new_item_positions) < config.new_item_slots):
            raise ValueError("Each slot needs a configured position")
        if config.train_ratio is not None and not 0 < config.train_ratio <= 1:
            raise ValueError("train_ratio must be in (0, 1]")
        if config.created_at_fallback not in (None, "first_seen"):
            raise ValueError("created_at_fallback must be first_seen or null")
        if not 0 <= config.session_weight <= 1 or not 0 <= config.trending_share <= 1:
            raise ValueError("session weight and trending share must be in [0, 1]")
        return config


@dataclass(frozen=True, slots=True)
class ItemMeta:
    product_id: str
    category: str
    created_at: datetime | None


@dataclass(frozen=True, slots=True)
class Recommendation:
    product_id: str
    score: float
    reason: str
    is_exploration: bool


@dataclass(frozen=True, slots=True)
class RerankResult:
    user_id: str
    recommendations: list[Recommendation]
    session_context: dict[str, Any]
    reranking_ms: float
    cold_start: bool

    def to_dict(self) -> dict[str, Any]:
        return {"user_id": self.user_id,
                "recommendations": [asdict(r) for r in self.recommendations],
                "session_context": self.session_context,
                "reranking_ms": self.reranking_ms,
                "cold_start": self.cold_start}


def category_display_name(category: str) -> str:
    return _CATEGORY_NAMES.get(category, (category,))[0]


def max_category_run(categories: Sequence[str]) -> int:
    """Longest run of the same category; used by tests and peer-review checks."""
    longest = current = 0
    for i, category in enumerate(categories):
        current = current + 1 if i and category == categories[i - 1] else 1
        longest = max(longest, current)
    return longest


# ---------------------------------------------------------------- loading
def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix == ".jsonl":
            yield from (json.loads(line) for line in handle if line.strip())
        else:
            yield from csv.DictReader(handle)


def load_items(config: RerankConfig) -> dict[str, ItemMeta]:
    """Products (optionally limited to the ids in catalog_path) with category and date."""
    allowed = ({str(row[config.catalog_id_column]) for row in _rows(config.catalog_path)}
               if config.catalog_path else None)
    items = {}
    for row in _rows(config.products_path):
        pid = str(row[config.product_id_column])
        if allowed is not None and pid not in allowed:
            continue
        created = row.get(config.created_at_column)
        items[pid] = ItemMeta(pid, str(row[config.category_column]),
                              parse_time(str(created)) if created else None)
    return items


def load_all_events(config: RerankConfig) -> list[Event]:
    """Every log row, time-sorted (stable). Search rows keep an empty product_id."""
    col = config.event_product_id_column
    events = [Event(str(row[config.user_id_column]), str(row.get(col) or ""),
                    str(row[config.event_type_column]), parse_time(str(row[config.timestamp_column])))
              for row in _rows(config.events_path)]
    events.sort(key=lambda e: e.timestamp)
    return events


def split_events(events: Sequence[Event], train: float = 0.8,
                 valid: float = 0.1) -> tuple[list[Event], list[Event], list[Event]]:
    """Time-based train/valid/test split over all rows, as in docs/split_report.json."""
    a, b = int(len(events) * train), int(len(events) * (train + valid))
    return list(events[:a]), list(events[a:b]), list(events[b:])


def load_events(config: RerankConfig) -> list[Event]:
    """Product events of the train part (first `train_ratio` of all rows)."""
    events = load_all_events(config)
    if config.train_ratio is not None:
        events = events[:int(len(events) * config.train_ratio)]
    events = [e for e in events if e.product_id]
    if not events:
        raise ValueError(f"No product events in {config.events_path}")
    return events


def with_first_seen(items: Mapping[str, ItemMeta], events: Iterable[Event]) -> dict[str, ItemMeta]:
    """Fill missing created_at with the item's first event time (simulation clock)."""
    first: dict[str, datetime] = {}
    for event in events:
        if event.product_id not in first or event.timestamp < first[event.product_id]:
            first[event.product_id] = event.timestamp
    return {pid: (item if item.created_at or pid not in first
                  else ItemMeta(pid, item.category, first[pid]))
            for pid, item in items.items()}


VECTOR_CACHE_DIR = Path("artifacts/recommendation/clip_vectors")


def make_bandit_store(config: RerankConfig) -> Any:
    """BanditStore from the exploration.store section, or None for backend none."""
    from .bandit_store import BanditStore

    s = config.bandit_store
    backend = str(s.get("backend", "none"))
    if backend == "none":
        return None
    return BanditStore(backend, Path(s["path"]) if s.get("path") else None,
                       s.get("redis_url"), str(s.get("key", "stage3:bandit")))


def export_clip_vectors(folder: Path, cache_dir: Path = VECTOR_CACHE_DIR) -> Path:
    """Copy the normalized CLIP image vectors out of the search team's HNSWFlat index
    into a .npy cache (faiss only, no torch in this process)."""
    import faiss
    import numpy as np

    index = faiss.read_index(str(folder / "products.index"))
    with (folder / "products.jsonl").open(encoding="utf-8") as handle:
        ids = [json.loads(line)["product_id"] for line in handle]
    if len(ids) != index.ntotal:
        raise ValueError("Index and product metadata counts differ")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "vectors.npy", index.reconstruct_n(0, index.ntotal))
    (cache_dir / "meta.json").write_text(json.dumps(
        {"source": _source_stamp(folder), "product_ids": ids}), encoding="utf-8")
    return cache_dir


def _source_stamp(folder: Path) -> dict[str, Any]:
    stat = (folder / "products.index").stat()
    return {"folder": str(folder.resolve()), "size": stat.st_size, "mtime": stat.st_mtime}


def load_clip_vectors(folder: Path, cache_dir: Path = VECTOR_CACHE_DIR) -> tuple[Any, dict[str, int]]:
    """CLIP item vectors + {product_id: row}. Reads the .npy cache; builds it in a
    child process when missing or stale, because loading faiss and torch in one
    process aborts on macOS (duplicate libomp)."""
    import subprocess
    import sys

    import numpy as np

    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.is_file() else None
    if meta is None or meta.get("source") != _source_stamp(folder):
        print(f"[rerank] exporting CLIP vectors to {cache_dir} (one time)", flush=True)
        subprocess.run([sys.executable, "-m", "src.recommendation.rerank", "export-vectors",
                        "--clip-dir", str(folder), "--cache-dir", str(cache_dir)], check=True)
        meta = json.loads(meta_path.read_text("utf-8"))
    vectors = np.load(cache_dir / "vectors.npy")
    ids = meta["product_ids"]
    if len(ids) != len(vectors):
        raise ValueError(f"Stale vector cache in {cache_dir}; delete it and retry")
    return np.ascontiguousarray(vectors, dtype="float32"), {pid: row for row, pid in enumerate(ids)}


# ---------------------------------------------------------------- reranker
class Reranker:
    def __init__(self, config: RerankConfig, items: Mapping[str, ItemMeta],
                 popularity: PopularityIndex, bandit: CategoryBandit, now: datetime,
                 vectors: Any = None, vector_rows: Mapping[str, int] | None = None,
                 session_encoder: Any = None, bandit_store: Any = None) -> None:
        self.config = config
        self.session_encoder = session_encoder
        self.bandit_store = bandit_store
        self.items = dict(items)
        self.popularity = popularity
        self.bandit = bandit
        self.now = now
        self.vectors = vectors
        self.vector_rows = dict(vector_rows or {})
        self.rng = random.Random(config.random_seed)
        by_category: dict[str, list[str]] = {}
        for pid in sorted(self.items, key=lambda p: (-popularity.normalized(p), p)):
            by_category.setdefault(self.items[pid].category, []).append(pid)
        self.by_category = by_category
        dated = sorted((item.created_at, item.product_id)
                       for item in self.items.values() if item.created_at)
        self._created_times = [created for created, _ in dated]
        self._created_ids = [pid for _, pid in dated]

    @classmethod
    def from_config(cls, config: RerankConfig, now: datetime | None = None,
                    events: Sequence[Event] | None = None) -> "Reranker":
        """events: pre-loaded train events (saves re-reading the log in evaluation)."""
        items = load_items(config)
        events = list(events) if events is not None else load_events(config)
        now = now or max(event.timestamp for event in events)
        if config.created_at_fallback == "first_seen":
            items = with_first_seen(items, (e for e in events if e.timestamp <= now))
        if not any(item.created_at for item in items.values()):
            print(f"[rerank] warning: no '{config.created_at_column}' values; "
                  "new-arrival slots stay empty", flush=True)
        popularity = PopularityIndex.from_events(events, config.event_weights, now,
                                                 config.trending_window_days,
                                                 config.trending_min_recent)
        bandit = CategoryBandit({item.category for item in items.values()},
                                config.bandit_algorithm, config.epsilon, config.ucb_c,
                                config.random_seed)
        store = make_bandit_store(config)
        if store is not None and store.load_into(bandit):
            print(f"[rerank] bandit state restored ({sum(bandit.pulls.values())} pulls)", flush=True)
        vectors, rows = (load_clip_vectors(config.clip_artifacts_dir)
                         if config.clip_artifacts_dir else (None, None))
        encoder = None
        if config.encoder_config and vectors is not None:
            from .session_encoder import SessionEncoder, SessionEncoderConfig
            enc_cfg = SessionEncoderConfig.from_yaml(config.encoder_config)
            if enc_cfg.model_path.is_file():
                encoder = SessionEncoder.load(enc_cfg.model_path, vectors, rows)
            else:
                print(f"[rerank] warning: {enc_cfg.model_path} not found; session vector = "
                      "weighted average of clicks (train it: python -m "
                      "src.recommendation.session_encoder train)", flush=True)
        return cls(config, items, popularity, bandit, now, vectors, rows, encoder, store)

    def category(self, product_id: str) -> str:
        item = self.items.get(product_id)
        return item.category if item else f"__unknown__:{product_id}"

    # ------------------------------------------------------------ public
    def rerank(self, user_id: str, candidates: Sequence[tuple[str, float]], *,
               history_count: int, recent_clicks: Sequence[str] = (),
               now: datetime | None = None, top_n: int = 10,
               session_vector: Any = None) -> RerankResult:
        """Build the final top_n list.

        candidates: Stage 2 output, any order. history_count: total user events
        (Redis or train log). recent_clicks: oldest -> newest. session_vector:
        optional output of the session encoder, in the same space as the item vectors.
        """
        started = time.perf_counter()
        if top_n < 1:
            raise ValueError("top_n must be positive")
        cfg = self.config
        now = now or self.now
        clicks = [pid for pid in recent_clicks if pid][-cfg.session_window:]
        interest = self._session_interest(clicks)
        session = self._session_vector(clicks, session_vector)
        blocked = set(clicks) if cfg.exclude_recent_clicks else set()
        cold = history_count < cfg.cold_start_min_events

        if cold or not candidates:
            ranked = self._fallback(top_n, blocked)
            stage2_scores: dict[str, float] = {}
        else:
            ranked = self._score_candidates(candidates, session, interest, blocked)
            stage2_scores = {r.product_id: r.score for r in ranked}

        budget = max(0, top_n - 1)            # always keep at least one ranked item
        n_explore = min(cfg.exploration_slots, budget)
        n_new = min(cfg.new_item_slots, budget - n_explore)
        head = ranked[:top_n]
        used = blocked | {r.product_id for r in head}
        new_items = self._new_arrivals(now, n_new, used, session, interest, stage2_scores)
        used |= {r.product_id for r in new_items}
        explore = self._exploration(n_explore, {self.category(r.product_id) for r in head},
                                    used, session, interest, stage2_scores)

        taken: set[int] = set()
        reserved = ([(self._position(p, top_n, taken), r)
                     for p, r in zip(cfg.new_item_positions, new_items)]
                    + [(self._position(p, top_n, taken), r)
                       for p, r in zip(cfg.exploration_positions, explore)])
        final = self._assemble(ranked, reserved, top_n, blocked)
        context = {"recent_clicks": clicks[-cfg.recent_click_display:],
                   "session_interest": category_display_name(interest) if interest else None}
        return RerankResult(user_id, final, context,
                            round((time.perf_counter() - started) * 1000, 3), cold)

    def record_feedback(self, product_id: str, clicked: bool) -> None:
        """Exploration item clicked (reward 1) or ignored (reward 0), from the API or
        the simulator. Persisted when a bandit store is configured."""
        arm, reward = self.category(product_id), 1.0 if clicked else 0.0
        if self.bandit_store is not None:
            self.bandit_store.record(self.bandit, arm, reward)
        else:
            self.bandit.update(arm, reward)

    # ------------------------------------------------------------ session
    def _session_interest(self, clicks: Sequence[str]) -> str | None:
        weights: Counter[str] = Counter()
        for position, pid in enumerate(clicks, 1):   # later clicks weigh more
            if pid in self.items:
                weights[self.items[pid].category] += position
        return weights.most_common(1)[0][0] if weights else None

    def _session_vector(self, clicks: Sequence[str], given: Any) -> Any:
        import numpy as np

        if given is not None:
            vector = np.asarray(given, dtype="float32").reshape(-1)
        elif self.session_encoder is not None:
            vector = self.session_encoder.encode(clicks)     # GRU, already normalized
            if vector is None:
                return None
        elif self.vectors is not None:
            rows = [(i, self.vector_rows[p]) for i, p in enumerate(clicks, 1) if p in self.vector_rows]
            if not rows:
                return None
            weights = np.array([i for i, _ in rows], dtype="float32")
            vector = (weights[:, None] * self.vectors[[r for _, r in rows]]).sum(axis=0)
        else:
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else None

    def _similarities(self, pids: Sequence[str], session: Any,
                      interest: str | None) -> dict[str, float | None]:
        """Session similarity in [0, 1]: vector cosine, else category match, else None."""
        result: dict[str, float | None] = {}
        with_vec = ([p for p in pids if p in self.vector_rows]
                    if session is not None and self.vectors is not None else [])
        if with_vec:
            cosines = self.vectors[[self.vector_rows[p] for p in with_vec]] @ session
            result.update((p, max(0.0, float(c))) for p, c in zip(with_vec, cosines))
        for pid in pids:
            if pid not in result:
                result[pid] = (None if interest is None
                               else float(self.category(pid) == interest))
        return result

    # ------------------------------------------------------------ lists
    def _score_candidates(self, candidates: Sequence[tuple[str, float]], session: Any,
                          interest: str | None, blocked: set[str]) -> list[Recommendation]:
        cfg = self.config
        unique: dict[str, float] = {}
        for pid, score in candidates:
            if pid not in blocked and pid not in unique:
                unique[pid] = float(score)
        if not unique:
            return []
        sims = self._similarities(list(unique), session, interest)
        bases = self._normalize(unique)
        ranked = []
        for pid in unique:
            base = bases[pid]
            sim = sims[pid]
            if sim is None:
                ranked.append(Recommendation(pid, round(base, 6), "ranking", False))
            else:
                blended = (1 - cfg.session_weight) * base + cfg.session_weight * sim
                reason = "session_interest" if sim >= cfg.session_reason_threshold else "ranking"
                ranked.append(Recommendation(pid, round(blended, 6), reason, False))
        ranked.sort(key=lambda r: -r.score)
        return ranked

    def _normalize(self, scores: Mapping[str, float]) -> dict[str, float]:
        """Stage 2 scores -> [0, 1]. minmax keeps score gaps (a skewed CTR head squeezes
        most items near 0); rank spreads them evenly by position (best = 1)."""
        if self.config.score_normalization == "rank":
            order = sorted(scores, key=lambda p: -scores[p])
            n = len(order)
            return {pid: 1.0 - i / n for i, pid in enumerate(order)}
        low, high = min(scores.values()), max(scores.values())
        return {pid: (s - low) / (high - low) if high > low else 1.0 for pid, s in scores.items()}

    def _fallback(self, top_n: int, blocked: set[str]) -> list[Recommendation]:
        pool = self.popularity.mixed(max(top_n * 5, 100), self.config.trending_share, blocked)
        return [Recommendation(pid, round(1 - i / len(pool), 6), reason, False)
                for i, (pid, reason) in enumerate(pool)]

    def _outside_score(self, pid: str, sim: float | None, stage2: Mapping[str, float]) -> float:
        if pid in stage2:
            return stage2[pid]
        return round(self.config.session_weight * (sim or 0.0), 6)

    def _new_arrivals(self, now: datetime, n: int, used: set[str], session: Any,
                      interest: str | None, stage2: Mapping[str, float]) -> list[Recommendation]:
        if n <= 0 or not self._created_times:
            return []
        lower = now - timedelta(days=self.config.new_item_days)
        start = bisect.bisect_left(self._created_times, lower)
        end = bisect.bisect_right(self._created_times, now)
        pool = [p for p in self._created_ids[start:end][-5000:] if p not in used]
        if not pool:
            return []
        sims = self._similarities(pool, session, interest)

        def key(pid: str) -> tuple[float, float, datetime]:
            return (stage2.get(pid, 0.0) + (sims[pid] or 0.0),
                    self.popularity.normalized(pid), self.items[pid].created_at)

        best = sorted(pool, key=key, reverse=True)[:n]
        return [Recommendation(p, self._outside_score(p, sims[p], stage2), "new_arrival", False)
                for p in best]

    def _exploration(self, n: int, head_categories: set[str], used: set[str], session: Any,
                     interest: str | None, stage2: Mapping[str, float]) -> list[Recommendation]:
        picks = []
        for arm in self.bandit.select(n, exclude=head_categories):
            pool = [p for p in self.by_category.get(arm, ())[:self.config.exploration_pool_size]
                    if p not in used]
            if pool:
                pid = self.rng.choice(pool)
                used.add(pid)
                sim = self._similarities([pid], session, interest)[pid]
                picks.append(Recommendation(pid, self._outside_score(pid, sim, stage2),
                                            "mab_exploration", True))
        return picks

    # ------------------------------------------------------------ assembly
    @staticmethod
    def _position(preferred: int, top_n: int, taken: set[int]) -> int:
        position = min(max(preferred, 0), top_n - 1)
        while position in taken and position > 0:
            position -= 1
        while position in taken:
            position += 1
        taken.add(position)
        return position

    def _allowed(self, chosen: Sequence[Recommendation], pid: str) -> bool:
        k = self.config.max_same_category_run
        if len(chosen) < k:
            return True
        category = self.category(pid)
        return not all(self.category(r.product_id) == category for r in chosen[-k:])

    def _assemble(self, ranked: list[Recommendation],
                  reserved: list[tuple[int, Recommendation]], top_n: int,
                  blocked: set[str]) -> list[Recommendation]:
        reserved_ids = {r.product_id for _, r in reserved}
        queue = [r for r in ranked if r.product_id not in reserved_ids]
        pending = sorted(reserved, key=lambda item: item[0])
        result: list[Recommendation] = []
        while len(result) < top_n:
            position = len(result)
            choice = None
            for i, (slot, rec) in enumerate(pending):          # due reserved slot
                if slot <= position and self._allowed(result, rec.product_id):
                    choice = pending.pop(i)[1]
                    break
            if choice is None:                                  # best allowed ranked item
                for i, rec in enumerate(queue):
                    if self._allowed(result, rec.product_id):
                        choice = queue.pop(i)
                        break
            if choice is None:                                  # leftover reserved items
                for i, (_, rec) in enumerate(pending):
                    if self._allowed(result, rec.product_id):
                        choice = pending.pop(i)[1]
                        break
            if choice is None:
                choice = self._diversity_filler(result, blocked | reserved_ids
                                                | {r.product_id for r in queue})
            if choice is None:
                break
            result.append(choice)
        return result

    def _diversity_filler(self, chosen: list[Recommendation],
                          excluded: set[str]) -> Recommendation | None:
        taken = excluded | {r.product_id for r in chosen}
        for pid in self.popularity.popular:
            if pid not in taken and self._allowed(chosen, pid):
                return Recommendation(pid, 0.0, "diversity_fill", False)
        return None


# ---------------------------------------------------------------- synthetic demo
DEMO_CATEGORIES = ("Hoodie", "T-shirt", "Dress", "Trousers", "Sneakers", "Bag", "Coat", "Skirt")


def make_synthetic(config: RerankConfig, n_items: int = 2000,
                   n_events: int = 20000) -> tuple[Reranker, dict[str, ItemMeta]]:
    """Fake catalog and train log, for developing before Stage 1/2 are ready."""
    rng = random.Random(config.random_seed)
    now = datetime(2024, 1, 31)
    items = {}
    for i in range(n_items):
        pid = f"P{i:05d}"
        age = rng.randint(0, 6) if i % 40 == 0 else rng.randint(8, 365)
        items[pid] = ItemMeta(pid, DEMO_CATEGORIES[i % len(DEMO_CATEGORIES)], now - timedelta(days=age))
    ids = list(items)
    events = [Event(f"U{rng.randint(0, 499):04d}", rng.choice(ids[: n_items // 2] if rng.random() < 0.7 else ids),
                    rng.choice(("view", "view", "view", "cart", "purchase")),
                    now - timedelta(days=rng.random() * 30))
              for _ in range(n_events)]
    popularity = PopularityIndex.from_events(events, config.event_weights, now,
                                             config.trending_window_days, config.trending_min_recent)
    bandit = CategoryBandit(DEMO_CATEGORIES, config.bandit_algorithm, config.epsilon,
                            config.ucb_c, config.random_seed)
    return Reranker(config, items, popularity, bandit, now), items


def skewed_candidates(items: Mapping[str, ItemMeta], favored: str, seed: int,
                      size: int = 300) -> list[tuple[str, float]]:
    """Mock Stage 2 output dominated by one category, to exercise the diversity rule."""
    rng = random.Random(seed)
    sample = rng.sample(sorted(items), size)
    return sorted(((pid, rng.random() + (1.0 if items[pid].category == favored else 0.0))
                   for pid in sample), key=lambda x: -x[1])


def _demo(config: RerankConfig) -> None:
    reranker, items = make_synthetic(config)
    candidates = skewed_candidates(items, "Hoodie", config.random_seed)
    clicks = [pid for pid in sorted(items) if items[pid].category == "Dress"][:4]
    warm = reranker.rerank("U0001", candidates, history_count=42, recent_clicks=clicks, top_n=10)
    cold = reranker.rerank("U_NEW", [], history_count=0, top_n=10)
    for result in (warm, cold):
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        cats = [reranker.category(r.product_id) for r in result.recommendations]
        print(f"categories: {cats}\nmax same-category run: {max_category_run(cats)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/rerank.yaml"))
    parser.add_argument("--clip-dir", type=Path, help="export-vectors: search index folder")
    parser.add_argument("--cache-dir", type=Path, default=VECTOR_CACHE_DIR)
    parser.add_argument("action", choices=["demo", "export-vectors"])
    args = parser.parse_args()
    if args.action == "export-vectors":
        folder = args.clip_dir or RerankConfig.from_yaml(args.config).clip_artifacts_dir
        print(export_clip_vectors(folder, args.cache_dir))
        return
    _demo(RerankConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
