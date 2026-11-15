"""Shared data layer for Stage 1 (Two-Tower) and Stage 2 (DeepFM).

- ``Stage12Config``  : typed view of ``config/stage12.yaml``
- ``ItemTable``      : catalog items -> integer feature codes (Item Tower / DeepFM input)
- ``EventLog``       : click events as numpy arrays + the time-based 8/1/1 split
- ``UserStates``     : per-user snapshot (profile, recent clicks, category/brand/price
                        aggregates) at a point in time. The same object feeds offline
                        evaluation and the online API, so features never diverge.

The split is the one in ``docs/split_report.json`` and ``rerank.split_events``: all
rows (search rows included) sorted by time (stable), first 80% train, next 10% valid,
last 10% test.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# faiss and torch each ship an OpenMP runtime; on Windows/macOS loading both in one
# process aborts unless faiss is imported first. Modules that use both import this one.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST = 0, 1, 2
UNKNOWN = 0          # code 0 of every categorical field = unknown / missing


# ---------------------------------------------------------------- config
@dataclass(frozen=True)
class Stage12Config:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Stage12Config":
        import yaml

        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        cfg = cls(raw, path)
        if not 0 < cfg.train_ratio < cfg.train_ratio + cfg.valid_ratio < 1:
            raise ValueError("split ratios must satisfy 0 < train < train+valid < 1")
        if cfg.faiss["index_type"] == "hnsw" and cfg.faiss["hnsw_ef_search"] < cfg.faiss["top_k"]:
            raise ValueError("faiss.hnsw_ef_search must be >= faiss.top_k")
        return cfg

    # sections
    @property
    def data(self) -> dict[str, Any]: return self.raw["data"]
    @property
    def columns(self) -> dict[str, str]: return self.raw["columns"]
    @property
    def two_tower(self) -> dict[str, Any]: return self.raw["two_tower"]
    @property
    def faiss(self) -> dict[str, Any]: return self.raw["faiss"]
    @property
    def deepfm(self) -> dict[str, Any]: return self.raw["deepfm"]
    @property
    def evaluation(self) -> dict[str, Any]: return self.raw["evaluation"]

    @property
    def train_ratio(self) -> float: return float(self.raw["split"]["train"])
    @property
    def valid_ratio(self) -> float: return float(self.raw["split"]["valid"])
    @property
    def seed(self) -> int: return int(self.raw["random_seed"])
    @property
    def click_events(self) -> tuple[str, ...]: return tuple(self.raw["click_events"])
    @property
    def event_weights(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.raw["event_weights"].items()}
    @property
    def output_dir(self) -> Path: return Path(self.raw["output"]["dir"])


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


# ---------------------------------------------------------------- items
@dataclass
class ItemTable:
    ids: np.ndarray                       # [n_items] product id strings
    fields: list[str]                     # feature field names, order of `codes` columns
    codes: np.ndarray                     # [n_items, n_fields] int64, 0 = unknown
    vocab: dict[str, list[str]]           # field -> values (index + 1 = code)
    edges: dict[str, list[float]]         # bucketed numeric field -> quantile edges
    price: np.ndarray                     # [n_items] float32 raw price
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {pid: i for i, pid in enumerate(self.ids)}

    @property
    def n_items(self) -> int: return len(self.ids)

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[f]) + 1 for f in self.fields]

    def column(self, name: str) -> np.ndarray:
        return self.codes[:, self.fields.index(name)]

    def lookup(self, product_ids: Iterable[str]) -> np.ndarray:
        """Product ids -> row indices (-1 if not in the catalog)."""
        return np.fromiter((self.index.get(str(p), -1) for p in product_ids), dtype=np.int64)

    @classmethod
    def build(cls, config: Stage12Config) -> "ItemTable":
        cols, data = config.columns, config.data
        item_col = cols["item_id"]
        catalog = pd.read_csv(data["catalog_path"], dtype={item_col: str}, encoding="utf-8-sig")
        products = pd.read_csv(data["products_path"], dtype=str, encoding="utf-8-sig")
        extra = [c for c in products.columns if c not in catalog.columns or c == item_col]
        table = catalog.merge(products[extra], on=item_col, how="left")
        table = table.drop_duplicates(item_col).sort_values(item_col).reset_index(drop=True)
        spec = config.raw["item_features"]
        vocab: dict[str, list[str]] = {}
        edges: dict[str, list[float]] = {}
        fields: list[str] = []
        columns = []
        for name in spec["categorical"]:
            values = table[name].fillna("").astype(str)
            vocab[name] = sorted(v for v in values.unique() if v)
            columns.append(_encode(values, vocab[name]))
            fields.append(name)
        for name, n_bins in spec["bucketed"].items():
            values = pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
            edges[name] = quantile_edges(values, int(n_bins))
            vocab[name] = [f"b{i}" for i in range(len(edges[name]) + 1)]
            codes = np.where(np.isnan(values), UNKNOWN,
                             np.searchsorted(edges[name], values, side="left") + 1)
            columns.append(codes.astype(np.int64))
            fields.append(name)
        price = pd.to_numeric(table["price"], errors="coerce").fillna(0).to_numpy(np.float32)
        return cls(table[item_col].to_numpy(dtype=object), fields,
                   np.stack(columns, axis=1).astype(np.int64), vocab, edges, price)

    # persistence: encoders.json holds everything needed to rebuild codes for serving
    def save(self, path: Path) -> None:
        payload = {"ids": [str(x) for x in self.ids], "fields": self.fields, "vocab": self.vocab,
                   "edges": self.edges, "codes": self.codes.tolist(), "price": self.price.tolist()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ItemTable":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(p["ids"], dtype=object), p["fields"],
                   np.array(p["codes"], dtype=np.int64), p["vocab"],
                   {k: list(v) for k, v in p["edges"].items()},
                   np.array(p["price"], dtype=np.float32))


def quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    """Bucket edges (use with searchsorted side="left": a value equal to an edge falls in
    the lower bucket). A zero-inflated column (e.g. discount_pct, 70% exactly 0) keeps
    its minimum as a bucket of its own and splits the rest into n_bins - 1 quantiles."""
    v = values[~np.isnan(values)]
    lo = float(v.min())
    if (v == lo).mean() >= 1.0 / n_bins:
        rest = v[v > lo]
        qs = np.quantile(rest, np.linspace(0, 1, n_bins)[1:-1]) if len(rest) else []
        return sorted({lo, *(float(q) for q in qs)})
    return sorted({float(q) for q in np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])})


def _encode(values: pd.Series, vocab: Sequence[str]) -> np.ndarray:
    lookup = {v: i + 1 for i, v in enumerate(vocab)}
    return values.map(lambda v: lookup.get(v, UNKNOWN)).to_numpy(dtype=np.int64)


# ---------------------------------------------------------------- events
@dataclass
class EventLog:
    """Click events (view/cart/purchase) of catalog items, in time order."""
    user: np.ndarray            # [n] int64 user index
    item: np.ndarray            # [n] int64 item row in ItemTable
    etype: np.ndarray           # [n] int64 index into event_types
    ts: np.ndarray              # [n] float64 unix seconds
    split: np.ndarray           # [n] int8 SPLIT_TRAIN / VALID / TEST
    user_ids: np.ndarray        # [n_users] user id strings
    persona: np.ndarray         # [n_users] int64 code (0 unknown)
    preferred: np.ndarray       # [n_users] int64 code (0 unknown)
    persona_vocab: list[str]
    preferred_vocab: list[str]
    event_types: list[str]
    split_times: dict[str, list[str]]
    rows: dict[str, int]
    dropped_not_in_catalog: int

    @property
    def n_users(self) -> int: return len(self.user_ids)

    def mask(self, *splits: int) -> np.ndarray:
        return np.isin(self.split, splits)

    @classmethod
    def load(cls, config: Stage12Config, items: ItemTable) -> "EventLog":
        c = config.columns
        usecols = [c["user_id"], c["event_type"], c["product_id"], c["timestamp"],
                   c["persona"], c["preferred_category"]]
        df = pd.read_csv(config.data["events_path"], usecols=usecols, dtype=str,
                         encoding="utf-8-sig")
        df["_t"] = pd.to_datetime(df[c["timestamp"]])
        df = df.sort_values("_t", kind="stable").reset_index(drop=True)
        n = len(df)
        a, b = int(n * config.train_ratio), int(n * (config.train_ratio + config.valid_ratio))
        split = np.full(n, SPLIT_TEST, dtype=np.int8)
        split[:a], split[a:b] = SPLIT_TRAIN, SPLIT_VALID
        times = df["_t"].astype(str).to_numpy()
        split_times = {"train": [times[0], times[a - 1]], "valid": [times[a], times[b - 1]],
                       "test": [times[b], times[-1]]}

        user_ids = np.array(sorted(df[c["user_id"]].unique()), dtype=object)
        uidx = {u: i for i, u in enumerate(user_ids)}
        prof = df.drop_duplicates(c["user_id"]).set_index(c["user_id"]).reindex(user_ids)
        persona_vocab = sorted(prof[c["persona"]].dropna().unique())
        preferred_vocab = sorted(prof[c["preferred_category"]].dropna().unique())
        persona = _encode(prof[c["persona"]].fillna(""), persona_vocab)
        preferred = _encode(prof[c["preferred_category"]].fillna(""), preferred_vocab)

        event_types = list(config.click_events)
        keep = df[c["event_type"]].isin(event_types).to_numpy() & df[c["product_id"]].notna().to_numpy()
        item = items.lookup(df.loc[keep, c["product_id"]])
        in_catalog = item >= 0
        sel = np.flatnonzero(keep)[in_catalog]
        ev = df.loc[sel]
        return cls(
            user=ev[c["user_id"]].map(uidx).to_numpy(np.int64),
            item=item[in_catalog],
            etype=ev[c["event_type"]].map({e: i for i, e in enumerate(event_types)}).to_numpy(np.int64),
            ts=(ev["_t"].astype("int64") // 10**9).to_numpy(np.float64),
            split=split[sel],
            user_ids=user_ids, persona=persona, preferred=preferred,
            persona_vocab=persona_vocab, preferred_vocab=preferred_vocab,
            event_types=event_types, split_times=split_times,
            rows={"total": n, "train": a, "valid": b - a, "test": n - b,
                  "click_events": int(len(sel))},
            dropped_not_in_catalog=int((~in_catalog).sum()))

    def truth(self, split: int, etypes: Sequence[str] | None = None) -> dict[int, np.ndarray]:
        """user index -> unique items with the given event types in `split`."""
        m = self.split == split
        if etypes is not None:
            m &= np.isin(self.etype, [self.event_types.index(e) for e in etypes])
        users, items = self.user[m], self.item[m]
        order = np.lexsort((items, users))
        users, items = users[order], items[order]
        out: dict[int, np.ndarray] = {}
        for u, start, stop in _groups(users):
            out[int(u)] = np.unique(items[start:stop])
        return out


def _groups(sorted_keys: np.ndarray) -> Iterable[tuple[Any, int, int]]:
    """(key, start, stop) for runs of equal keys in a sorted array."""
    if len(sorted_keys) == 0:
        return
    bounds = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate([[0], bounds])
    stops = np.concatenate([bounds, [len(sorted_keys)]])
    for s, e in zip(starts, stops):
        yield sorted_keys[s], int(s), int(e)


# ---------------------------------------------------------------- user snapshot
@dataclass
class UserContext:
    """Everything the towers and the ranker need about one user at request time."""
    user_index: int                 # -1 = unknown user
    persona: int
    preferred: int
    n_events: int
    recent_items: np.ndarray        # [<=H] item rows, oldest -> newest
    recent_etypes: np.ndarray       # [<=H]
    cat_share: np.ndarray           # [n_category_small + 1] weighted share of history
    brand_share: np.ndarray         # [n_brand + 1]
    bucket_share: np.ndarray        # [n_category_bucket + 1]
    mean_price: float               # weighted mean price of history (0 if none)
    seen: np.ndarray                # sorted unique item rows ever clicked
    last_ts: float                  # unix seconds of the last event (nan if none)


@dataclass
class UserStates:
    """Per-user aggregates of all click events up to a snapshot."""
    user_ids: np.ndarray
    persona: np.ndarray
    preferred: np.ndarray
    n_events: np.ndarray            # [U]
    recent: np.ndarray              # [U, H] item rows, padded with -1 on the left
    recent_etypes: np.ndarray       # [U, H]
    cat: np.ndarray                 # [U, n_cat] weighted counts
    brand: np.ndarray               # [U, n_brand]
    bucket: np.ndarray              # [U, n_bucket]
    price_sum: np.ndarray           # [U]
    weight_sum: np.ndarray          # [U]
    seen: list[np.ndarray]
    last_ts: np.ndarray             # [U]
    snapshot: str
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {str(u): i for i, u in enumerate(self.user_ids)}

    @classmethod
    def build(cls, log: EventLog, items: ItemTable, mask: np.ndarray, history: int,
              weights: dict[str, float], snapshot: str) -> "UserStates":
        u, it, et, ts = log.user[mask], log.item[mask], log.etype[mask], log.ts[mask]
        n_users = log.n_users
        w = np.array([weights.get(e, 1.0) for e in log.event_types], dtype=np.float32)[et]
        cat_col, brand_col = items.column("category_small"), items.column("brand")
        bucket_col = items.column("category_bucket")
        card = dict(zip(items.fields, items.cardinalities()))

        def agg(col: np.ndarray, size: int) -> np.ndarray:
            out = np.zeros((n_users, size), dtype=np.float32)
            np.add.at(out, (u, col[it]), w)
            return out

        price_sum = np.zeros(n_users, dtype=np.float64)
        np.add.at(price_sum, u, w * items.price[it])
        weight_sum = np.bincount(u, weights=w, minlength=n_users)
        n_events = np.bincount(u, minlength=n_users)
        recent = np.full((n_users, history), -1, dtype=np.int64)
        recent_et = np.zeros((n_users, history), dtype=np.int64)
        last_ts = np.full(n_users, np.nan)
        seen: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_users
        order = np.argsort(u, kind="stable")              # keeps time order inside a user
        for user, start, stop in _groups(u[order]):
            rows = order[start:stop]
            tail = rows[-history:]
            recent[user, history - len(tail):] = it[tail]
            recent_et[user, history - len(tail):] = et[tail]
            last_ts[user] = ts[rows[-1]]
            seen[user] = np.unique(it[rows])
        return cls(log.user_ids, log.persona, log.preferred, n_events, recent, recent_et,
                   agg(cat_col, card["category_small"]), agg(brand_col, card["brand"]),
                   agg(bucket_col, card["category_bucket"]), price_sum, weight_sum, seen,
                   last_ts, snapshot)

    def context(self, user_id: str, extra_items: Sequence[int] = (),
                extra_etypes: Sequence[int] | None = None) -> UserContext:
        """Snapshot of `user_id`, optionally extended with fresh session clicks
        (item rows, oldest -> newest) that are not in the snapshot yet."""
        i = self.index.get(str(user_id), -1)
        h = self.recent.shape[1]
        if i < 0:
            recent = np.empty(0, dtype=np.int64)
            recent_et = np.empty(0, dtype=np.int64)
            cat, brand, bucket = (np.zeros(a.shape[1], np.float32)
                                  for a in (self.cat, self.brand, self.bucket))
            price_sum = weight_sum = 0.0
            n, seen, last, persona, preferred = 0, np.empty(0, np.int64), np.nan, UNKNOWN, UNKNOWN
        else:
            keep = self.recent[i] >= 0
            recent, recent_et = self.recent[i][keep], self.recent_etypes[i][keep]
            cat, brand, bucket = self.cat[i].copy(), self.brand[i].copy(), self.bucket[i].copy()
            price_sum, weight_sum = float(self.price_sum[i]), float(self.weight_sum[i])
            n, seen, last = int(self.n_events[i]), self.seen[i], float(self.last_ts[i])
            persona, preferred = int(self.persona[i]), int(self.preferred[i])
        if len(extra_items):
            extra = np.asarray(extra_items, dtype=np.int64)
            ets = (np.asarray(extra_etypes, dtype=np.int64) if extra_etypes is not None
                   else np.zeros(len(extra), dtype=np.int64))
            recent = np.concatenate([recent, extra])[-h:]
            recent_et = np.concatenate([recent_et, ets])[-h:]
            seen = np.union1d(seen, extra)
            n += len(extra)
        total = lambda a: a / a.sum() if a.sum() > 0 else a  # noqa: E731
        return UserContext(i, persona, preferred, n, recent, recent_et, total(cat), total(brand),
                           total(bucket), price_sum / weight_sum if weight_sum else 0.0,
                           seen, last)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "UserStates":
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def item_popularity(log: EventLog, mask: np.ndarray, n_items: int,
                    weights: dict[str, float]) -> np.ndarray:
    """Weighted event count per item over the masked events (popularity baseline)."""
    w = np.array([weights.get(e, 1.0) for e in log.event_types], dtype=np.float64)
    return np.bincount(log.item[mask], weights=w[log.etype[mask]], minlength=n_items)


def load_or_build_items(config: Stage12Config) -> ItemTable:
    path = config.output_dir / "encoders.json"
    if path.is_file():
        return ItemTable.load(path)
    items = ItemTable.build(config)
    items.save(path)
    return items
