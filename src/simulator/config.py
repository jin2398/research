"""시뮬레이터 설정 로딩·검증 (config/simulator.yaml)"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Persona:
    name: str
    ratio: float
    category_loyalty: float
    price_max: float | None
    new_item_bias: float
    explore_click: float
    cart: float
    purchase: float


@dataclass(frozen=True)
class SimulatorConfig:
    api_base_url: str
    send_to_api: bool
    catalog_path: Path
    products_path: Path
    log_path: Path
    users: int
    user_id_prefix: str
    events_per_second: float
    max_events: int | None
    random_seed: int
    session_min: int
    session_max: int
    search_probability: float
    recommend_probability: float
    categories: dict[str, float]
    personas: tuple[Persona, ...]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SimulatorConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        session = raw["session"]
        personas = tuple(Persona(name=str(name), **{k: v for k, v in values.items()})
                         for name, values in raw["personas"].items())
        config = cls(
            api_base_url=(os.getenv("SIMULATOR_API_URL") or raw["api_base_url"]).rstrip("/"),
            send_to_api=bool(raw.get("send_to_api", True)),
            catalog_path=Path(raw["data"]["catalog_path"]),
            products_path=Path(raw["data"]["products_path"]),
            log_path=Path(raw["data"]["log_path"]),
            users=int(raw["users"]),
            user_id_prefix=str(raw["user_id_prefix"]),
            events_per_second=float(raw["events_per_second"]),
            max_events=None if raw.get("max_events") is None else int(raw["max_events"]),
            random_seed=int(raw["random_seed"]),
            session_min=int(session["min_events"]),
            session_max=int(session["max_events"]),
            search_probability=float(session["search_probability"]),
            recommend_probability=float(session["recommend_probability"]),
            categories={str(k): float(v) for k, v in raw["categories"].items()},
            personas=personas,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.personas:
            raise ValueError("personas가 비어 있습니다.")
        total = sum(p.ratio for p in self.personas)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"페르소나 ratio 합계가 1이 아닙니다: {total:.3f}")
        if not self.categories or min(self.categories.values()) < 0:
            raise ValueError("categories는 비어 있지 않고 가중치가 0 이상이어야 합니다.")
        if not 1 <= self.session_min <= self.session_max:
            raise ValueError("session.min_events ≤ max_events 여야 합니다.")
        if self.users < 1:
            raise ValueError("users는 1 이상이어야 합니다.")
        for p in self.personas:
            for field in ("category_loyalty", "explore_click", "cart", "purchase"):
                if not 0 <= getattr(p, field) <= 1:
                    raise ValueError(f"{p.name}.{field}는 0~1 사이여야 합니다.")
