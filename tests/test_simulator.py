"""시뮬레이터: config 파일만 바꿔서 페르소나 비율·카테고리 구조가 반영되는지"""

import csv
from pathlib import Path

import pytest
import yaml

from src.simulator.config import SimulatorConfig
from src.simulator.simulator import LOG_COLUMNS, Simulator

BASE = Path("config/simulator.yaml")


def write_config(tmp_path, **changes):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["send_to_api"] = False
    raw["events_per_second"] = 0
    raw["data"]["log_path"] = str(tmp_path / "log.csv")
    for key, value in changes.items():
        raw[key] = value
    path = tmp_path / "simulator.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_persona_ratio_and_categories_follow_config(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    personas = {"trendsetter": {**raw["personas"]["trendsetter"], "ratio": 1.0}}
    path = write_config(tmp_path, personas=personas, categories={"신발": 1.0}, users=50,
                        max_events=200)
    sim = Simulator(SimulatorConfig.from_yaml(path))
    assert {u.persona.name for u in sim.users} == {"trendsetter"}
    assert {u.preferred for u in sim.users} == {"신발"}
    assert set(sim.catalog.bucket) == {"신발"}             # 다른 카테고리 상품 제외

    summary = sim.run()
    assert summary["events"] == 200
    with open(tmp_path / "log.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == LOG_COLUMNS
    assert len(rows) == 200
    viewed = [r for r in rows if r["product_id"]]
    assert viewed and all(r["category_bucket"] == "신발" for r in viewed)


def test_log_is_appended_not_overwritten(tmp_path):
    path = write_config(tmp_path, max_events=30, users=5)
    Simulator(SimulatorConfig.from_yaml(path)).run()
    Simulator(SimulatorConfig.from_yaml(path)).run()
    with open(tmp_path / "log.csv", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 60      # CT 모니터용 append 로그


def test_invalid_persona_ratio_rejected(tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    personas = {name: {**p, "ratio": 0.5} for name, p in raw["personas"].items()}
    with pytest.raises(ValueError, match="ratio"):
        SimulatorConfig.from_yaml(write_config(tmp_path, personas=personas))
