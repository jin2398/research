"""Stage 3 rules on synthetic data. Run from the project root: python -m pytest tests/test_rerank.py"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.recommendation.bandit import CategoryBandit
from src.recommendation.popularity import Event, PopularityIndex
from src.recommendation.rerank import (DEMO_CATEGORIES, RerankConfig, make_synthetic,
                                       max_category_run, skewed_candidates)

CONFIG = RerankConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / "rerank.yaml")


@pytest.fixture()
def setup():
    return make_synthetic(CONFIG)


def categories(reranker, result):
    return [reranker.category(r.product_id) for r in result.recommendations]


@pytest.mark.parametrize("top_n", [10, 50])
def test_no_three_same_category_in_a_row(setup, top_n):
    reranker, items = setup
    for seed, favored in enumerate(DEMO_CATEGORIES):
        result = reranker.rerank(f"U{seed}", skewed_candidates(items, favored, seed),
                                 history_count=30, top_n=top_n)
        assert len(result.recommendations) == top_n
        assert max_category_run(categories(reranker, result)) <= CONFIG.max_same_category_run


def test_exploration_slots_marked(setup):
    reranker, items = setup
    result = reranker.rerank("U1", skewed_candidates(items, "Coat", 1), history_count=30)
    explore = [r for r in result.recommendations if r.is_exploration]
    assert len(explore) == CONFIG.exploration_slots
    assert all(r.reason == "mab_exploration" for r in explore)


def test_new_arrival_within_seven_days(setup):
    reranker, items = setup
    result = reranker.rerank("U1", skewed_candidates(items, "Bag", 2), history_count=30)
    new = [r for r in result.recommendations if r.reason == "new_arrival"]
    assert len(new) == CONFIG.new_item_slots
    for rec in new:
        assert reranker.now - items[rec.product_id].created_at <= timedelta(days=CONFIG.new_item_days)


def test_cold_start_ignores_stage2(setup):
    reranker, items = setup
    result = reranker.rerank("U_NEW", skewed_candidates(items, "Hoodie", 3), history_count=2)
    assert result.cold_start
    reasons = {r.reason for r in result.recommendations}
    assert "ranking" not in reasons and "session_interest" not in reasons
    assert reasons & {"popular", "trending"}


def test_session_interest_reflected(setup):
    reranker, items = setup
    clicks = [pid for pid in sorted(items) if items[pid].category == "Dress"][:4]
    with_session = reranker.rerank("U1", skewed_candidates(items, "Hoodie", 4), history_count=30,
                                   recent_clicks=clicks, top_n=20)
    without = reranker.rerank("U1", skewed_candidates(items, "Hoodie", 4), history_count=30, top_n=20)
    assert with_session.session_context["session_interest"] in {"Dress", "원피스"}
    assert with_session.session_context["recent_clicks"] == clicks
    count = lambda res: categories(reranker, res).count("Dress")
    assert count(with_session) > count(without)


def test_api_fields(setup):
    reranker, items = setup
    payload = reranker.rerank("U1", skewed_candidates(items, "Skirt", 5), history_count=30).to_dict()
    assert {"user_id", "recommendations", "session_context", "reranking_ms"} <= payload.keys()
    assert {"recent_clicks", "session_interest"} <= payload["session_context"].keys()
    for rec in payload["recommendations"]:
        assert {"product_id", "score", "reason", "is_exploration"} <= rec.keys()


def test_deterministic_with_seed():
    outputs = []
    for _ in range(2):
        reranker, items = make_synthetic(CONFIG)
        result = reranker.rerank("U1", skewed_candidates(items, "Dress", 6), history_count=30)
        outputs.append([r.product_id for r in result.recommendations])
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("algorithm", ["epsilon_greedy", "ucb"])
def test_bandit_prefers_rewarded_arm(algorithm):
    bandit = CategoryBandit(["A", "B", "C"], algorithm, epsilon=0.0, ucb_c=0.1)
    for _ in range(50):
        bandit.update("A", 1.0)
        bandit.update("B", 0.0)
        bandit.update("C", 0.0)
    assert bandit.select(1) == ["A"]
    assert CategoryBandit.from_dict(bandit.to_dict()).mean("A") == 1.0


def test_popularity_ignores_future_events():
    now = datetime(2024, 1, 31)
    events = [Event("U", "past", "view", now - timedelta(days=1)),
              Event("U", "future", "purchase", now + timedelta(days=1))]
    index = PopularityIndex.from_events(events, CONFIG.event_weights, now, 7)
    assert "future" not in index.scores


def test_ucb_config_runs(setup):
    _, items = setup
    reranker, _ = make_synthetic(replace(CONFIG, bandit_algorithm="ucb"))
    result = reranker.rerank("U1", skewed_candidates(items, "Coat", 7), history_count=30)
    assert any(r.is_exploration for r in result.recommendations)


@pytest.mark.parametrize("norm", ["minmax", "rank"])
def test_score_normalization(setup, norm):
    reranker, _ = setup
    reranker.config = replace(CONFIG, score_normalization=norm)
    scores = {"A": 0.9, "B": 0.01, "C": 0.005, "D": 0.0}        # skewed like a CTR head
    base = reranker._normalize(scores)
    assert base["A"] == 1.0 and list(sorted(base, key=base.get, reverse=True)) == list(scores)
    if norm == "rank":
        assert base == {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}   # evenly spread
    else:
        assert base["B"] < 0.02                                    # squeezed near 0


def test_unknown_score_normalization_rejected(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "config" / "rerank.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "rerank.yaml"
    bad.write_text(text.replace("score_normalization: rank", "score_normalization: zscore"),
                   encoding="utf-8")
    with pytest.raises(ValueError):
        RerankConfig.from_yaml(bad)
