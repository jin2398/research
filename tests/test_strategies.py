"""추천 정렬 전략 교체 (전략 패턴) 테스트: 코드 수정 없이 config/환경변수로 바뀌는지"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.recommendation import strategies


def fake_service():
    return SimpleNamespace(
        items=SimpleNamespace(ids=np.array(["A", "B", "C", "D"], dtype=object)),
        popularity=np.array([1.0, 50.0, 10.0, 0.0]),
        item_newness=lambda: np.array([0.1, 0.2, 0.99, 0.5]),
    )


ROWS = np.array([0, 1, 2, 3])
TWO_TOWER = np.array([0.9, 0.1, 0.5, 0.3])


@pytest.mark.parametrize("name, expected_first", [
    ("similarity", "A"),     # Two-Tower 점수 최고
    ("newest", "C"),         # newness 최고
    ("popularity", "B"),     # 인기도 최고
])
def test_strategies_change_order(name, expected_first):
    ranked = strategies.get_strategy(name).rank(fake_service(), None, ROWS, TWO_TOWER, 0.0)
    assert ranked[0][0] == expected_first
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_strategy_selected_from_config_and_env(monkeypatch):
    monkeypatch.delenv(strategies.ENV_VAR, raising=False)
    assert strategies.resolve_name({}) == "deepfm"
    assert strategies.resolve_name({"serving": {"ranking_strategy": "newest"}}) == "newest"
    monkeypatch.setenv(strategies.ENV_VAR, "popularity")
    assert strategies.resolve_name({"serving": {"ranking_strategy": "newest"}}) == "popularity"


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        strategies.get_strategy("does_not_exist")
