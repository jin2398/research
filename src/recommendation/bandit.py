"""Category-level multi-armed bandit for the Stage 3 exploration slots.

Each arm is a product category. A slot chosen by the bandit shows one item of
that category; the reward is 1 when the user clicks it and 0 otherwise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any


class CategoryBandit:
    ALGORITHMS = ("epsilon_greedy", "ucb")

    def __init__(self, arms: Iterable[str], algorithm: str = "epsilon_greedy",
                 epsilon: float = 0.1, ucb_c: float = 1.0, seed: int = 42) -> None:
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"algorithm must be one of {self.ALGORITHMS}")
        if not 0 <= epsilon <= 1 or ucb_c < 0:
            raise ValueError("epsilon must be in [0, 1] and ucb_c non-negative")
        self.arms = sorted(set(arms))
        if not self.arms:
            raise ValueError("At least one arm is required")
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.pulls = {arm: 0 for arm in self.arms}
        self.rewards = {arm: 0.0 for arm in self.arms}
        self.rng = random.Random(seed)

    def mean(self, arm: str) -> float:
        pulls = self.pulls.get(arm, 0)
        return self.rewards.get(arm, 0.0) / pulls if pulls else 0.0

    def _value(self, arm: str, total_pulls: int) -> float:
        if self.algorithm == "ucb":
            if not self.pulls[arm]:
                return math.inf          # every arm is tried once first
            bonus = self.ucb_c * math.sqrt(math.log(total_pulls) / self.pulls[arm])
            return self.mean(arm) + bonus
        return self.mean(arm)

    def select(self, k: int, exclude: Iterable[str] = ()) -> list[str]:
        """Choose up to k distinct arms; excluded arms are used only if nothing else is left."""
        blocked = set(exclude)
        pool = [arm for arm in self.arms if arm not in blocked] or list(self.arms)
        total = max(1, sum(self.pulls.values()))
        chosen: list[str] = []
        for _ in range(min(k, len(pool))):
            options = [arm for arm in pool if arm not in chosen]
            if self.algorithm == "epsilon_greedy" and self.rng.random() < self.epsilon:
                chosen.append(self.rng.choice(options))
                continue
            values = {arm: self._value(arm, total) for arm in options}
            best = max(values.values())
            chosen.append(self.rng.choice([arm for arm, v in values.items() if v == best]))
        return chosen

    def update(self, arm: str, reward: float) -> None:
        if not 0 <= reward <= 1:
            raise ValueError("reward must be in [0, 1]")
        if arm not in self.pulls:
            self.arms.append(arm)
            self.arms.sort()
            self.pulls[arm], self.rewards[arm] = 0, 0.0
        self.pulls[arm] += 1
        self.rewards[arm] += reward

    def to_dict(self) -> dict[str, Any]:
        """State for saving between API restarts (e.g. JSON file or Redis hash)."""
        return {"algorithm": self.algorithm, "epsilon": self.epsilon, "ucb_c": self.ucb_c,
                "pulls": dict(self.pulls), "rewards": dict(self.rewards)}

    @classmethod
    def from_dict(cls, state: dict[str, Any], seed: int = 42) -> "CategoryBandit":
        bandit = cls(state["pulls"].keys(), state["algorithm"], state["epsilon"],
                     state["ucb_c"], seed)
        bandit.pulls.update({k: int(v) for k, v in state["pulls"].items()})
        bandit.rewards.update({k: float(v) for k, v in state["rewards"].items()})
        return bandit
