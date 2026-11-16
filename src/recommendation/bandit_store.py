"""Persistent state for the exploration bandit (pull and reward counts per category).

Backends
- json : one file, rewritten atomically. Local runs and a single API process.
- redis: two hashes ``{key}:pulls`` / ``{key}:rewards`` updated with HINCRBY /
         HINCRBYFLOAT, so several API workers can record feedback at once.

Feedback flow: the API shows an item with is_exploration=true; when the user
clicks it (reward 1) or the impression expires without a click (reward 0), call
``Reranker.record_feedback(product_id, clicked)``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .bandit import CategoryBandit


class BanditStore:
    def __init__(
        self,
        backend: str = "json",
        path: Path | None = None,
        redis_url: str | None = None,
        key: str = "stage3:bandit",
        client: Any = None,
    ) -> None:
        if backend not in ("json", "redis", "none"):
            raise ValueError("bandit store backend must be json, redis or none")
        if backend == "json" and path is None:
            raise ValueError("json backend needs a state path")
        self.backend, self.path, self.key = backend, path, key
        self.client = client
        if backend == "redis" and client is None:
            import redis

            url = os.getenv("REDIS_URL") or redis_url
            if not url:
                raise ValueError("redis backend needs redis_url or REDIS_URL")
            self.client = redis.Redis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------ read
    def load_into(self, bandit: CategoryBandit) -> bool:
        """Overwrite the bandit counts with the stored ones. False when nothing is stored."""
        pulls, rewards = self._read()
        if not pulls:
            return False
        for arm in pulls:
            if arm not in bandit.pulls:
                bandit.arms.append(arm)
                bandit.rewards[arm] = 0.0
        bandit.arms.sort()
        bandit.pulls.update({arm: int(float(v)) for arm, v in pulls.items()})
        bandit.rewards.update({arm: float(v) for arm, v in rewards.items()})
        return True

    def _read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.backend == "redis":
            return (
                self.client.hgetall(f"{self.key}:pulls"),
                self.client.hgetall(f"{self.key}:rewards"),
            )
        if self.backend == "json" and self.path and self.path.is_file():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            return state.get("pulls", {}), state.get("rewards", {})
        return {}, {}

    # ------------------------------------------------------------ write
    def record(self, bandit: CategoryBandit, arm: str, reward: float) -> None:
        """Update the in-memory bandit and persist the same change."""
        bandit.update(arm, reward)
        if self.backend == "redis":
            pipe = self.client.pipeline()
            pipe.hincrby(f"{self.key}:pulls", arm, 1)
            pipe.hincrbyfloat(f"{self.key}:rewards", arm, float(reward))
            pipe.execute()
            self.load_into(bandit)  # pick up feedback recorded by other workers
        elif self.backend == "json":
            self.save(bandit)

    def save(self, bandit: CategoryBandit) -> None:
        if self.backend != "json" or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False
        ) as handle:
            json.dump(bandit.to_dict(), handle, ensure_ascii=False)
        os.replace(handle.name, self.path)  # atomic on one filesystem
