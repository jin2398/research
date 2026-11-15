"""Popular and trending items for cold-start users, also the popularity baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class Event:
    user_id: str
    product_id: str
    event_type: str
    timestamp: datetime


def parse_time(value: str) -> datetime:
    """ISO date or datetime; timezone-aware values become naive UTC for comparison."""
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def count_events_by_user(events: Iterable[Event]) -> Counter[str]:
    """History length per user, used for the cold-start rule in offline evaluation."""
    return Counter(event.user_id for event in events)


class PopularityIndex:
    def __init__(self, scores: Mapping[str, float], trending: list[str]) -> None:
        self.scores = dict(scores)
        self.popular = sorted(self.scores, key=lambda pid: (-self.scores[pid], pid))
        self.trending = trending
        self._max = max(self.scores.values(), default=0.0) or 1.0

    @classmethod
    def from_events(cls, events: Iterable[Event], weights: Mapping[str, float],
                    now: datetime, window_days: int, min_recent: float = 3.0) -> "PopularityIndex":
        """Popularity = weighted event count up to `now`.

        Trending = growth of the last `window_days` over the window before it,
        limited to items with at least `min_recent` weighted events recently.
        Events after `now` are ignored so the test period never leaks in.
        """
        window = timedelta(days=window_days)
        total: dict[str, float] = defaultdict(float)
        recent: dict[str, float] = defaultdict(float)
        previous: dict[str, float] = defaultdict(float)
        for event in events:
            weight = weights.get(event.event_type, 0.0)
            if not weight or event.timestamp > now:
                continue
            total[event.product_id] += weight
            age = now - event.timestamp
            if age <= window:
                recent[event.product_id] += weight
            elif age <= 2 * window:
                previous[event.product_id] += weight
        growth = {pid: (score + 1) / (previous[pid] + 1)
                  for pid, score in recent.items() if score >= min_recent}
        trending = sorted(growth, key=lambda pid: (-growth[pid], -recent[pid], pid))
        return cls(total, trending)

    def normalized(self, product_id: str) -> float:
        return self.scores.get(product_id, 0.0) / self._max

    def mixed(self, n: int, trending_share: float,
              exclude: Iterable[str] = ()) -> list[tuple[str, str]]:
        """Interleave popular and trending items: [(product_id, "popular" | "trending")]."""
        seen = set(exclude)
        sources = {"popular": iter(self.popular), "trending": iter(self.trending)}
        result: list[tuple[str, str]] = []
        taken_trending = 0
        while len(result) < n:
            want = "trending" if taken_trending < round((len(result) + 1) * trending_share) else "popular"
            order = (want, "popular" if want == "trending" else "trending")
            picked = None
            for name in order:
                pid = next((p for p in sources[name] if p not in seen), None)
                if pid is not None:
                    picked = (pid, name)
                    break
            if picked is None:
                break
            seen.add(picked[0])
            taken_trending += picked[1] == "trending"
            result.append(picked)
        return result
