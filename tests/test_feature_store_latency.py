"""Feature Store 조회 속도 검증 (p95 10ms 이내)

실제 Redis가 필요하다. 연결할 수 없으면 건너뛴다.
  docker compose up -d redis  후  pytest
"""

import pytest
import redis

from src.serving import feature_store as fs

USER_ID = "latency_test_user"


@pytest.fixture
def real_redis():
    try:
        fs.redis_client.ping()
    except redis.RedisError:
        pytest.skip(f"Redis에 연결할 수 없음: {fs.REDIS_URL}")

    # 최악의 경우를 재기 위해 기록을 최대치(100개씩)로 채움
    for i in range(fs.MAX_HISTORY):
        fs.record_view(USER_ID, f"p{i}")
        fs.record_click(USER_ID, f"p{i}")

    yield

    fs.redis_client.delete(*(f"user:{USER_ID}:{k}" for k in ("views", "clicks", "session")))


def test_history_read_p95_under_10ms(real_redis):
    result = fs.measure_read_latency(USER_ID, repeat=1000)
    print(f"\nFeature Store 조회 속도: {result}")

    assert result["p95_ms"] <= fs.LATENCY_THRESHOLD_MS, result
    assert result["passed"] is True
