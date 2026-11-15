# RE:SEARCH — 멀티모달 패션 검색 · 개인화 추천 시스템

텍스트/이미지로 상품을 찾는 **멀티모달 검색 API**와, 사용자의 실시간 세션 행동을 반영하는
**Multi-Stage 개인화 추천 API**, 그리고 품질·A/B·재학습(CT)을 보는 **대시보드**를
`docker compose up` 한 번으로 실행합니다.

## 1. 빠른 실행

필요한 것: Docker Desktop (메모리 4GB 이상 할당)

> Docker Desktop이 없으면 [RUN_GUIDE.md](RUN_GUIDE.md)를 보세요. 다른 Docker 도구, Python만으로 실행, Redis 없이 실행하는 방법이 있습니다.

```bash
docker compose up --build        # 처음 빌드는 5~10분 (torch 설치)
```

| 서비스 | 주소 | 설명 |
|---|---|---|
| API 서버 | http://localhost:8000/docs | 검색·추천 API (Swagger에서 직접 호출 가능) |
| 대시보드 | http://localhost:8501 | 쇼핑 화면 + 검색/추천 품질, A/B 테스트, CT 모니터링 |
| Redis | localhost:6379 | Feature Store |
| 시뮬레이터 | (로그) | 가상 사용자 300명이 API를 호출하며 행동 로그 생성 → `data/simulated_logs.csv` |

정상 기동 로그:

```
[api-server] CLIP + FAISS 검색 인덱스 로딩 완료 (70,521 상품)
[api-server] Two-Tower + FAISS 후보 생성 모델 로딩 완료 (70,831 상품)
[api-server] DeepFM Ranking 모델 로딩 완료 (정렬 전략: deepfm)
[api-server] Stage 3 Re-ranking 로딩 완료
[api-server] http://localhost:8000 에서 API 서버 시작 (문서: /docs)
[simulator] 행동 로그 생성 시작... → data/simulated_logs.csv
```

> **CLIP 모델 가중치:** `artifacts/search/model_cache/`에 없으면 첫 기동 때 Hugging Face에서
> 약 600MB를 내려받습니다(인터넷 필요). 이 폴더는 볼륨으로 연결되어 한 번만 받습니다.

끄기: `docker compose down` (Redis 데이터 유지) / `docker compose down -v` (Redis 데이터 삭제)

## 2. API

### 멀티모달 검색 `POST /api/search`

```bash
# 텍스트
curl -X POST localhost:8000/api/search -H 'Content-Type: application/json' \
     -d '{"query_text": "black hoodie", "top_k": 10}'
# 이미지
curl -X POST localhost:8000/api/search -F file=@query.jpg -F top_k=10
# 텍스트 + 이미지 (hybrid)
curl -X POST localhost:8000/api/search -F file=@query.jpg -F query_text="beige coat"
```

| 필수 필드 | 타입 | 설명 |
|---|---|---|
| `search_type` | string | `text` / `image` / `hybrid` (입력에 따라 자동 결정) |
| `results[]` | array | `product_id`, `name`, `score`, `price` (+ `category`, `image_url` 등) |
| `latency_ms` | float | 검색 API 전체 처리 시간 (모델 처리만은 `engine_latency_ms`) |
| `total_count` | int | 반환한 결과 수 |

- `top_k` 기본 10, 최대 100. 기존 UI 호환을 위해 `query_text` 대신 `query`도 받습니다.
- 텍스트: BM25 + CLIP 순위 결합(RRF) + 속성(색상·패턴·카테고리) 매칭. 이미지/하이브리드: CLIP + FAISS HNSW.

### 개인화 추천 `GET /api/recommend?user_id=U00001&top_n=10`

| 필수 필드 | 타입 | 설명 |
|---|---|---|
| `user_id` | string | 요청 사용자 ID |
| `recommendations[]` | array | `product_id`, `score`, `reason`, `is_exploration` |
| `pipeline_latency` | object | `candidate_ms`, `ranking_ms`, `reranking_ms`, `total_ms` (+ `feature_store_ms`, `api_ms`) |
| `session_context` | object | `recent_clicks`, `session_interest` |

- `reason`: `ranking`, `session_interest`, `new_arrival`, `mab_exploration`, `popular`, `trending` 등
- 이력이 5건 미만인 신규 사용자는 인기·트렌드 상품으로 폴백합니다 (`cold_start: true`).
- Redis에 쌓인 현재 세션 클릭이 요청마다 User Tower·DeepFM·Re-ranking에 즉시 반영됩니다.

### 행동 로그 · Feature Store

| 엔드포인트 | 설명 |
|---|---|
| `POST /api/events` | `{"user_id", "product_id", "event_type": view/click/cart/purchase}` 저장 |
| `GET /api/users/{user_id}/history` | 최근 조회/클릭 기록 |
| `GET /api/feature-store/latency?repeat=1000` | 피처 조회 속도 측정 (p95 ≤ 10ms 판정) |
| `GET /api/status` | 모델 로딩 상태, 현재 추천 정렬 전략 |

MAB 피드백은 자동입니다. 추천에서 보여준 탐색 상품을 클릭하면 보상 1, 다음 추천 요청 때까지
클릭하지 않으면 보상 0이 UCB 밴딧에 기록됩니다.

## 3. 아키텍처

```
                 ┌──────────── simulator (config/simulator.yaml) ────────────┐
                 │  가상 사용자: 검색 → 추천 확인 → 조회/클릭 → 장바구니 → 구매   │
                 └──────┬───────────────────────────────┬────────────────────┘
                        │ HTTP                          │ append
                        ▼                               ▼
┌──────────────────────── api (FastAPI) ─────────┐   data/simulated_logs.csv
│ POST /api/search                               │        │
│   CLIP 텍스트/이미지 인코더 → FAISS HNSW (7만)    │        ▼
│   + BM25 + 속성 매칭 (텍스트)                     │   ct_monitor.py
│ GET /api/recommend                             │   (신규 로그 1만 건 / HitRate@50 < 0.2
│   Redis 세션 클릭 ─┐                            │    → 재학습 트리거)
│   Stage 1 Two-Tower + FAISS 후보 300개          │
│   Stage 2 DeepFM CTR 정렬 (전략 교체 가능)        │
│   Stage 3 Re-ranking: 세션 관심사, 다양성,         │
│           신상품 슬롯, MAB 탐색 슬롯, Cold Start   │
│ POST /api/events ──→ Redis Feature Store ◀──────┼── redis
└────────────────────────────────────────────────┘
                        ▲
                        │ HTTP (브라우저)
             dashboard (Streamlit :8501)
             쇼핑 화면 · MRR/NDCG · HitRate/Coverage · A/B(p-value, CI) · CT
```

### 폴더 구조

```
config/                  모든 설정 (코드 수정 없이 동작 변경)
  stage12.yaml             추천 Stage 1·2 + serving.ranking_strategy
  rerank.yaml              Stage 3 (다양성, 신상품, MAB, 세션, Cold Start)
  search_full.yaml         CLIP 검색
  simulator.yaml           시뮬레이터 (페르소나 비율, 카테고리 구조, 행동 확률)
src/
  search/                  CLIP + FAISS, BM25, 텍스트 결합
  recommendation/          Two-Tower, DeepFM, Re-ranking, MAB, 세션 GRU, 정렬 전략
  serving/                 통합 API(api.py), 응답 규격(schemas.py), Redis Feature Store
  simulator/               행동 로그 시뮬레이터
  dashboard/               Streamlit 대시보드, A/B 테스트 통계
  evaluation/              오프라인 평가 (MRR, NDCG, Recall, HitRate, Coverage)
ct_monitor.py            Continuous Training 모니터 (성능 저하·로그 누적 → 재학습)
scripts/benchmark_api.py API 응답 시간 측정 (p95 ≤ 200ms)
docker/                  Dockerfile.api, Dockerfile.dashboard
data/                    상품·행동 로그 (볼륨)
artifacts/               학습된 모델, FAISS 인덱스 (볼륨)
docs/                    평가 결과·실험 리포트, docs/guides/ 모듈별 상세 문서
tests/                   단위·규격·속도 테스트
```

## 4. 설정만 바꿔서 동작 바꾸기

**추천 알고리즘 교체** (전략 패턴, `src/recommendation/strategies.py`)

```yaml
# config/stage12.yaml
serving:
  ranking_strategy: newest   # deepfm(기본) | similarity | newest(최신순) | popularity
```

```bash
docker compose restart api                       # 또는
RANKING_STRATEGY=newest docker compose up -d api
```

Stage 1 후보 생성과 Stage 3 Re-ranking은 그대로 두고 Stage 2 정렬 기준만 바뀝니다.
새 전략은 `RankingStrategy`를 상속하고 `@register`를 붙이면 추가됩니다.
평가 지표(5절)는 기본값 `deepfm` 기준입니다.

**시뮬레이터 데이터 변경**: `config/simulator.yaml`의 `personas`(비율·행동 확률)와
`categories`(카테고리 구조)를 고친 뒤 `docker compose restart simulator`.

## 5. 성능 검증 결과

데이터 분할: 행동 로그 100만 건을 시간순 정렬 후 **train 80% / valid 10% / test 10%**
(`docs/split_report.json`, test 사용자 8,054명).

| 구분 | 지표 | 목표 | 결과 | 달성 | 출처 |
|---|---|---|---|---|---|
| 검색 | MRR (텍스트, attribute_fusion) | ≥ 0.55 | 0.9986 | ✅ | `docs/search_text_fusion.json` (test 11,300 질의) |
| 검색 | NDCG@10 (텍스트, attribute_fusion) | ≥ 0.50 | 0.9961 | ✅ | 〃 |
| 검색 | MRR / NDCG@10 (BM25 단독) | ≥ 0.55 / 0.50 | 0.9245 / 0.8751 | ✅ | 〃 |
| 검색 | MRR / NDCG@10 (CLIP 단독, 전체 인덱스) | ≥ 0.55 / 0.50 | 0.6698 / 0.5612 | ✅ | 〃 |
| 추천 | Recall@300 | ≥ 0.30 | 0.2315 | ❌ | `docs/stage12_eval.json` |
| 추천 | HitRate@50 | ≥ 0.20 | 0.2703 | ✅ | 〃 (Stage 1→2→3) |
| 추천 | NDCG@50 | ≥ 0.08 | 0.0243 | ❌ | 〃 |
| Ranking | AUC (DeepFM) | ≥ 0.70 | 0.7482 | ✅ | 〃 |
| 추천 | Coverage | ≥ 0.20 | 0.2759 | ✅ | 〃 |
| Re-ranking | 같은 카테고리 3연속 | 0건 | 0건 | ✅ | 〃 |
| 서빙 | Feature Store 조회 p95 | ≤ 10ms | 0.995ms | ✅ | `GET /api/feature-store/latency` (1,000회) |
| 서빙 | 추천 API p95 | ≤ 200ms | 84.4ms (신규 사용자 35.5ms) | ✅ | `docs/api_latency.json` |
| 서빙 | 텍스트 검색 API p95 | ≤ 200ms | 178.1ms | ✅ | 〃 |
| 서빙 | 이미지 / 하이브리드 검색 API p95 | ≤ 200ms | 350.5ms / 305.7ms | ❌ | 〃 |

- 검색 지표는 검색 담당자의 오프라인 평가 코드로 잰 값입니다. API의 텍스트 검색은 같은 설정
  (depth 100, CLIP 가중치 0.15, RRF 상수 60, 속성 매칭)을 쓰지만 BM25 구현이 평가 코드와 달라,
  이 수치가 라이브 API의 실측 성능과 같다고 보장하지는 않습니다 (`docs/guides/README_SEARCH_SETUP.md` 3절).
- 미달 항목(Recall@300, NDCG@50)의 원인 분석은 `docs/stage12_report.md` 4-2절에 있습니다.

**응답 시간 측정 조건:** `docker compose up` 전체 실행 중(시뮬레이터 부하 포함), MacBook의
Docker Desktop(Linux arm64 VM, CPU 8 / 메모리 4GB), HTTP 요청 전체 시간, 워밍업 10회 후 100회.
같은 조건에서도 측정마다 p95가 크게 흔들립니다 (텍스트 검색 42~178ms).

- **이미지·하이브리드 검색 미달:** 처리 시간의 약 97%가 CLIP 이미지 인코더(ViT-B/32) CPU 추론입니다.
  검색 담당자의 파일럿 측정(Windows x86, CPU 8)에서도 이미지 p95가 298.7ms였습니다
  (`docs/search_latency_pilot.json`). 개선하려면 ONNX Runtime 변환, int8 양자화, GPU 서빙 같은
  모델 측 작업이 필요하며 정확도 재검증이 따라야 합니다.
- **스레드 설정:** torch·faiss 기본값(CPU 전체 사용)에서는 요청 간 스레드 경합으로 추천 p95가
  201ms였습니다. `OMP_NUM_THREADS=4`에서 84ms로 줄었습니다 (1·2·4개 비교 후 선택).
- 같은 코드를 Mac에서 Docker 없이 실행하면 모두 통과합니다 (이미지 p95 56ms, 추천 20ms).

응답 시간 재측정:

```bash
docker compose exec api python scripts/benchmark_api.py --calls 100 --save
```

## 6. 로컬 개발 · 테스트

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
docker compose up -d redis
pytest                       # 규격·전략·시뮬레이터·Feature Store 속도 테스트 포함
```

- **macOS:** faiss와 torch가 각자 OpenMP를 불러와 충돌합니다. 로컬 실행 시
  `export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` 를 먼저 설정하세요 (Docker에서는 불필요).
- `tests/test_storefront_html.py::test_demo_standalone_is_the_same_html`은 알려진 실패입니다
  (실제 `data/recommendation_metrics.json`을 읽어 스냅샷과 달라짐, `docs/guides/README_STAGE12.md` 6절).

## 7. 팀 역할

> TODO: 이름과 기여 내용을 채워 주세요. 아래는 코드·문서에서 확인되는 모듈 기준입니다.

| 이름 | 역할 | 주요 기여 (구현 모듈, 문서) |
|---|---|---|
| 문석민 | 멀티모달 검색 | `src/search/` CLIP + FAISS 인덱스, 텍스트 결합 · `docs/search_*.json` |
| (이름) | 추천 Stage 1·2 | `src/recommendation/two_tower.py`, `deepfm.py` · `docs/stage12_report.md` |
| 권진영 | 추천 Stage 3 | `src/recommendation/rerank.py`, MAB, 세션 GRU · `docs/stage3_report.md` |
| (이름) | 대시보드 · A/B · CT | `src/dashboard/`, `ct_monitor.py` · `docs/guides/README_AB_CT.md` |
| 김신비 | API 서빙 & Feature Store | `src/serving/`, Redis Feature Store, Docker Compose, 시뮬레이터 |

## 8. 상세 문서

- `docs/guides/README_STAGE12.md` — 추천 Stage 1·2 (Two-Tower, DeepFM)
- `docs/stage3_report.md` — Stage 3 Re-ranking 실험
- `docs/guides/README_SEARCH_SETUP.md` — 검색 모듈
- `docs/guides/README_AB_CT.md` — A/B 테스트 · CT 모니터링
- `docs/guides/README_UI.md` — 대시보드 화면
