# RE:SEARCH — Stage 1·2 추천 모델 (Two-Tower + DeepFM)

Multi-Stage 추천의 앞 두 단계를 담당하는 모듈입니다.

- **Stage 1 Candidate Generation:** PyTorch Two-Tower로 사용자·상품 벡터를 만들고, 상품 벡터를 오프라인에서 FAISS HNSW 인덱스로 저장해 사용자별 후보 300개를 추출합니다.
- **Stage 2 Ranking:** DeepFM(FM + DNN)이 후보의 CTR/CVR을 예측해 정렬합니다.
- 출력 `[(product_id, score)]`는 Stage 3 Re-ranking(`src/recommendation/rerank.py`)의 입력 형식 그대로이며, `RecommendationService`가 Stage 1→2→3를 묶어 추천 API 응답을 만듭니다.

실험 상세는 [`docs/stage12_report.md`](docs/stage12_report.md)를 참고하세요.

## 1. 성능 요약

| 지표 (test 사용자 8,054명) | 목표 | 결과 | |
|---|---|---|---|
| Recall@300 (Two-Tower + FAISS) | ≥ 0.30 | 0.2315 | ❌ 인기도 0.0758 대비 +205%. 데이터 상한 분석상 0.27 근처가 한계 (리포트 4-2절) |
| 후보 300개 추출 p95 | ≤ 100 ms | 3.5 ms | ✅ |
| Ranking AUC (DeepFM) | ≥ 0.70 | 0.7482 | ✅ |
| HitRate@50 (Stage 1→2→3) | ≥ 0.20 | 0.2703 | ✅ |
| NDCG@50 (Stage 1→2→3) | ≥ 0.08 | 0.0243 | ❌ |
| Coverage (Stage 1→2→3) | ≥ 0.20 | 0.2759 | ✅ |
| 추천 전체 p95 (HTTP 제외) | ≤ 200 ms | 17.5 ms | ✅ |

실측 결과는 `data/recommendation_metrics.json`에 저장되며, 대시보드와 `ct_monitor.py`가 이 파일을 읽습니다(예시값 아님).

## 2. 구성

```
config/stage12.yaml                      # 모든 설정 (경로, 피처, 하이퍼파라미터, 평가)
requirements-recommendation.txt
src/recommendation/stage12_data.py       # 상품 인코딩 · 시간 분할 · 사용자 스냅샷
src/recommendation/two_tower.py          # Stage 1: Two-Tower + FAISS, CandidateGenerator
src/recommendation/deepfm.py             # Stage 2: DeepFM (CTR/CVR), Ranker
src/recommendation/pipeline.py           # RecommendationService: Stage 1→2→3 → API 응답, 재학습 명령
src/evaluation/evaluate_stage12.py       # 평가 + Recall@300 상한 분석 → docs/stage12_eval.json, data/recommendation_metrics.json
tests/test_stage12.py                    # 합성 데이터 테스트 8개
docs/stage12_report.md                   # 실험 리포트
docs/stage12_eval.json                   # 모든 평가 결과 (지표, 응답 시간, 상한 분석, API 벤치마크)
data/recommendation_metrics.json         # 대시보드/CT용 실측 지표
artifacts/recommendation/stage12/        # 학습 결과 (약 87MB, 아래 표)
```

| 산출물 | 설명 |
|---|---|
| `encoders.json` | 상품 70,831개의 피처 코드 (Item Tower·DeepFM 공용) |
| `two_tower.pt`, `item_embeddings.npy`, `items.index` | Two-Tower 가중치, 오프라인 상품 벡터(64차원), FAISS HNSW 인덱스 |
| `deepfm.pt` | DeepFM 가중치 |
| `user_state.pkl`, `popularity.npy` | 서비스용 사용자 스냅샷(전체 로그 기준), 상품 인기도 |
| `two_tower_report.json`, `deepfm_report.json` | 학습 기록 (epoch별 loss·검증 지표, 설정값) |

이 모듈은 기존 파일을 수정하지 않습니다. 단, 평가에서 Stage 3를 실행하면 `rerank.py`가 `artifacts/recommendation/clip_vectors/` 캐시를 자동으로 다시 만들 수 있습니다(경로 기반 캐시 검사, 내용은 동일).

## 3. 설치와 실행 (Windows PowerShell)

Python 3.11~3.13, CPU 환경 기준입니다.

```powershell
py -3.12 -m venv .venv-rec
.\.venv-rec\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-recommendation.txt
python -m pytest tests/test_stage12.py -q          # 8 passed
```

학습 결과가 포함되어 있으므로 **재학습 없이** 바로 실행할 수 있습니다.

```powershell
python -m src.recommendation.pipeline demo --user-id U00001 --top-n 10
python -m src.recommendation.pipeline demo --user-id U00001 --recent 0837810001,0889550002   # 세션 클릭 반영
python -m src.recommendation.pipeline demo --user-id NEW_USER                               # Cold start
python -m src.recommendation.pipeline bench --calls 100 --save                              # p95 → docs/stage12_eval.json
```

처음부터 재현하는 방법은 리포트 8절에 있습니다 (Two-Tower 약 13분 + DeepFM 약 5분 + 평가 약 4분, CPU).

> **faiss와 torch를 한 프로세스에서 쓸 때는 faiss를 먼저 import** 하세요. 순서가 바뀌면 Windows에서 OpenMP 충돌(`libiomp5md.dll already initialized`)로 종료됩니다. 이 모듈들은 이미 그렇게 되어 있습니다.

## 4. Stage 3 (Re-ranking) 연동

- Stage 2 출력은 `[(product_id, score)]`, score = DeepFM p_ctr (0~1, 높을수록 좋음), 사용자당 최대 320개입니다.
- 후보 풀에는 FAISS 상위 300개와, 그 안에 없는 사용자의 최근 클릭 상품 최대 20개가 들어갑니다(재방문 신호, `config/stage12.yaml` → `deepfm.recent_candidates`). 이 옵션으로 최종 Coverage가 0.17 → 0.28, HitRate@50이 0.260 → 0.270이 됐습니다.
- 실제 Stage 2 후보로 측정한 Stage 3 결과는 `docs/stage12_eval.json`의 `two_tower+deepfm+stage3`에 있습니다.
  - HitRate@50 0.2703, NDCG@50 0.0243, Coverage 0.2759, 같은 카테고리 3연속 위반 0건, 목록당 탐색 슬롯 2.0 · 신상품 1.0
  - 같은 후보를 Stage 3 없이 쓰면 HitRate@50 0.2832 / NDCG@50 0.0275 / Coverage 0.2115입니다. Stage 3 기본값(세션 가중 0.6, 탐색 슬롯 2)은 인기도 후보 기준으로 튜닝된 값이라, 이 후보로 다시 튜닝할 여지가 있습니다.

```python
from src.recommendation.pipeline import RecommendationService
svc = RecommendationService()            # config/stage12.yaml + config/rerank.yaml
res = svc.recommend("U00001", top_n=50)
```

## 5. API 서빙 연동 (FastAPI · Redis · Docker)

`RecommendationService`를 서버 시작 시 한 번 만들고, 요청마다 `recommend()`를 호출하면 미션 규격의 응답이 나옵니다.

```python
from fastapi import FastAPI, Query
from src.recommendation.pipeline import RecommendationService

app = FastAPI()
svc = RecommendationService()            # 기동 약 10초

@app.get("/api/recommend")
def recommend(user_id: str, top_n: int = Query(10, ge=1, le=100)):
    recent = []                          # TODO: Redis에서 이 사용자의 현재 세션 클릭 (오래된 것 → 최신)
    return svc.recommend(user_id, top_n, recent_clicks=recent)
```

응답 예 (실제 출력, 상품 일부 생략):

```json
{
  "user_id": "U00001",
  "recommendations": [
    {"product_id": "0730361001", "score": 0.494185, "reason": "ranking", "is_exploration": false},
    {"product_id": "0629381002", "score": 0.6, "reason": "new_arrival", "is_exploration": false},
    {"product_id": "0581363001", "score": 0.001987, "reason": "mab_exploration", "is_exploration": true}
  ],
  "pipeline_latency": {"candidate_ms": 2.052, "ranking_ms": 1.565, "reranking_ms": 7.12, "total_ms": 10.737},
  "session_context": {"recent_clicks": ["0545263002", "0780955001", "0719109001", "0751720001", "0751720001"],
                      "session_interest": "원피스"},
  "cold_start": false,
  "candidate_count": 320
}
```

- **Feature Store 연결:** `recent_clicks`(product_id 리스트)를 넘기면 사용자 스냅샷 뒤에 붙어 User Tower·DeepFM·Stage 3 모두에 즉시 반영됩니다. 넘기지 않으면 `user_state.pkl`(로그 전체 스냅샷)만 사용합니다.
- **MAB 피드백:** 탐색 상품(`is_exploration=true`)이 클릭/무시되면 `svc.record_feedback(product_id, clicked)`를 호출합니다.
- **응답 시간:** 위 수치는 HTTP·Redis를 제외한 값입니다. API 서버 구간을 측정해 합산해야 합니다.
- **Docker:** `requirements-recommendation.txt` + CPU torch(3절 명령)를 설치하고, `artifacts/recommendation/stage12/`(87MB), `config/`, `data/product_features.csv`, `data/hm_products.csv`, `data/behavior_logs.csv`(Stage 3 인기도 계산용)를 이미지나 볼륨에 포함합니다. 서비스 프로세스 메모리는 약 0.8GB(Stage 3 포함 실측)입니다.

## 6. 대시보드 · CT 연동

- `data/recommendation_metrics.json`에 실측값(`recall_at_300`, `auc`, `hit_rate_at_50`, `ndcg_at_50`, `coverage` + `_source`)이 들어 있어, 대시보드 추천 품질 탭이 "미제공" 대신 실제 수치를 표시합니다. `evaluate_stage12`를 다시 실행하면 자동 갱신됩니다 (`--no-dashboard`로 끌 수 있음).
- **CT 재학습 명령:** `ct_monitor.py`는 셸 없이 한 프로세스를 실행하므로 한 줄짜리 명령을 제공합니다. Two-Tower → FAISS → DeepFM → 사용자 상태 → 평가(`data/recommendation_metrics.json` 갱신)까지 수행합니다 (약 20분, 종료 코드 0 = 모든 단계 성공).
  ```powershell
  python ct_monitor.py --logs data/behavior_logs.csv --train-command "python -m src.recommendation.pipeline retrain"
  ```
- **A/B 테스트 전략 후보:** `RecommendationService(use_stage3=False)`(Stage 2까지) vs 기본(Stage 3 포함), 또는 `deepfm.recent_candidates` 0 vs 20.
- **알려진 테스트 실패:** `tests/test_storefront_html.py::test_demo_standalone_is_the_same_html`은 실제 `data/` 폴더를 읽기 때문에, `data/recommendation_metrics.json`이 생긴 뒤로 화면에 `0.2703`이 표시되어 스냅샷과 달라집니다 (파일을 치우면 통과). 테스트가 임시 폴더를 쓰도록 바꾸거나 스냅샷을 갱신해야 합니다. 나머지 테스트 67개는 통과합니다(`test_search_bridge.py`는 fastapi 미설치 환경에서 제외).

## 7. 한계

1. Recall@300과 NDCG@50은 목표에 미달합니다. 리포트 4-2절의 상한 분석(test 기간 정답 버킷을 알려줘도 0.267)이 근거이며, 수치를 높이기 위한 지표 정의 변경은 하지 않았습니다.
2. Two-Tower(0.2315)가 단순 규칙(0.2502)보다 약간 낮아 개선 여지가 있습니다 (리포트 6절).
3. 응답 시간은 Windows 11, Intel 10세대 12스레드, CPU 전용 환경에서 측정했습니다. 최소 사양(4코어)에서는 다시 측정해 기록해야 합니다.
