# Stage 3 Re-ranking & 세션 기반 실시간 추천 실험 리포트

담당: 권진영 · 기준일: 2026-11-09 · 원자료: `docs/stage3_eval.json`, `docs/stage3_experiments.json`, `docs/session_encoder_report.json`

> **읽기 전 주의:** 이 리포트의 HitRate@50 / NDCG@50은 Stage 1(Two-Tower)·Stage 2(DeepFM)가 아직 연결되지 않은 상태에서, **인기도 후보** 또는 **세션 kNN 후보**를 Stage 2 대용으로 넣어 측정한 값이다. 최종 수치는 Stage 1/2 연결 후 같은 스크립트로 재측정한다 (6절).

## 1. Stage 3가 하는 일

```
Stage 2 후보 [(product_id, score)] ─┐
Redis: 이력 수, 최근 클릭 ──────────┤
                                    ▼
             ┌── 이력 < 5 ? ── 예 ──> 인기 + 트렌딩 폴백 (Cold Start)
             │        아니오
             ▼
  GRU 세션 인코더(최근 클릭 20개 → 세션 벡터, CLIP 512차원)
             ▼
  최종 점수 = (1 − w) · Stage2 점수(정규화) + w · cos(세션 벡터, 상품 CLIP 벡터)
             ▼
  슬롯 예약: 신상품 1개(위치 2), MAB 탐색 2개(위치 6, 9)
             ▼
  조립: 같은 카테고리 3개 연속 금지 (막히면 다음 후보 → 인기 상품으로 채움)
             ▼
  recommendations[] (product_id, score, reason, is_exploration) + session_context
```

| 요구사항 | 구현 | 코드 |
|---|---|---|
| 같은 카테고리 3개 이상 연속 금지 | 조립 단계에서 직전 2개와 같은 카테고리면 건너뜀 | `rerank.py` `_allowed`, `_assemble` |
| 신규 사용자(이력 5개 미만) 폴백 | 인기 60% + 트렌딩 40% 교차 배치 | `popularity.py` `mixed` |
| 신상품(등록 7일 이내) 노출 보장 | 7일 이내 상품 중 세션 유사도·인기 순 1개를 위치 2에 고정 | `rerank.py` `_new_arrivals` |
| MAB 탐색 슬롯 1~2개 | 카테고리 단위 bandit(UCB / ε-greedy)이 고른 카테고리의 인기 상위 500개 중 무작위 1개 | `bandit.py`, `rerank.py` `_exploration` |
| 탐색 피드백 반영 | 클릭 = 보상 1, 무시 = 0. JSON 파일 또는 Redis 해시에 누적 | `bandit_store.py`, `Reranker.record_feedback` |
| 세션 인코더(GRU) | 최근 클릭 CLIP 벡터 → GRU → 다음 클릭 벡터 예측 | `session_encoder.py` |
| 장기 + 단기 결합 | Stage 2 점수(장기 선호)와 세션 유사도(단기 관심)의 가중합, w = 0.6 | `rerank.py` `_score_candidates` |

**등록일 처리:** 시뮬레이터 상품 데이터에 등록일이 없어, 로그에 처음 등장한 시각을 등록일로 사용한다 (시뮬레이션 시계 기준, `created_at_fallback: first_seen`). 상품의 `is_new` 플래그는 카탈로그의 20%에 붙어 있어 "7일 이내"로 쓰기에는 너무 넓다.

## 2. 평가 프로토콜

| 항목 | 값 |
|---|---|
| 데이터 | 시뮬레이터 로그 `data/behavior_logs.csv` 1,000,000행, 사용자 10,000명 |
| 분할 | 전체 행 기준 시간순 train / valid / test = 8 / 1 / 1 (800,000 / 100,000 / 100,000행), `docs/split_report.json`과 동일 규칙 |
| 평가 시점 | valid 마지막 이벤트 시각 2025-03-22 23:59:59. 인기도·이력·최근 클릭은 train+valid로 계산 |
| 정답 | test 기간 동안 사용자가 view / cart / purchase 한 상품 집합 (사용자당 평균 9.3개) |
| 평가 사용자 | test 기간 활동 사용자 8,054명 |
| 카탈로그 | 시뮬레이터 카탈로그 70,831개 (`data/product_features.csv`). 아동·유아 상품 제외 |
| k | 50 (추천), 후보 수 300 |
| 시드 | random_seed = 42 (분할, bandit, 탐색 무작위 선택, GRU 학습) |

### 지표 정의 (`src/evaluation/metrics.py`)

- **HitRate@k** = 상위 k개 중 정답이 하나라도 있으면 1, 없으면 0 → 사용자 평균
- **NDCG@k** = DCG@k / IDCG@k, DCG@k = Σᵢ relᵢ / log₂(i + 1), IDCG@k = Σ_{i=1..min(k, |정답|)} 1 / log₂(i + 1)
- **Coverage** = 전체 사용자에게 추천된 고유 상품 수 / 카탈로그 상품 수(70,831)
- **Recall@k** = |상위 k ∩ 정답| / |정답|, **MRR@k** = 첫 정답 순위의 역수

## 3. 결과: 베이스라인 대비

| 전략 | HitRate@50 | NDCG@50 | Coverage | HitRate 개선율 |
|---|---|---|---|---|
| 인기도 (베이스라인) | 0.0995 | 0.0083 | 0.0007 | — |
| 세션 kNN (GRU 벡터와 가까운 CLIP 상품, 재정렬 없음) | 0.0508 | 0.0029 | 0.0179 | −48.9% |
| **인기도 + Stage 3** | **0.1127** | **0.0089** | **0.0975** | **+13.3%** |
| 세션 kNN + Stage 3 | 0.0482 | 0.0027 | 0.1164 | −51.6% |
| 인기도 + Stage 3, GRU 대신 가중 평균 (ablation) | 0.1055 | 0.0084 | 0.0982 | +6.0% |
| **목표** | **0.20** | **0.08** | **0.20** | |

- Stage 3는 같은 인기도 후보에서 HitRate@50을 13.3%, NDCG@50을 7.2%, Coverage를 0.0007에서 0.0975로 올린다.
- GRU 세션 인코더는 기존 가중 평균 대비 HitRate@50을 0.1055 → 0.1127 (+6.8%) 올린다.
- 세 지표 모두 목표에 미달한다. 원인은 후보 품질이다. Stage 3는 받은 후보를 재정렬할 뿐이고, 인기도 후보 300개는 모든 사용자에게 같다.

### Stage 3 규칙 검증 (인기도 + Stage 3, 8,054개 목록)

| 항목 | 결과 |
|---|---|
| 같은 카테고리 최대 연속 수 | 2 (위반 목록 0개) |
| 목록당 MAB 탐색 슬롯 (`is_exploration = true`) | 2.0 |
| 목록당 신상품 | 1.0 |
| 신규 사용자(이력 0) 폴백 | `cold_start = true`, popular 29 · trending 19 · new_arrival 1 · mab_exploration 1 |
| 세션 관심 사유(`session_interest`) | 1,074건 (세션 유사도 ≥ 0.75) |

## 4. 세션 인코더 (GRU)

- **구조:** 클릭 CLIP 벡터(512) → Linear+GELU(256) → GRU(256) → Linear(512) + 마지막 클릭 잔차 → L2 정규화
- **학습:** train 구간 세션 105,001개(클릭 2개 이상), 다음 클릭 예측, 배치 내 음성 InfoNCE (τ = 0.07), 3 epoch, CPU 94초. loss 5.754 → 5.706
- **평가:** valid 세션 5,000개, 마지막 클릭을 정답으로 전체 CLIP 상품 70,521개에서 상위 20개 검색

| 세션 신호 | 다음 클릭과 같은 카테고리 비율@20 | 카테고리 적중률@20 | 정확 상품 HitRate@20 |
|---|---|---|---|
| 마지막 클릭 | 0.176 | 0.334 | 0.0016 |
| 가중 평균 (이전 방식) | 0.197 | 0.427 | 0.0016 |
| **GRU** | **0.247** | **0.533** | 0.0012 |

정확한 상품 적중은 세 방식 모두 0에 가깝다. 시뮬레이터가 선호 카테고리 안에서 상품을 거의 무작위로 고르기 때문이다. Stage 3가 쓰는 신호는 카테고리 수준의 관심이고, 그 기준에서 GRU가 가중 평균보다 25% 정확하다.

## 5. 실험

### 5-1. Coverage: 탐색 슬롯 · 탐색 풀 · 세션 가중치 (`stage3_experiments coverage`)

24개 조합, 전체 사용자 8,054명, 피드백 없음. 인기도 후보 기준 주요 결과:

| 탐색 슬롯 | 탐색 풀 | 세션 가중치 w | HitRate@50 | NDCG@50 | Coverage |
|---|---|---|---|---|---|
| 1 | 50 | 0.3 | 0.1018 | 0.0082 | 0.0373 |
| 1 | 50 | 0.6 | 0.1152 | 0.0092 | 0.0375 |
| 1 | 500 | 0.6 | 0.1150 | 0.0091 | 0.0587 |
| **2** | **500** | **0.6** | **0.1127** | **0.0089** | **0.0954** |
| 2 | 5000 | 0.6 | 0.1129 | 0.0089 | 0.1008 |
| 2 (세션 kNN 후보) | 5000 | 0.6 | 0.0484 | 0.0027 | 0.1200 |

- 세션 가중치를 0.3 → 0.6으로 올리면 HitRate@50이 0.1018 → 0.1150 (+13%) 오른다.
- 탐색 슬롯 2개와 탐색 풀 500개는 Coverage를 2.5배 올리고, HitRate 손실은 0.003 이내다 (0.1152 → 0.1127).
- 탐색 풀을 500 → 5000으로 키워도 Coverage는 0.005만 더 오른다. 탐색 상품의 품질(인기 상위)을 유지하기 위해 500을 기본값으로 정했다.
- **채택한 기본값:** 탐색 슬롯 2, 탐색 풀 500, w = 0.6 (`config/rerank.yaml`)

### 5-2. MAB 알고리즘 비교 (`stage3_experiments bandit`)

test 사용자를 첫 test 이벤트 시각 순으로 재생하면서, 탐색 슬롯마다 피드백을 준다(16,108회).

- **보상 대리 지표:** 탐색한 카테고리의 상품을 사용자가 test 기간에 하나라도 이용했으면 1이다. 오프라인 로그에서는 상품 단위 클릭이 너무 드물어 알고리즘 비교가 불가능하다(16,108회 중 1~11회).
- **최적 고정 arm:** Dress, 보상률 0.321

| 알고리즘 | 보상률 | 전반 → 후반 | Regret | HitRate@50 | Coverage | 가장 많이 고른 카테고리 |
|---|---|---|---|---|---|---|
| random | 0.035 | 0.038 → 0.032 | 4,615 | 0.1127 | 0.0965 | (고르게 분산) |
| ε-greedy (ε = 0.1) | **0.276** | 0.295 → 0.258 | **724** | 0.1134 | **0.0399** | Dress 7,398회, Trousers 6,170회 |
| **UCB (c = 1.0)** | 0.139 | **0.114 → 0.165** | 2,930 | 0.1130 | **0.0953** | Dress 3,613회, Trousers 2,335회 |

- ε-greedy는 보상이 가장 높지만, 탐색 슬롯의 84%를 두 카테고리에 써서 Coverage가 0.096 → 0.040으로 무너진다. 탐색 슬롯이 사실상 활용(exploitation) 슬롯이 된다.
- UCB는 후반으로 갈수록 보상이 오르면서(학습 중) Coverage를 유지한다.
- **채택한 기본값:** UCB. Coverage가 담당 목표 지표이고, 탐색 슬롯의 목적이 노출 다양화이기 때문이다.

## 6. 응답 시간 (warm-up 10회 후 500회, 단건, CPU)

| 구간 | p50 | p95 |
|---|---|---|
| GRU 세션 인코딩 | 0.68 ms | 1.25 ms |
| Stage 3 재정렬 전체 (후보 300개, top 50) | 3.71 ms | 5.68 ms |

측정 환경: macOS (Apple, CPU 전용), Python 3.12, PyTorch 2.14. 추천 API 200ms 예산 중 Stage 3가 차지하는 부분은 약 6ms 이하다.

## 7. 한계와 다음 단계

1. **후보 의존성:** HitRate@50 ≥ 0.20, NDCG@50 ≥ 0.08은 Stage 1/2 후보 품질에 좌우된다. 데이터 분석상 test 사용자의 21%가 이전에 본 상품을 다시 보며, 사용자별 "최근 본 상품 50개"만으로 HitRate@50 = 0.156이 나온다(인기도 0.0995). Stage 1 후보에 이 신호를 넣는 것을 Stage 1 담당과 논의한다.
2. **Coverage 0.20:** Stage 3만으로는 0.10~0.12가 한계였다. 개인화된 Stage 2 후보가 들어오면 기본 목록 자체가 사용자마다 달라져 Coverage가 오른다. 재측정 후에도 부족하면 보너스 과제의 MMR을 추가한다.
3. **재측정 방법:** `evaluate_stage3.py`의 후보 함수에 Stage 2 출력 `[(product_id, score)]`를 넣고 실행한다. 나머지 절차는 같다.
4. **bandit 보상:** 시뮬레이션 보상은 카테고리 단위 대리 지표다. 운영에서는 API가 노출(`is_exploration = true`)과 클릭을 받아 `record_feedback`을 호출해야 한다.

## 8. 재현 방법

```bash
# 0) 데이터: data/behavior_logs.csv, data/product_features.csv (multi_core.zip)
#            data/hm_products.csv (category_metadata/data/processed/products.csv)
# 1) GRU 세션 인코더 학습 + 평가 → artifacts/recommendation/session_gru.pt, docs/session_encoder_report.json
python -m src.recommendation.session_encoder all
# 2) Stage 3 오프라인 평가 → docs/stage3_eval.json
python -m src.evaluation.evaluate_stage3
# 3) Coverage 그리드 + bandit 비교 → docs/stage3_experiments.json
python -m src.evaluation.stage3_experiments all
# 4) 테스트
python -m unittest tests.test_stage3_session_eval tests.test_bandit_feedback -v
python -m pytest tests/test_rerank.py
```

CLIP 벡터는 첫 실행 때 검색팀 인덱스(`artifacts/search/clip_hnsw_full`)에서 `artifacts/recommendation/clip_vectors/`로 한 번 추출된다. macOS에서 faiss와 torch를 한 프로세스에 함께 로드하면 OpenMP 충돌로 종료되기 때문에, 추출은 별도 프로세스에서 한다.

설정 파일: `config/rerank.yaml`, `config/session_encoder.yaml`, `config/evaluate_stage3.yaml`
