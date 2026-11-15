# Stage 1 Two-Tower + Stage 2 DeepFM 추천 모델 실험 리포트

기준일: 2026-11-12 · 원자료: `docs/stage12_eval.json` (지표·응답 시간·상한 분석·API 벤치마크), `artifacts/recommendation/stage12/*_report.json`

## 요약

| 지표 | 목표 | 결과 | 판정 | 베이스라인 대비 |
|---|---|---|---|---|
| Recall@300 (Two-Tower + FAISS HNSW) | ≥ 0.30 | **0.2315** | 미달 | 인기도 0.0758 대비 **+205%** |
| 후보 300개 추출 p95 (User Tower + FAISS) | ≤ 100 ms | **3.48 ms** | 달성 | |
| Ranking AUC (DeepFM, CTR) | ≥ 0.70 | **0.7482** | 달성 | Two-Tower 점수 0.6257 대비 +19.6% |
| HitRate@50 (Stage 1→2→3 전체) | ≥ 0.20 | **0.2703** | 달성 | 인기도 0.0995 대비 **+172%** |
| NDCG@50 (전체) | ≥ 0.08 | **0.0243** | 미달 | 인기도 0.0083 대비 +193% |
| Coverage (전체) | ≥ 0.20 | **0.2759** | 달성 | 인기도 0.0007 |
| 추천 API 전체 p95 (Stage 1+2+3, top 10) | ≤ 200 ms | **17.5 ms** | 달성 | |

- **Recall@300 미달 원인:** 이 시뮬레이터 로그에서는 사용자가 test 기간에 볼 **대분류 버킷 분포를 정답으로 미리 알려줘도** 같은 구조의 점수로 Recall@300 = 0.2667이다(4-2절). test 기간에 어떤 소분류를 보는지는 세션마다 거의 무작위라서 이력으로 예측할 수 없다. 0.30은 이 데이터에서 이력 기반 모델이 넘기 어려운 수치로 판단한다.
- **Two-Tower는 단순 규칙(0.2502)보다 0.019 낮다.** 개선 여지가 있다(6절).
- Stage 1/2가 연결되면서 Stage 3 목표 중 HitRate@50과 Coverage가 처음으로 달성됐다(Stage 3 리포트의 인기도 후보 기준 0.1127 / 0.0975).

## 1. 파이프라인

```
사용자 이력(최근 30클릭) + 프로필(페르소나, 선호 카테고리, 장기 카테고리/브랜드/버킷 비율, 평균 가격)
        │  User Tower (PyTorch) → 64차원 단위벡터                         ~2 ms
        ▼
FAISS HNSW (상품 70,831개, Item Tower로 오프라인 사전 계산) → 후보 300개
        + 최근 클릭 상품 중 후보에 없는 최대 20개 (재방문 신호)
        │  DeepFM (FM 1·2차 + DNN, CTR/CVR 두 헤드)                      ~2 ms
        ▼
Stage 2 출력 [(product_id, p_ctr)] ─→ Stage 3 Reranker                     ~7 ms
        ▼
recommendations[] + pipeline_latency + session_context
```

| 요구사항 | 구현 | 코드 |
|---|---|---|
| User Tower: 최근 행동 시퀀스 + 프로필 | 최근 30클릭의 상품 속성 임베딩 + 이벤트 유형(view/cart/purchase) 가중 평균(전체·최근 5개 두 가지), 페르소나·선호 카테고리 임베딩, 이력 길이, 장기 프로필 148차원 | `two_tower.py` `UserTower` |
| Item Tower: 카테고리·속성·가격 | 대/중/소분류, 버킷, 브랜드, 상품군, 소재군, 색상, 신상/인기 플래그, 가격·할인·신선도·인기점수 구간 + 상품 ID 임베딩 | `two_tower.py` `ItemTower` |
| Item 임베딩 Offline 사전 계산 → FAISS | 학습 후 전 상품 벡터를 `item_embeddings.npy`로 저장, HNSW(M=32, efSearch=512) 인덱스 `items.index` | `two_tower.py` `write_index` |
| 후보 300개 100 ms 이내 | p95 3.48 ms (10회 워밍업 + 100회) | `evaluate_stage12.py` |
| Ranking: User/Item/Cross/Context 피처 | 4-1절 표 | `deepfm.py` `FeatureBuilder` |
| CTR/CVR 예측 | 공유 임베딩 위에 CTR 헤드(클릭=view/cart/purchase), CVR 헤드(cart/purchase) | `deepfm.py` `DeepFM` |
| Negative Sampling | Two-Tower: 양성 1개당 랜덤 네거티브 4개(+배치 내 네거티브, logQ 보정) / Ranking: 노출(=후보) 후 비클릭 | `two_tower.train`, `deepfm.build_impressions` |
| 설정 분리 | 하이퍼파라미터·경로 전부 `config/stage12.yaml` | |

## 2. 평가 프로토콜

| 항목 | 값 |
|---|---|
| 데이터 | `data/behavior_logs.csv` 1,000,000행 (클릭 이벤트 888,780행), 사용자 10,000명, 카탈로그 `data/product_features.csv` 70,831개 |
| 분할 | 전체 행 시간순 train / valid / test = 8 / 1 / 1. `rerank.split_events`와 같은 규칙 (train ~ 2025-03-13, valid ~ 03-22, test 03-23 ~ 03-31) |
| Two-Tower 학습 | train 구간. 사용자-상품 첫 상호작용 592,406개를 앵커로, 앵커 이후 20클릭 중 하나를 양성으로 매 배치 샘플링 |
| DeepFM 학습 | 문맥 = train 스냅샷, 노출 = 그 시점 Two-Tower 후보, 라벨 = valid 기간 클릭. 사용자 10%(796명)는 조기 종료용으로 분리. 누수 없음 |
| 평가 시점 | 문맥(이력·인기도) = train+valid, 정답 = test 기간에 view/cart/purchase한 상품 집합 (사용자당 평균 9.34개) |
| 평가 사용자 | test 기간 활동 사용자 8,054명 (`docs/stage3_report.md` 평가와 동일) |
| 시드 | random_seed = 42 (네거티브·타깃 샘플링, 가중치 초기화, 검증 사용자 샘플링). 분할은 시간 기준이라 무작위 요소 없음 |

### 지표 정의 (`src/evaluation/metrics.py`와 동일)

- **Recall@300** = |FAISS 상위 300 ∩ 정답| / |정답| → 사용자 평균
- **AUC** = 무작위 양성의 점수 > 무작위 음성의 점수일 확률 (Mann-Whitney U, 동점 0.5). test 노출(사용자별 후보 320개 이내, 2,554,590행) 기준
- **HitRate@50 / NDCG@50 / Coverage** = `docs/stage3_report.md` 2절과 같음. Coverage 분모 = 70,831

## 3. 결과

### 3-1. Stage 1 후보 생성 (test, 8,054명)

| 방법 | Recall@300 |
|---|---|
| 인기도 상위 300 (베이스라인) | 0.0758 |
| Two-Tower + 정확 검색(Flat) | 0.2317 |
| **Two-Tower + FAISS HNSW** | **0.2315** (HNSW 손실 0.0002) |
| 랭킹 후보 풀 (HNSW 300 + 최근 클릭 ≤20) | 0.2389 |

페르소나별 Recall@300: 가성비추구 0.356 · 실용주의자 0.293 · 트렌드세터 0.209 · 신중탐색 0.113 · 충동구매 0.101 · 브랜드충성 0.085.
속성 선호가 뚜렷한 페르소나(할인 / 인기 / 신상)는 잘 맞히고, 속성 선호가 약한 페르소나는 낮다.

### 3-2. Stage 2 Ranking (test 노출 2,554,590행, CTR 0.57%)

| 점수 | AUC (CTR) |
|---|---|
| 인기도 | 0.6803 |
| Two-Tower 내적 | 0.6257 |
| **DeepFM** | **0.7482** |
| DeepFM, FAISS 후보만 (최근 클릭 추가분 제외) | 0.7424 |
| DeepFM CVR 헤드 (cart/purchase) | 0.7157 |

### 3-3. 추천 목록 (k = 50)

| 전략 | HitRate@50 | NDCG@50 | Coverage | HitRate 개선율 (인기도 대비) |
|---|---|---|---|---|
| 인기도 (베이스라인) | 0.0995 | 0.0083 | 0.0007 | — |
| Two-Tower 단독 (Ranking 없이 후보만, 베이스라인) | 0.2482 | 0.0211 | 0.0380 | +149% |
| Two-Tower + DeepFM | **0.2832** | **0.0275** | 0.2115 | +185% |
| **Two-Tower + DeepFM + Stage 3 (최종)** | 0.2703 | 0.0243 | **0.2759** | +172% |

- DeepFM은 Two-Tower 단독 대비 HitRate@50 +14.1%, NDCG@50 +30.3%.
- Stage 3는 HitRate@50을 0.013 낮추고 Coverage를 0.21 → 0.28로 올린다(탐색 슬롯 2, 신상품 1, 세션 가중 0.6). 규칙 검증: 같은 카테고리 최대 연속 2, 위반 0건, 목록당 탐색 슬롯 2.0, 신상품 1.0.

## 4. 실험 과정

### 4-1. DeepFM 입력 피처 (sparse 27개 필드 + dense 8개)

| 구분 | 피처 |
|---|---|
| User | 페르소나, 선호 카테고리, 이력 길이 구간 |
| Item | Item Tower와 같은 14개 필드, 인기도(문맥 구간의 가중 이벤트 수) 구간 |
| Cross | 사용자 이력 중 해당 상품 소분류·브랜드·버킷 비율 구간, 상품 가격 / 사용자 평균 가격 비율 구간, 이미 본 상품 여부 |
| Context | Two-Tower 순위 구간, 직전 클릭과 소분류 일치 여부, 최근 5클릭 중 같은 소분류 비율, 마지막 활동 후 경과일 |
| Dense | Two-Tower 점수, log 인기도, 위 비율 3종, 세션 소분류 비율, log 가격 비율, log 이력 길이 |

### 4-2. Recall@300 상한 분석 (`evaluate_stage12.ceiling`, 결과는 `docs/stage12_eval.json`의 `ceiling`)

모든 상품을 `카테고리 가중치 × 페르소나별 속성(신상/인기/할인) 선호도 × (1 + 인기도)`로 점수 매겨 상위 300개를 뽑는다(브랜드충성은 브랜드 비율 추가). 카테고리 가중치만 바꾼다.

| 카테고리 가중치 | Recall@300 | 비고 |
|---|---|---|
| 사용자 이력의 소분류 비율 | 0.2395 | 규칙 |
| 사용자 이력의 버킷 비율 × 전체 P(소분류 \| 버킷) | **0.2502** | 규칙 (최고) |
| **test 기간 버킷 비율(정답)** × P(소분류 \| 버킷) | **0.2667** | 오라클, 모델 아님 |
| test 기간 소분류 비율(정답) | 0.4103 | 오라클, 모델 아님 |
| (참고) Two-Tower | 0.2315 | |

- 사용자 이력의 **소분류** 비율이 **버킷** 비율보다 나쁘다 → 시뮬레이터는 선호 버킷 안에서 소분류를 세션마다 거의 무작위로 고른다.
- 다음 9일의 버킷을 정확히 알아도 0.267이므로, 0.30은 test 기간 소분류를 예측해야만 넘을 수 있다. 이력에는 그 정보가 거의 없다.
- 분석 중 확인한 시뮬레이터 특성: 트렌드세터 클릭의 88%가 신상품(카탈로그 20%), 실용주의자 80%가 인기 상품, 가성비추구는 할인 상위 구간에 3배 집중, 브랜드충성은 상위 1개 브랜드에 78%. test 클릭 중 이전에 본 상품 재방문 3.2%, 선호 버킷 안 상품 78%.

### 4-3. Two-Tower 변경 이력 (valid Recall@300, valid 사용자 3,000명)

| 버전 | 변경 | valid | test |
|---|---|---|---|
| v1 | 최근 30클릭 평균 + 페르소나, 다음 클릭 예측, 배치 내 네거티브(보정 없음) | 0.1435 (2 epoch에서 중단) | — |
| v2 | + 장기 프로필 148차원, 배치 내 네거티브 logQ 보정 | 0.2341 | — |
| v3 | + 학습 타깃을 "다음 20클릭 중 하나"로 (평가가 다음 9일 전체이므로) | 0.2360 | 0.2306 |
| **v4** | + 할인율 구간 버그 수정 (0%와 5~19% 할인이 같은 구간이던 문제) | **0.2386** | **0.2315** |

- 배치 내 네거티브는 인기 상품일수록 자주 뽑혀 인기 상품 점수를 깎는다. 이 데이터는 상품 인기도가 test 클릭과 강하게 상관(train↔test 상품별 클릭 수 Pearson 0.85)하므로 logQ 보정이 중요했다.
- 4 epoch 중 valid 최고 epoch를 저장한다(v4는 1 epoch). epoch당 약 3분(CPU).

### 4-4. 최근 클릭 후보 추가 (Stage 2 후보 풀)

`docs/stage3_report.md` 7-1절의 제안(재방문 신호)을 반영했다. FAISS 상위 300개에 없는, 사용자의 최근 클릭 상품을 최대 20개 추가해 DeepFM이 함께 점수를 매긴다. **Recall@300은 FAISS 300개만으로 보고한다.**

| | AUC | TT+DeepFM HitRate / NDCG / Coverage | 최종(+Stage 3) HitRate / NDCG / Coverage |
|---|---|---|---|
| 추가 없음 | 0.7420 | 0.2707 / 0.0255 / 0.0874 | 0.2599 / 0.0228 / 0.1710 |
| **최근 클릭 ≤20 추가 (채택)** | 0.7482 | 0.2832 / 0.0275 / 0.2115 | 0.2703 / 0.0243 / 0.2759 |

Coverage가 크게 오른 이유: 최근 클릭 상품은 사용자마다 다르므로 목록이 개인화된다.

## 5. 응답 시간

측정: 워밍업 10회 후 100회, 단건, p95. Windows 11, Intel Core 10세대(Comet Lake) 12 논리 코어, CPU 전용, torch 스레드 4, Python 3.13.9, torch 2.15.0+cpu, faiss-cpu 1.16.0.

| 구간 | p50 | p95 |
|---|---|---|
| 후보 생성 (User Tower + HNSW 300 + 최근 클릭) | 2.43 ms | 3.48 ms |
| Ranking (피처 생성 + DeepFM, 후보 ≤320) | 2.15 ms | 2.67 ms |
| Stage 3 재정렬 (top 10) | 7.88 ms | 9.81 ms |
| `RecommendationService.recommend` 전체 (`stage12_eval.json`의 `api_bench`) | 14.3 ms | 17.5 ms |

HTTP·Redis 오버헤드는 포함하지 않았다(API 서버 구간). 서비스 기동(모델·인덱스·사용자 상태 로드) 약 10.5초. 같은 PC에서도 측정 시점의 부하에 따라 p95가 수 ms 달라진다(이전 측정 후보 2.4 ms, 전체 13.9 ms).

## 6. 한계와 다음 단계

1. **Recall@300 0.2315 < 0.30.** 4-2절 근거로 데이터 한계가 크다. 다만 Two-Tower가 규칙(0.2502)보다 낮으므로, 사용자 버킷 비율을 더 직접 쓰는 구조(예: 버킷별 사용자 벡터 혼합)나 출력 차원·온도 조정으로 0.25 근처까지는 올릴 여지가 있다.
2. **NDCG@50 0.0243 < 0.08.** 정답이 사용자당 평균 9.3개인데 상위 50 안에는 평균 약 0.5개만 들어온다(Recall@50 0.052). 후보 품질이 오르지 않는 한 순서 조정만으로는 어렵다.
3. **Stage 3 기본값 재조정 필요.** Stage 3 기본값은 인기도 후보로 튜닝됐다. 개인화 후보에서는 Stage 3가 HitRate를 0.013 깎는다. `stage3_experiments`의 그리드를 이 후보로 다시 돌리면 좋다.
4. Two-Tower는 train만으로 학습했다(DeepFM 학습 데이터 누수 방지). 서비스용으로는 train+valid로 재학습할 수 있다.
5. 시뮬레이터 로그가 바뀌면 `encoders.json`을 지우고 1~3단계(8절)를 다시 실행해야 한다.

## 7. 산출물

| 파일 | 내용 |
|---|---|
| `config/stage12.yaml` | 데이터 경로, 피처, Two-Tower / FAISS / DeepFM / 평가 설정 |
| `src/recommendation/stage12_data.py` | 상품 인코딩, 시간 분할, 사용자 스냅샷(오프라인·온라인 공용) |
| `src/recommendation/two_tower.py` | Two-Tower 모델·학습·FAISS 인덱스·`CandidateGenerator` |
| `src/recommendation/deepfm.py` | DeepFM 모델·피처·학습·`Ranker` |
| `src/recommendation/pipeline.py` | `RecommendationService` (Stage 1→2→3, API 응답 규격) |
| `src/evaluation/evaluate_stage12.py` | 이 리포트의 모든 지표와 4-2절 상한 분석, `data/recommendation_metrics.json` 작성 |
| `tests/test_stage12.py` | 합성 데이터로 학습·검색·응답 필드·한 줄 재학습 검증 (8개) |
| `artifacts/recommendation/stage12/` | 학습된 모델, 상품 벡터, FAISS 인덱스, 사용자 상태, 학습 기록 (약 87MB) |

## 8. 재현 방법

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-recommendation.txt
python -m src.recommendation.two_tower train          # ~13분 (CPU) → two_tower.pt, items.index
python -m src.recommendation.deepfm train             # ~5분 → deepfm.pt
python -m src.recommendation.pipeline build-state     # 서비스용 사용자 상태 (전체 로그)
python -m src.evaluation.evaluate_stage12             # ~5분 → docs/stage12_eval.json, data/recommendation_metrics.json
python -m src.recommendation.pipeline bench --save    # API 응답 시간 → docs/stage12_eval.json (api_bench)
python -m pytest tests/test_stage12.py -q
# 위 학습~평가를 한 번에 (CT 재학습용): python -m src.recommendation.pipeline retrain
```
