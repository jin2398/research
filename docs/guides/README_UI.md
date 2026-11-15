> **새로운 검색 모듈 연결:** 문석민 팀원 ZIP 2개를 설치할 때는 먼저 [`README_SEARCH_SETUP.md`](README_SEARCH_SETUP.md)를 확인하세요. ZIP 안의 `src`, `config`, `docs`, `artifacts` 폴더를 프로젝트 루트에 병합하면 됩니다.

# RE:SEARCH — 패션 검색·추천 UI v1

> 기존 `codisay.zip`을 기반으로 만든 **Streamlit 화면 시제품**입니다. RE:SEARCH 디자인 목업의 색감과 구성(다크 사이드바, 베이지 패션 배너, 검색창, 상품 카드, 추천 패널, KPI)을 재현했습니다.

## 빠른 화면 확인 (설치 없이)

`codisay_preview.html` 파일을 Chrome/Edge로 열면 UI 디자인을 바로 확인할 수 있습니다. 검색, 카테고리 필터, 정렬, 찜하기, 메뉴 전환이 브라우저에서 동작합니다. 이 HTML은 독립형 **화면 시연 파일**이며, 대시보드/A/B·CT 실제 데이터 처리는 아래 Streamlit 앱에서 실행합니다. 디자인 스크린샷은 `docs/ui_preview.png`에 있습니다.

## 화면 일치도 수정 (v1.1)

기존 Streamlit 화면은 **HTML 미리보기와 별개로 구성한 Streamlit 위젯**이었고, 테마/레이아웃 때문에 크게 달랐습니다. v1.1에서는 `src/dashboard/app.py`가 HTML 미리보기 자체를 Streamlit 컴포넌트에 표시합니다. 따라서 홈 화면의 이미지 배너·사이드바·상품 카드·색상·검색 UI는 **HTML과 같은 소스**를 사용합니다.

- 홈, 텍스트 검색, 정렬, 상품 필터, 찜하기 등은 HTML 시연 UI에서 동작합니다.
- 왼쪽 `대시보드`와 `실험관리`는 기존 Streamlit 분석 화면을 **새 탭으로 여는 링크**입니다. (Streamlit의 HTML iframe 보안 제한으로 동일 탭 이동은 사용하지 않습니다.)
- 수정 전 순수 Streamlit 화면은 `src/dashboard/app_native.py`에 보존했습니다. 별도로 실행할 수 있습니다.
- 예전과 달리 Streamlit 홈 화면은 HTML iframe을 사용하므로 HTML 내부의 프론트엔드 상태는 페이지 이동 시 초기화됩니다.
- **중요:** 시연용 키워드 검색/AI 추천은 실제 AI 호출이 아닙니다. 외부 API 호출 및 더 풍부한 사용자 조작은 `app_native.py`에 보존된 별도 개발용 화면에 남아 있으며, HTML형 프론트엔드와 API의 완전한 연결은 후속 작업입니다.
- Streamlit 자체는 이 제작 환경에서 실행할 수 없어 iframe 통합을 브라우저에서 실측 검증하지는 못했습니다. 대신 HTML 원본과 생성된 홈 화면의 정합성 및 단위 테스트를 점검했습니다.

## 1. 실행 (Windows PowerShell)

```powershell
cd <압축을 푼 폴더>
python -m pip install -r requirements-dashboard.txt
python -m streamlit run src/dashboard/app.py
```

브라우저에서 **http://localhost:8501** 에 접속합니다.

`run_windows.bat`를 더블클릭해 실행해도 됩니다(의존성은 먼저 설치해야 함).

**Streamlit 설치 전에도 디자인을 확인하려면** `codisay_preview.html`을 더블클릭하거나 `python run_storefront.py`를 실행해 접속할 수 있습니다. 브라우저용 미리보기와 Streamlit의 홈 디자인은 동일한 HTML로 생성됩니다.

## 2. 화면 구성

| 메뉴 | UI v1 기능 | 데이터 출처 |
|---|---|---|
| 홈 | 상단 패션 배너, 검색 패널, 상품 카드, 추천 패널, 성능 요약 | 샘플 상품 / 기존 실제 평가 보고서 |
| 멀티모달 검색 | 텍스트 검색, 이미지 업로드, 검색 방식 선택, 카테고리/가격 정렬 | **샘플 키워드 검색**; 실제 CLIP 추론 미연결 |
| 개인화 추천 | 스타일 프로필 변경, 추천 카드, 찜하기 | **단순 규칙 기반 UI 시연**; 실제 추천 모델 미연결 |
| 대시보드 | 기존 BM25·CLIP 파일럿·추천 평가·세션 통계 | 기존 `docs/*.json` / 추후 추천 평가 JSON |
| 실험관리 | 기존 A/B 수동 시뮬레이션·CSV 분석, CT 모니터링 | 기존 코드 및 사용자 제공 실험 데이터 |

**중요:** 시연용 상품, 이미지, 가격, 평점, 추천 순위는 실제 카탈로그나 AI 예측 결과가 아닙니다. 화면에 데모라는 표시가 있습니다. 검색·추천 지표는 `docs` 및 `data`의 실제 보고서가 있을 때만 표시하며, 예시 JSON 값으로 결과를 채우지 않습니다.

## 3. 팀원 데이터 연결

### 3-1. 상품 CSV

실제 카탈로그 데이터가 들어오면 프로젝트 루트에 `data/products.csv`로 저장합니다. 요구 열은 다음과 같습니다.

```csv
product_id,name,category,price,image_path,color,style
P1001,베이지 코트,아우터,89000,data/images/P1001.jpg,베이지,미니멀
P1002,검정 슬랙스,하의,54000,data/images/P1002.jpg,블랙,미니멀
```

- `product_id`, `name`, `category`는 **필수**입니다. 그 외는 선택 사항입니다.
- `image_path`는 프로젝트 루트 기준 상대 경로입니다. 파일이 없으면 **이미지 없음**으로 표시합니다.
- CSV 파서는 처음 1,500행을 읽으며, HTML형 홈 화면은 성능을 위해 이 중 최대 48개만 표시합니다. 서비스 규모로 확장할 때는 백엔드의 페이지네이션이 필요합니다.
- 이 CSV를 넣으면 샘플 카탈로그 대신 CSV 카탈로그로 전환되며, 검색은 **CSV 키워드 기반**으로만 작동합니다. CLIP/FAISS 검색으로 오인하면 안 됩니다.
- 팀원의 `products.csv`에 필드명이 다르면 합의된 규격으로 변환해야 합니다.

### 3-2. 검색·추천 API (선택)

FastAPI의 계약이 정해진 다음 환경 변수를 설정해 주세요.

```powershell
$env:CODISAY_API_BASE_URL="http://localhost:8000"
python -m streamlit run src/dashboard/app_native.py
```

실제 모델 API 연결 스위치는 **기존 원본형 Streamlit 화면인 `src/dashboard/app_native.py`에만 구현되어 있습니다.** HTML형 홈 화면에는 아직 연결되지 않았습니다. API 규격은 다음과 같이 준비되어 있습니다.

- `POST /api/search` JSON: `{"query":"베이지 코트", "mode":"text", "top_k":20}` (텍스트 검색만 지원)
- `GET /api/recommend?user_id=user_001&k=20`
- 응답은 `{"results":[{"product_id":"P1001","name":"베이지 코트","price":89000}]}` 형태 또는 `products`, `recommendations`, `items` 배열이 포함된 형태를 지원합니다.

**주의:** 실제 팀원 API의 입력/출력 스키마가 달라질 수 있으므로 `src/dashboard/api_adapter.py`를 그 규격에 맞게 수정해야 합니다. 이미지·하이브리드 검색 업로드 규격은 아직 미합의이므로 **미연결**입니다. API 호출에 실패하면 오류를 표시하며, 성공한 것처럼 샘플 상품으로 바꾸지 않습니다.

### 3-3. 모델 평가 JSON

- `docs/search_baseline.json`, `docs/search_baseline_pilot.json`, `docs/search_clip_pilot.json`, `docs/search_latency_pilot.json`은 기존 파일 그대로 사용합니다.
- 실제 추천 평가 결과는 `data/recommendation_metrics.json`에 저장합니다. 기존 README의 스키마를 따릅니다.
- 실제 A/B 실험이 아닌 CSV는 A/B 결과로 주장하지 않으며, 샘플 실험은 **시연 모드**로 구분합니다.
- CT 기준 로그 수 초기화는 원본 `README_INTEGRATION.md`를 참고하세요. `artifacts/ct_state.original_example.json`은 참고용이며 현재 서비스 상태로 사용하지 않습니다.

## 4. 원본 기능 보존

- 원래 `src/dashboard/app.py` 코드는 `src/dashboard/legacy_dashboard.py`에 백업했습니다.
- 기존 평가/실험 로직은 `src/dashboard/analytics_views.py`에 함수 단위로 나누어 새 `대시보드` / `실험관리` 메뉴에서 호출합니다.
- `src/dashboard/integration.py`, `ab_test.py`, `ab_experiment.py`, `ct_monitor.py` 및 기존 테스트의 비즈니스 로직은 수정하지 않았습니다.

## 5. 테스트

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

기존 26개 테스트 + 새 카탈로그/API 스키마 테스트 4개를 포함합니다. 독립 HTML 미리보기는 Chromium 브라우저에서 상품 카드 7개 표시, 「베이지 코트」 검색 후 결과 1개 표시, 대시보드 메뉴 전환을 확인했습니다.

## 6. 구현 범위 / 한계

- 현재 구현 완료: HTML과 같은 홈 디자인, 5개 메뉴 탐색(검색/추천은 시연용 안내 페이지), 7개 데모 상품, 홈 화면 검색·정렬, 이미지 업로드 및 미연결 안내, 찜하기(페이지 내 유지), 기존 분석 화면 연결, 외부 CSV·JSON 입력 구조. 선택형 API 어댑터는 `app_native.py`에 보존됨.
- 앞으로 필요한 작업: 실제 상품 이미지 경로 연결, API 스키마 검증, CLIP 이미지 검색 실연동, Two-Tower/DeepFM 실제 추천, 사용자 로그인 및 데이터베이스, 반응형 최적화/브라우저 실측 확인.
- 브라우저/Streamlit 실행 확인: 독립 HTML과 Streamlit용 HTML iframe을 Chromium에서 렌더링하여 레이아웃, 7개 상품 표시, 키워드 검색을 확인했습니다. 다만 이 환경에는 Streamlit 패키지가 없어 **실제 Streamlit 서버 구동 화면은 검증하지 못했습니다**. Python 문법 검사와 단위 테스트도 수행했습니다.
