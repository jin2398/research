# RE:SEARCH — 문석민 팀원 검색 모듈 연결 가이드

> 제공 ZIP 파일은 ① 검색 소스 업데이트(18KB) ② 완성 FAISS HNSW 검색 인덱스(약 134MB) 입니다. **모델 가중치나 상품 사진 파일은 두 ZIP에 없습니다.** 이 패키지는 코드를 받으면 곧바로 실행할 수 있도록 검색 API 브리지와 쇼핑 UI 호출 기능을 준비한 버전입니다.

## 1. 두 압축파일 복사 위치 (중요)

**`RESEARCH/` 폴더를 프로젝트 루트**로 사용합니다. ZIP을 임시 폴더에 각각 해제하고, **ZIP 내부의 폴더를 루트에 병합**하세요. `01_검색업데이트` 같은 새 상위 폴더를 추가하지 마세요.

| ZIP | ZIP 내부 경로 | 프로젝트 내 최종 경로 |
|---|---|---|
| `01_검색업데이트.zip` | `src/search/clip_faiss.py` | `RESEARCH/src/search/clip_faiss.py` |
| | `src/search/query_text.py` | `RESEARCH/src/search/query_text.py` |
| | `src/search/text_fusion.py` | `RESEARCH/src/search/text_fusion.py` |
| | `src/search/evaluate_text_fusion.py` | `RESEARCH/src/search/evaluate_text_fusion.py` |
| | `config/search_full.yaml` | `RESEARCH/config/search_full.yaml` |
| | `docs/search_text_fusion.json` | `RESEARCH/docs/search_text_fusion.json` |
| | `README_TEXT_FUSION.md` | `RESEARCH/README_TEXT_FUSION.md` |
| `02_전체검색인덱스.zip` | `artifacts/search/clip_hnsw_full/products.index` | `RESEARCH/artifacts/search/clip_hnsw_full/products.index` |
| | `artifacts/search/clip_hnsw_full/products.jsonl` | `RESEARCH/artifacts/search/clip_hnsw_full/products.jsonl` |
| | `artifacts/search/clip_hnsw_full/manifest.json` | `RESEARCH/artifacts/search/clip_hnsw_full/manifest.json` |

**주의:** `01_검색업데이트.zip`의 `config/search_full.yaml`에는 다른 PC의 `E:/research_image/images` 경로와 `model_local_files_only: true`가 있어요. 복사한 다음 **아래 두 항목을 반드시 수정**하세요.

```yaml
image_root: data/product_images
model_local_files_only: false
```

검색 API는 첫 로드에 `SEARCH_OFFLINE_ONLY=1`을 설정하지 않는 한 모델 다운로드를 허용하도록 작성했으므로 ZIP에 들어 있던 `true`를 그대로 둔 경우에도 자동 연결할 수 있습니다. 그러나 팀원 실험 재현과 설정 일관성을 위해 위와 같이 고쳐 두는 것을 권장합니다.

- 상품 이미지를 이미 가지고 있다면 `SEARCH_IMAGE_ROOT` 환경변수에 이미지가 담긴 폴더를 지정하세요. 예: `$env:SEARCH_IMAGE_ROOT='D:\\research_image\\images'`. 없으면 이미지 카드에는 기본 이미지가 표시됩니다.
- **모델 가중치가 없는 첫 실행**에는 인터넷이 필요합니다. `model_local_files_only: false` 설정 또는 브리지의 기본 설정으로 Hugging Face에서 `openai/clip-vit-base-patch32` 가중치를 다운로드합니다. 처음은 느릴 수 있지만 전체 인덱스를 다시 만드는 과정은 아닙니다.
- `products.jsonl`은 상품 ID·이름·가격·카테고리 등 메타데이터가 있어 기본 검색에 별도 `products.csv`는 필요하지 않습니다. 다만 원본 평가 스크립트를 *재실행*하려면 원본 카탈로그 CSV, 분할 데이터, 이미지 및 기존 평가 모듈이 추가로 필요합니다.

## 2. 설치와 실행 (Windows PowerShell)

Python 3.11~3.12 권장. `RESEARCH/` 폴더에서:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-search-bridge.txt
python check_search_setup.py
```

**터미널 A** (검색 API 서버, 그대로 열어 두기):

```powershell
python -m uvicorn src.serving.search_bridge:app --host 127.0.0.1 --port 8000
```

**터미널 B** (같은 가상환경, 화면 실행):

```powershell
python -m streamlit run src/dashboard/app.py
```

화면: <http://localhost:8501> · 검색 API 상태: <http://127.0.0.1:8000/api/search/status>

검색할 때 자동으로 `http://127.0.0.1:8000/api/search`에 요청하고, 성공하면 실제 검색 결과를 상품 카드로 표시합니다. API가 켜져 있지 않거나 파일이 빠졌으면 UI에는 그 상태를 알리고 **샘플 결과를 실제 검색인 것처럼 표시하지 않습니다.**

### 3. 검색 API 규격

- 텍스트: `POST /api/search` JSON `{"query":"베이지 코트","mode":"text","top_k":20}`
- 이미지: `POST /api/search` multipart `file=<이미지 파일>`, `mode=image`, `top_k=20`
- 하이브리드: multipart `file=<이미지>`, `query=베이지 코트`, `mode=hybrid`, `top_k=20`
- 상태 확인: `GET /api/search/status`

텍스트 검색은 상품 메타데이터를 이용한 로컬 BM25 + CLIP + 속성 매칭으로 구성돼 있고, 이미지/하이브리드는 팀원이 제공한 CLIP+FAISS 인덱스를 사용합니다. **이 통합 BM25 구현은 팀원의 평가 코드와 다르므로 기존 보고서 지표를 이 라이브 검색의 실측 성능이라고 주장하지 않습니다.**

### 4. 예상 이슈

- `ModuleNotFoundError: No module named src.search.clip_faiss`: 01 업데이트 압축 위치 확인.
- `products.index` 누락: 02 인덱스 압축 위치 확인. 재구축 명령은 필요 없습니다.
- `CLIP 모델을 로드할 수 없습니다`: 패키지 설치와 Hugging Face 다운로드/캐시 확인. Python 3.11~3.12 권장.
- Streamlit은 보이는데 API 검색이 안 된다면 **두 터미널 모두 켜져 있는지** 확인하세요.
- 사진이 전부 기본 이미지: 02 ZIP에 사진이 없기 때문입니다. 이미지 폴더를 별도로 확보해 `SEARCH_IMAGE_ROOT`에 지정하세요.
- 일반 UI 화면은 검색 서버 없이도 데모 상품을 볼 수 있지만, **실제 검색 결과는 검색 서버가 켜진 상태에서만** 표시합니다.
