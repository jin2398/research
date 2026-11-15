# RE:SEARCH 실행 가이드

Docker Desktop이 없어도 실행할 수 있습니다. 내 환경에 맞는 방법을 하나 고르세요.

| 방법 | 필요한 것 | 되는 기능 |
|---|---|---|
| [A. Docker Desktop](#a-docker-desktop) | Docker Desktop | 전부 (명령 한 줄) |
| [B. Docker Desktop 없이 Docker 사용](#b-docker-desktop-없이-docker-사용) | OrbStack · Colima · WSL2 등 | 전부 (A와 같은 명령) |
| [C. Docker 없이 Python으로 실행](#c-docker-없이-python으로-실행) | Python + Redis | 전부 (터미널 3~4개) |
| [D. Redis 없이 최소 실행](#d-redis-없이-최소-실행) | Python | 검색·추천·대시보드 (행동 기록·세션 반영 제외) |
| [E. 서버 없이 모델만 확인](#e-서버-없이-모델만-확인) | Python | 명령줄에서 추천·검색 결과 출력 |

---

## 공통 준비: 프로젝트 폴더

어떤 방법이든 아래 폴더가 모두 있어야 합니다. 코드만 있으면 실행되지 않습니다.

| 폴더 | 크기 | 내용 |
|---|---|---|
| `src/`, `config/`, `scripts/` | 작음 | 코드와 설정 |
| `data/` | 약 180MB | 상품 정보, 행동 로그 100만 건 |
| `artifacts/recommendation/` | 약 90MB | 학습된 추천 모델 (Two-Tower, DeepFM, GRU) |
| `artifacts/search/clip_hnsw_full/` | 약 175MB | CLIP 검색 인덱스 (상품 70,521개) |
| `artifacts/search/model_cache/` | 약 1.1GB | CLIP 모델. **없으면 첫 실행 때 인터넷으로 자동 다운로드** |

모든 명령은 **프로젝트 폴더(`docker-compose.yml`이 있는 위치)에서** 실행합니다.

---

## A. Docker Desktop

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/)을 설치하고 실행합니다.
2. Settings → Resources → Memory를 **4GB 이상**으로 설정합니다 (과제 권장 8GB).
3. 실행:

```bash
docker compose up --build     # 처음 한 번은 5~10분 (이후에는 docker compose up)
```

`api-server`가 `(healthy)`가 되면 준비 완료입니다 (`docker compose ps`로 확인, 모델 로딩 30초~1분).

| 주소 | 내용 |
|---|---|
| http://localhost:8000/docs | API (Try it out → Execute로 직접 호출) |
| http://localhost:8501 | 대시보드 |

자주 쓰는 명령:

```bash
docker compose down                  # 끄기
docker compose up -d --build         # 코드를 고친 뒤 다시 켜기 (--build 필수)
docker compose restart simulator     # config/simulator.yaml 수정 반영
docker compose restart api           # config/stage12.yaml 수정 반영
docker logs -f simulator             # 시뮬레이터 로그 보기
```

---

## B. Docker Desktop 없이 Docker 사용

`docker compose` 명령만 쓸 수 있으면 됩니다. 아래 도구 중 하나를 설치한 뒤 **A의 3번부터 그대로** 진행하세요.
(이 프로젝트는 Docker Desktop에서만 실제로 테스트했습니다. 아래 도구도 같은 `docker compose`를 쓰므로 동작해야 하지만, 처음 실행할 때 확인이 필요합니다.)

### macOS

```bash
# 방법 1: OrbStack (앱 설치형, 가볍고 빠름)
brew install orbstack          # 설치 후 OrbStack 앱 실행

# 방법 2: Colima (명령줄 전용, 무료)
brew install colima docker docker-compose
colima start --cpu 4 --memory 6      # 메모리 6GB 할당
```

### Windows (WSL2 + Docker Engine)

```powershell
wsl --install -d Ubuntu        # PowerShell(관리자). 설치 후 재부팅
```

Ubuntu 터미널에서:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER        # 적용하려면 터미널을 닫고 다시 열기
sudo service docker start
cd /mnt/c/Users/<사용자>/Desktop/research-recommenditon-api
docker compose up --build
```

> WSL2는 Windows 메모리의 일부만 씁니다. 부족하면 `C:\Users\<사용자>\.wslconfig`에
> `[wsl2]` 아래 `memory=6GB`를 적고 `wsl --shutdown` 후 다시 실행하세요.
> `/mnt/c/...` 경로는 느릴 수 있어서, 폴더를 WSL 안(`~/`)으로 복사하면 더 빠릅니다.

### Linux

배포판의 Docker Engine과 Compose 플러그인을 설치하면 A와 같습니다.

---

## C. Docker 없이 Python으로 실행

Docker 컨테이너 4개가 하던 일을 터미널 여러 개로 나눠 실행합니다.

### 1) Python 환경 (처음 한 번)

Python **3.11 ~ 3.13** (3.12, 3.13에서 테스트됨)

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

> torch를 먼저 CPU 전용으로 설치해야 수 GB짜리 GPU 버전이 설치되지 않습니다.
> `Activate.ps1` 실행이 막히면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 먼저 실행하세요.

### 2) Redis 설치·실행 (처음 한 번)

| OS | 명령 |
|---|---|
| macOS | `brew install redis` → `brew services start redis` |
| Ubuntu / WSL | `sudo apt install redis-server` → `sudo service redis-server start` |
| Windows | Redis는 Windows를 공식 지원하지 않습니다. WSL에 설치(위 줄)하면 Windows에서도 `localhost:6379`로 접속됩니다. |

확인: `redis-cli ping` → `PONG`

Redis 설치가 어려우면 [D. Redis 없이 최소 실행](#d-redis-없이-최소-실행)으로 진행하세요.

### 3) 실행 (터미널 3~4개, 모두 가상환경 활성화 상태)

**터미널 1: API 서버**

```bash
# macOS / Linux
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

```powershell
# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:OMP_NUM_THREADS="4"
python -m uvicorn src.serving.api:app --host 127.0.0.1 --port 8000
```

`[api-server] http://localhost:8000 에서 API 서버 시작`이 나오면 준비 완료입니다.

**터미널 2: 대시보드**

```bash
python -m streamlit run src/dashboard/app.py      # → http://localhost:8501
```

**터미널 3: 시뮬레이터** (선택, API가 켜진 다음에)

```bash
python -m src.simulator                           # 멈추려면 Ctrl+C
```

**터미널 4: 테스트** (선택)

```bash
pytest
python scripts/benchmark_api.py                   # 응답 시간 측정
```

> **`KMP_DUPLICATE_LIB_OK`가 필요한 이유:** API 서버는 검색(torch)과 추천(faiss)을 한 프로세스에서 씁니다.
> macOS와 Windows에서는 두 라이브러리가 각자 OpenMP를 불러와 `OMP: Error #15` 또는
> `libiomp5md.dll already initialized`로 종료될 수 있습니다. Docker(Linux)에서는 필요 없습니다.

---

## D. Redis 없이 최소 실행

C의 1)과 3)만 하고 2) Redis는 건너뜁니다.

| 기능 | Redis 없을 때 |
|---|---|
| 검색 `POST /api/search` | ✅ 정상 |
| 추천 `GET /api/recommend` | ✅ 정상. 저장된 학습 로그 기준으로 추천하고 응답에 `"feature_store": "unavailable"` 표시 |
| 대시보드 | ✅ 정상 |
| 행동 기록 `POST /api/events` | ❌ 503 오류 |
| 세션 클릭 실시간 반영, MAB 피드백 | ❌ 동작 안 함 |
| Feature Store 속도 측정 | ❌ 503 오류 |
| 시뮬레이터 | API 대신 `python -m src.simulator --no-api`로 CSV 로그만 생성 |

---

## E. 서버 없이 모델만 확인

Redis와 서버 없이, 학습된 모델이 동작하는지만 명령줄에서 봅니다. (C의 1) Python 환경만 필요)

```bash
# 추천 (Stage 1→2→3 결과 JSON 출력)
python -m src.recommendation.pipeline demo --user-id U00001 --top-n 10
python -m src.recommendation.pipeline demo --user-id NEW_USER                  # 신규 사용자
python -m src.recommendation.pipeline demo --user-id U00001 --recent 0837810001,0889550002   # 세션 클릭 반영

# 검색 (CLIP + FAISS, 텍스트 또는 이미지)
python -m src.search.clip_faiss --config config/search_full.yaml query --text "black boots"
python -m src.search.clip_faiss --config config/search_full.yaml query --image 사진.jpg

# 시뮬레이터 (CSV 로그만 300건)
python -m src.simulator --no-api --max-events 300
```

---

## 문제 해결

| 증상 | 해결 |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop(또는 OrbStack, Colima)이 켜져 있는지 확인 |
| `address already in use` / 포트 8000·8501·6379 사용 중 | 다른 방법으로 이미 실행 중입니다. `docker compose down` 하거나 다른 터미널의 서버를 종료 |
| `OMP: Error #15`, `libiomp5md.dll already initialized` | C-3)의 `KMP_DUPLICATE_LIB_OK=TRUE` 설정 후 다시 실행 |
| `Segmentation fault` (macOS 테스트 중) | `OMP_NUM_THREADS=1`로 설정 후 다시 실행 |
| 첫 실행이 몇 분째 멈춤 | CLIP 모델(약 600MB) 다운로드 중일 수 있습니다. `artifacts/search/model_cache/`를 팀원에게 받아 넣으면 생략 |
| `CLIP 검색 모델을 로드할 수 없습니다` | 인터넷 연결 확인, 또는 `artifacts/search/` 폴더가 빠지지 않았는지 확인 |
| 추천 API가 503 | `artifacts/recommendation/stage12/` 폴더와 `data/*.csv` 확인 |
| 컨테이너가 갑자기 종료 (`Exited (137)`) | 메모리 부족. Docker 메모리를 4GB 이상으로 |
| 코드를 고쳤는데 그대로임 (Docker) | `docker compose up -d --build`로 다시 빌드 |
| 이미지 검색이 200ms를 넘음 | 알려진 한계 (CPU에서 CLIP 이미지 인코딩). README 5절 참고 |

메모리 사용량(실측): API 서버 약 1.5GB, 대시보드 약 0.1GB, 시뮬레이터 약 0.25GB, Redis 약 0.02GB
