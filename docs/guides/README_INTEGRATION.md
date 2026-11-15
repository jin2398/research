# 팀원 전달 자료 대시보드 연동 패치

## 이 패치가 하는 일

- `multi_session_team_bundle.zip`의 `docs/search_*.json` 검색 평가 결과를 **자동으로 읽습니다**. CLIP 모델 다운로드 또는 전체 FAISS 인덱스 재생성이 필요 없습니다.
- 1,000개 파일럿 CLIP vs 파일럿 BM25 지표, BM25 전체 평가, 검색 처리시간 p95를 별도 구분해 시각화합니다.
- `docs/split_report.json`의 100만 건 **합성** 이벤트/세션 분할 통계를 보여줍니다. 대시보드 실행 시 100만 건 CSV 전체를 읽지 않습니다.
- 실제 추천 모델 지표는 아직 전달 파일에 없으므로 `미제공`으로 표시합니다.
- A/B 테스트는 독립 사용자 Control/Treatment 수를 직접 입력하는 **시연 모드**입니다. 합성 로그에 실험군 필드가 없으므로 실제 A/B 테스트라고 주장하지 않습니다.
- CT는 HitRate@50 지표를 확인하고, 별도 `ct_monitor.py` CLI로 기존 로그 100만 건 초기 기준을 등록할 수 있습니다. 별도 train-command를 지정하지 않는 한 모델 재학습은 수행되지 않습니다.

## 가장 쉬운 실행 방법 (Windows PowerShell)

1. **두 ZIP을 하나로 합쳐 다시 압축할 필요 없습니다.** 먼저 `multi_session_team_bundle.zip`을 원하는 폴더에 풉니다.
2. 이 패치 ZIP을 풀고, 안의 `src`, `ct_monitor.py`, `requirements-dashboard.txt`, `tests` 및 README 파일을 **위 폴더에 같은 폴더 구조로 복사/덮어쓰기** 합니다. 원본의 `data/behavior_logs_enriched.csv`, `docs/*.json`은 그대로 둡니다.
3. 루트 폴더 (`README_TEAM_HANDOFF.md`, `data`와 `docs` 폴더가 보이는 위치)에서 PowerShell을 열고 실행합니다.

```powershell
python -m pip install -r requirements-dashboard.txt
python -m streamlit run src/dashboard/app.py
```

4. 브라우저: <http://localhost:8501>

> Python 3.14에서도 패키지 버전에 따라 설치 문제가 있을 수 있습니다. CLIP 검색 엔진의 CPU/Faiss/Torch 의존성은 별도 Python 3.12 가상환경에서 설치하는 것이 권장됩니다. 대시보드만 볼 때는 CLIP 의존성이 필요 없습니다.

## 정상 화면

- 검색 품질 탭: BM25 전체 MRR@10 = 0.9245, NDCG@10 = 0.8751
- CLIP 비교 탭: 같은 1,000개 인덱스의 CLIP MRR@10 = 0.4773 (모든 test 질의 기준)
- 검색 지연 p95: text 27.8ms, image 298.7ms, hybrid 376.0ms (CPU, **API 오버헤드 미포함**)
- 세션 로그 탭: 합성 로그 = 1,000,000건, 세션 = 159,068건
- 추천 품질 탭: **미제공** (성능 미측정/자료 없음, 0점 아님)
- A/B 테스트 탭: 시연용 기본값으로 p-value 계산
- CT 탭: HitRate 데이터가 없으면 경고를 억지로 띄우지 않습니다.

## 추천 담당자에게 받아야 할 실제 JSON 규격

모델 담당자의 검증을 거친 평가 결과를 `data/recommendation_metrics.json`으로 저장합니다.

```json
{
  "recall_at_300": 0.32,
  "auc": 0.72,
  "hit_rate_at_50": 0.23,
  "ndcg_at_50": 0.09,
  "coverage": 0.25
}
```

**위 값은 형식 예시이지 프로젝트 측정값이 아닙니다.**
`data/recommendation_metrics.example.json`의 수치를 그대로 제출하지 마세요.

## CT 연결 절차 (초기 100만 건을 신규 로그로 잘못 처리하지 않도록)

추천 평가 결과가 준비되면 모델 담당자가 평가한 `HitRate@50` 수치를 `data/latest_metrics.json`에 넣습니다.

```json
{"hit_rate": 0.23}
```

위 0.23도 **형식 예시**입니다. 기존 로그 100만 건은 **초기 기준점**으로 기록합니다.

```powershell
python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/latest_metrics.json --init-baseline
```

나중에 **새로운** 로그가 누적되면 아래처럼 현재 상태를 갱신합니다. 실제 학습 스크립트가 없는 상태에서는 10,000건을 넘더라도 훈련은 시작하지 않습니다.

```powershell
python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/latest_metrics.json
```

실제 학습 코드가 완성되고 실행 커맨드가 검증된 경우에만 `--train-command`로 연결합니다. **기존 ZIP의 로그를 그대로 다시 생성/덮어쓰는 방식은 로그 증가량 감시에 적합하지 않습니다.** 운영 단계에서는 append-only 로그 또는 이벤트 저장소를 별도로 지정해야 합니다.

## 자동 테스트

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

이 패치는 기존의 `src/search/` 코드를 변경하지 않습니다. 실제 서비스 검색 API를 호출하는 실시간 분석 기능도 아직 연결하지 않았습니다.
