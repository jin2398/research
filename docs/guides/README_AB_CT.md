# RE:SEARCH — A/B 테스트 + Continuous Training 대시보드 확장판

> 이 ZIP은 **기존 프로젝트에 덮어쓰는 소형 패치**입니다. `src/search`, 모델 코드, 대용량 CSV를 포함하거나 수정하지 않습니다. 팀원 이름은 화면 제목·설명에 표기하지 않았습니다.

## Windows PowerShell에서 실행하기

1. 팀 프로젝트의 최상위 폴더(`src`, `data`, `docs`가 있는 `codissay/`)를 VS Code로 엽니다.
2. **이 ZIP 내부 파일과 폴더**를 프로젝트 최상위 폴더에 복사하여 **같은 상대 경로의 파일만** 덮어씁니다. `src` 폴더를 삭제하지 마세요. `src/search` 및 프로젝트 다른 모듈은 그대로 유지됩니다.
3. 다음 명령을 프로젝트 루트에서 실행합니다.

```powershell
python -m pip install -r requirements-dashboard.txt
python -m streamlit run src/dashboard/app.py
```

브라우저 주소: http://localhost:8501

Python이 인식되지 않으면 `python`을 `py`로 바꿔 실행하세요. Streamlit CLI가 PATH에 없어도 `python -m streamlit`을 사용하므로 됩니다.

## A/B 테스트 이용 방법

대시보드의 `🧪 A/B 테스트` 탭에서

- **수동 시뮬레이션**: Control/Treatment 사용자 수와 전환 사용자 수를 입력하면 계산 결과·비교 그래프·결과 보고서가 나옵니다. 기본 1,000 / 100, 1,000 / 130은 **가상 수치**입니다.
- **실험 CSV 분석**: `data/ab_experiment.demo.csv`를 업로드하면 CSV 구조 검증, 실제 계산, 리포트 내보내기를 테스트할 수 있습니다. 데모 CSV도 가상 데이터입니다.
- **실제 실험 CSV**: 사용자별 1행, `user_id,group,converted` 세 컬럼이 필요합니다. `group`은 `control` 또는 `treatment`, `converted`는 `0` 또는 `1`입니다. 중복 사용자는 허용하지 않습니다.
- 통계: 양측 두 비율 Z-test + Newcombe–Wilson 95% CI (Treatment - Control 비율 차이, 단위 `%p`). CVR·상대 Lift(%)·p-value는 별도 표기합니다.
- **주의**: 업로드한 데이터는 구조만 검증할 수 있습니다. 무작위 배정, 독립성, 전환 정의, 봇 제거 여부 등은 실험 책임자가 별도로 확인해야 합니다. 실험군 정보가 없는 기존 `data/behavior_logs.csv`는 A/B 입력으로 사용할 수 없습니다.
- 보고서는 **Markdown**과 **JSON**으로 다운로드할 수 있습니다. 실험 설계 체크박스는 사용자 선언일 뿐 독립 검증을 뜻하지 않습니다.

### 데모 A/B 결과

`data/ab_experiment.demo.csv`는 Control 사용자 1,000명 중 100명 전환, Treatment 사용자 1,000명 중 130명 전환으로 구성되어 있습니다. 따라서 Control CVR=10%, Treatment CVR=13%, Lift=+30%가 보여야 합니다. **실제 서비스 결과가 아닙니다.**

## CT 모니터링 이용 방법

대시보드의 `🔄 CT 모니터링` 탭에서

- **CT 시연**: HitRate@50 및 로그 수를 바꾸면서 `0.20` 이하 경고와 `10,000`건 이상 트리거를 테스트합니다. 아무 실제 모델도 학습시키지 않고 파일도 수정하지 않습니다.
- **파일 기반 CT 상태**: JSON에서 제공된 HitRate@50와 `artifacts/ct_state.json`에서 읽은 로그 기준점·모델 버전·알림·이력을 표시합니다.
- `현재 로그를 초기 기준으로 등록` 버튼: 데이터 CSV의 **기존 로그를 기준으로 등록**합니다. 수백만 건도 재학습 대상으로 오판하지 않게 하는 중요한 최초 1회 작업입니다. 이미 등록했으면 버튼이 비활성화됩니다.
- `현재 로그/지표 상태 갱신` 버튼: CSV의 현재 행 수를 세고 상태를 갱신하지만 **실제 재학습은 실행하지 않습니다.** 모델 버전도 변화하지 않습니다.
- 신규 로그 10,000건 이상 감지 후, 실제 재학습 **스크립트를 안전하게 준비**하고 `--train-command`를 CLI에서 명시한 경우에만 학습 명령을 실행합니다. 학습 명령이 종료 코드 0을 반환하면 `v1.0→v1.1`처럼 마이너 버전을 올리고 `artifacts/ct_state.json`에 기록합니다. 종료 코드 0이 모델 품질/배포 성공을 보장하지는 않습니다.

### 추천 평가 JSON 규격

실제 추천 평가 파일: `data/recommendation_metrics.json`. 모델 평가 결과를 제공받았을 때에만 생성하세요.

```json
{
  "recall_at_300": 0.32,
  "auc": 0.72,
  "hit_rate_at_50": 0.23,
  "ndcg_at_50": 0.09,
  "coverage": 0.25
}
```

위 값은 **구조 예시**이며 프로젝트의 실측치가 아닙니다. **예시 JSON을 실제 데이터 파일로 저장하여 프로젝트 실측치인 것처럼 보고하지 마세요.**

### CLI 명령 (실제 로그만 추적)

프로젝트 루트에서 처음 한 번 기존 로그 기준 등록:

```powershell
python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/recommendation_metrics.json --init-baseline
```

이후 한 번 상태 확인:

```powershell
python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/recommendation_metrics.json
```

이후 주기적 감시 (터미널이 켜져 있는 동안만 60초마다 실행):

```powershell
python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/recommendation_metrics.json --watch --interval-sec 60
```

실제 학습 스크립트 `scripts/train_model.py`가 구현되고 안전하게 테스트되었을 때만 다음처럼 실행하세요.

```powershell
python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/recommendation_metrics.json --train-command "python scripts/train_model.py"
```

**주의**: `--train-command`는 전달받은 임의 명령 문자열을 웹에 입력해 실행하는 기능이 아닙니다. 신뢰할 수 있는 로컬 학습 코드로만 설정하세요. 코드가 아직 없으면 사용하지 마세요. 로그 CSV를 append 방식으로 갱신해야 기존 대비 신규 행 증가를 정확히 계산할 수 있습니다. 기존 로그 파일을 덮어쓰거나 로테이션하면 `로그 감소` 경고 및 기준 재설정이 발생합니다.

## 현재 통합 데이터와 화면

- 동봉 `docs/*.json` 5개는 이전에 받은 팀 통합본 `multi_session_team_bundle.zip`에서 추출한 검색/세션 보고서이며, 수치를 새로 생성한 게 아닙니다.
- 검색 품질: BM25 전체, BM25/CLIP 파일럿 결과 (대상 데이터 규모가 서로 다름), 검색 p95 차트.
- 세션 로그: 기존 합성 로그 분할 결과.
- 추천 품질: `data/recommendation_metrics.json`이 없으면 `미제공`.
- A/B 테스트: **시뮬레이션 또는 별도 실험 CSV 업로드가 있어야** 계산.
- CT: **시뮬레이션 또는 별도 로그·추천 지표 파일이 있어야** 상태 표시.
- `data/ab_experiment.demo.csv`는 시연 전용으로, `data/recommendation_metrics.json` 대신 사용되지 않습니다.

## 코드 파일 및 테스트

| 파일 | 기능 |
|---|---|
| `src/dashboard/app.py` | 5개 탭 Streamlit UI |
| `src/dashboard/ab_test.py` | 비율 계산·Z-test·신뢰구간 |
| `src/dashboard/ab_experiment.py` | 실험 CSV 검증·리포트 내보내기 |
| `src/dashboard/integration.py` | 팀 검색/추천/세션 JSON 로딩 |
| `ct_monitor.py` | CT 로그 기준·경보·학습 트리거·버전/이력 |
| `tests/*.py` | 통계·입력·데이터 연결·CT 회귀 테스트 |

테스트 실행:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

이 패치의 계산 함수와 상태 관리 로직은 자동화 테스트를 통과했지만, **실제 Streamlit 브라우저 실행 및 모델 학습/배포 연동은 개별 PC·팀 환경에서 추가 확인이 필요**합니다.
