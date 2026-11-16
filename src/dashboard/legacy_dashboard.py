"""CodiSay team E dashboard. Run at repository root:
python -m streamlit run src/dashboard/app.py

Reads team reports from /docs; no CLIP model or million-row CSV load is necessary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ct_monitor import (
    HIT_RATE_THRESHOLD,
    NEW_LOGS_THRESHOLD,
    assess_ct,
    monitor_once,
    read_hit_rate,
    read_state,
)
from src.dashboard.ab_experiment import (
    check_normal_approximation,
    make_report,
    report_json,
    report_markdown,
    summarize_experiment_csv,
)
from src.dashboard.ab_test import analyze_ab_test
from src.dashboard.integration import (
    load_recommendation_metrics,
    load_search_reports,
    load_session_report,
    search_compare_rows,
)

st.set_page_config(page_title="RE:SEARCH | 성능·실험·CT", page_icon="📊", layout="wide")
st.title("📊 RE:SEARCH | 성능·실험·CT 대시보드")
st.caption("팀 자료 자동 연동 · 검색/추천 평가 · A/B 실험 리포트 · CT 상태/시연")
st.sidebar.subheader("데이터 연동")
st.sidebar.caption("프로젝트 루트의 docs/*.json 평가 결과를 자동으로 불러옵니다.")
st.sidebar.code("python -m streamlit run src/dashboard/app.py")

try:
    search = load_search_reports(ROOT)
    session = load_session_report(ROOT)
    rec = load_recommendation_metrics(ROOT)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

search_tab, rec_tab, logs_tab, ab_tab, ct_tab = st.tabs(
    [
        "🔎 검색 품질",
        "🎯 추천 품질",
        "🗂️ 세션 로그",
        "🧪 A/B 테스트",
        "🔄 CT 모니터링",
    ]
)

with search_tab:
    st.subheader("검색 엔진 평가 결과")
    full = search["bm25_full"]
    if full:
        metric = full.get("metrics", {})
        a, b, c = st.columns(3)
        a.metric("BM25 전체 MRR@10", f"{metric.get('mrr', float('nan')):.4f}")
        b.metric("BM25 전체 NDCG@10", f"{metric.get('ndcg_at_10', float('nan')):.4f}")
        c.metric("전체 검색 대상", f"{full.get('catalog', {}).get('included', 0):,}개")
        st.caption(
            "실측 평가 결과 · BM25 전체 검색 카탈로그 기준, 합성 검색어/정답 규칙 사용"
        )
    else:
        st.info("docs/search_baseline.json 파일이 아직 없습니다.")

    st.markdown("**같은 상품 1,000개 인덱스 기준의 시범 비교**")
    if search["bm25_pilot"] and search["clip_pilot"]:
        subset = st.radio(
            "평가 질의 기준",
            ["전체 11,300개 test 질의", "인덱스 내 정답이 있는 7,645개 질의"],
            horizontal=True,
        )
        covered = subset.startswith("인덱스 내")
        comparison = pd.DataFrame(search_compare_rows(search, covered_only=covered))
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("방법")[["MRR@10", "NDCG@10"]])
        st.warning(
            "CLIP은 1,000개 시범 인덱스만 평가한 결과입니다. 전체 70,521개 대상 최종 목표 달성으로 보고하면 안 됩니다."
        )
    else:
        st.info("시범 평가용 JSON 파일 2개가 있으면 비교 차트가 활성화됩니다.")

    latency = search["latency_pilot"]
    if latency:
        timings = latency.get("timings", {})
        st.markdown("**검색 지연시간 p95 (CPU · 검색 함수 자체만 측정)**")
        p95 = pd.DataFrame(
            [
                {"유형": kind, "p95 (ms)": stats.get("p95_ms", 0)}
                for kind, stats in timings.items()
            ]
        )
        st.dataframe(p95, use_container_width=True, hide_index=True)
        st.bar_chart(p95.set_index("유형"))
        if any(row["p95 (ms)"] > 200 for row in p95.to_dict("records")):
            st.warning(
                "일부 검색 방식은 200ms 목표를 초과했습니다. API 네트워크 시간은 측정에 포함되지 않았습니다."
            )
    if not any(search.values()):
        st.info(
            "검색 결과 파일이 없습니다. 팀 통합 ZIP의 docs 폴더와 인덱스 메타데이터를 복사하세요."
        )

with rec_tab:
    st.subheader("추천 모델 평가 결과")
    if rec is None:
        st.info(
            "아직 추천 평가 결과(data/recommendation_metrics.json)가 없습니다. 검색 평가 JSON을 추천 HitRate로 대신 사용하지 않습니다."
        )
    elif not rec:
        st.warning(
            "추천 평가 JSON 파일에 알려진 키가 없습니다. README_INTEGRATION.md 규격을 확인하세요."
        )
    else:
        labels = {
            "recall_at_300": ("Recall@300", 0.30),
            "auc": ("AUC", 0.70),
            "hit_rate_at_50": ("HitRate@50", 0.20),
            "ndcg_at_50": ("NDCG@50", 0.08),
            "coverage": ("Coverage", 0.20),
        }
        cols = st.columns(5)
        for i, (key, (label, threshold)) in enumerate(labels.items()):
            if key in rec:
                cols[i].metric(
                    label,
                    f"{rec[key]:.4f}",
                    delta=f"기준 {threshold:.2f}",
                    delta_color="off",
                )
            else:
                cols[i].metric(label, "미제공")
        st.caption(
            "이 값은 data/recommendation_metrics.json에서 읽었으며 평가 방식은 모델 담당자가 검증해야 합니다."
        )
    with st.expander("추천 평가 JSON 규격"):
        st.code(
            """{
  "recall_at_300": 0.32,
  "auc": 0.72,
  "hit_rate_at_50": 0.23,
  "ndcg_at_50": 0.09,
  "coverage": 0.25
}""",
            language="json",
        )
        st.caption("위 수치는 형식 설명을 위한 예시로 실제 모델 성능이 아닙니다.")

with logs_tab:
    st.subheader("세션 행동 로그 시뮬레이터 — 데이터 분할 검증")
    if session:
        a, b, c = st.columns(3)
        a.metric("행동 로그", f"{session.get('total_events', 0):,}건")
        stats = session.get("session_stats", {})
        b.metric("세션 수", f"{stats.get('n_sessions', 0):,}개")
        c.metric(
            "검색어 채움 비율",
            f"{session.get('search_query_filled_ratio', 0) * 100:.1f}%",
        )
        ratios = pd.DataFrame(
            [
                {"구분": split, "비율": r}
                for split, r in session.get("ratios", {}).items()
            ]
        )
        if not ratios.empty:
            st.bar_chart(ratios.set_index("구분"))
        events = session.get("event_type_ratio", {})
        if events:
            frame = pd.DataFrame(events).T
            st.caption("train / valid / test의 이벤트 유형 비율")
            st.bar_chart(frame)
        st.caption(
            "이 로그는 합성 시뮬레이션 자료입니다. 실제 서비스 유입 및 무작위 A/B 실험 데이터가 아닙니다."
        )
        warnings = session.get("warnings", [])
        for msg in warnings:
            st.warning(msg)
    else:
        st.info("docs/split_report.json이 있으면 분할 리포트를 보여줍니다.")

with ab_tab:
    st.subheader("A/B 테스트 · 통계 검정 및 리포트")
    st.caption(
        "Control/Treatment는 독립적인 사용자 단위여야 합니다. 전환율·Lift·양측 Z-test p-value·95% 신뢰구간을 계산합니다."
    )
    ab_mode = st.radio(
        "분석 방식",
        ["수동 시뮬레이션", "실험 CSV 분석"],
        horizontal=True,
        key="ab_mode",
    )
    design_verified = False
    summary = None
    result = None
    source_name = ""

    if ab_mode == "수동 시뮬레이션":
        st.info(
            "시연 모드: 입력값으로 계산하는 예시입니다. 실제 A/B 실험 수행 결과가 아닙니다."
        )
        left, right = st.columns(2)
        with left:
            n0 = int(
                st.number_input("Control 사용자 수", min_value=1, value=1000, step=100)
            )
            y0 = int(
                st.number_input(
                    "Control 전환 사용자 수",
                    min_value=0,
                    max_value=n0,
                    value=min(100, n0),
                )
            )
        with right:
            n1 = int(
                st.number_input(
                    "Treatment 사용자 수", min_value=1, value=1000, step=100
                )
            )
            y1 = int(
                st.number_input(
                    "Treatment 전환 사용자 수",
                    min_value=0,
                    max_value=n1,
                    value=min(130, n1),
                )
            )
        summary = {
            "row_count": n0 + n1,
            "groups": {
                "control": {"users": n0, "converted": y0},
                "treatment": {"users": n1, "converted": y1},
            },
        }
        result = analyze_ab_test(n0, y0, n1, y1)
        source_name = "사용자 수동 입력 (시뮬레이션)"
    else:
        st.caption(
            "다른 서비스 로그를 그대로 넣지 마세요. 실험군 배정 후 사용자별로 집계한 데이터만 허용합니다."
        )
        st.code(
            "user_id,group,converted\nuser_001,control,0\nuser_002,treatment,1",
            language="csv",
        )
        uploaded = st.file_uploader(
            "실험 데이터 CSV 업로드 (UTF-8, 최대 8MB / 10만 사용자)",
            type=["csv"],
            key="ab_uploaded",
        )
        if uploaded is None:
            st.info(
                "실험 CSV를 업로드하면 수치와 결과 리포트가 나옵니다. 현재 합성 behavior_logs.csv에는 실험군 정보가 없습니다."
            )
        else:
            try:
                summary, result = summarize_experiment_csv(uploaded.getvalue())
            except ValueError as exc:
                st.error(f"CSV 검증 오류: {exc}")
            else:
                source_name = f"실험 CSV: {uploaded.name} (검증되지 않은 업로드 데이터)"
                design_verified = st.checkbox(
                    "이 CSV는 무작위 배정된 독립 사용자 1명당 1행이며, 전환 규칙을 확인했습니다 (실험 담당자 확인)",
                    value=False,
                )
                if design_verified:
                    st.success(
                        "실험 설계 확인 항목에 체크했습니다. 무작위 배정 자체를 프로그램이 검증한 것은 아닙니다."
                    )
                else:
                    st.warning(
                        "무작위 배정·전환 정의가 확인되지 않아 실험 결과는 '미검증 데이터 분석'으로 표기합니다."
                    )
                st.caption(
                    f"검증된 CSV 구조: {summary['row_count']:,}행 / 중복 사용자 없음 / 전환값 0 또는 1"
                )

    if result is not None and summary is not None:
        g = summary["groups"]
        cols = st.columns(4)
        cols[0].metric("Control CVR", f"{result.control_cvr_pct:.2f}%")
        cols[1].metric("Treatment CVR", f"{result.treatment_cvr_pct:.2f}%")
        cols[2].metric(
            "상대 Lift",
            f"{result.lift_pct:+.2f}%" if result.lift_pct is not None else "정의 불가",
        )
        cols[3].metric("양측 p-value", f"{result.p_value:.4g}")
        plot_frame = pd.DataFrame(
            [
                {
                    "그룹": "Control",
                    "사용자": g["control"]["users"],
                    "전환": g["control"]["converted"],
                    "CVR (%)": result.control_cvr_pct,
                },
                {
                    "그룹": "Treatment",
                    "사용자": g["treatment"]["users"],
                    "전환": g["treatment"]["converted"],
                    "CVR (%)": result.treatment_cvr_pct,
                },
            ]
        )
        st.bar_chart(plot_frame.set_index("그룹")[["CVR (%)"]])
        st.dataframe(plot_frame, use_container_width=True, hide_index=True)
        st.write(
            f"**전환율 차이(Treatment − Control) 95% 신뢰구간:** {result.ci_low_pp:+.2f}%p ~ {result.ci_high_pp:+.2f}%p"
        )
        if result.significant_at_005:
            st.success(
                "양측 p-value < 0.05: 통계적으로 유의한 차이가 관측됐습니다. (인과관계는 실험 설계 검증 필요)"
            )
        else:
            st.info(
                "양측 p-value ≥ 0.05: 유의한 차이를 확인하지 못했습니다. '효과가 없다'는 뜻은 아닙니다."
            )
        if not check_normal_approximation(summary):
            st.warning(
                "일부 그룹의 전환/비전환 수가 5 미만이므로 Z-test 근사가 부정확할 수 있습니다. 정확검정 등 추가 검토를 권장합니다."
            )
        st.caption(
            "95% CI는 비율 차이(%p)의 Newcombe–Wilson 구간이며 상대 Lift(%)의 CI가 아닙니다. 반복 검정·지표 선택 등에 따른 한계가 있습니다."
        )
        report = make_report(
            summary,
            result,
            data_source=source_name,
            experimental_design_verified=design_verified,
        )
        left, right = st.columns(2)
        left.download_button(
            "A/B 리포트 Markdown 다운로드",
            report_markdown(report),
            file_name="codisay_ab_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        right.download_button(
            "A/B 리포트 JSON 다운로드",
            report_json(report),
            file_name="codisay_ab_report.json",
            mime="application/json",
            use_container_width=True,
        )

with ct_tab:
    st.subheader("Continuous Training · 모니터링/재학습 트리거")
    st.caption(
        "HitRate@50 ≤ 0.20이면 경고 / **기준 등록 이후 새 로그** 10,000건 이상이면 재학습 대상"
    )
    view_mode = st.radio(
        "확인 방식", ["CT 시연", "파일 기반 CT 상태"], horizontal=True, key="ct_mode"
    )
    if view_mode == "CT 시연":
        st.info(
            "시연 모드입니다. 실제 모델 성능이나 로그 수를 의미하지 않으며 파일을 수정하거나 학습을 실행하지 않습니다."
        )
        left, right = st.columns(2)
        with left:
            simulation_hit = st.slider("가상 HitRate@50", 0.0, 1.0, 0.18, 0.01)
        with right:
            simulation_logs = st.number_input(
                "기준 등록 후 가상 신규 로그 수",
                min_value=0,
                max_value=30000,
                value=10000,
                step=1000,
            )
        outcome = assess_ct(simulation_hit, int(simulation_logs))
        a, b, c = st.columns(3)
        a.metric("HitRate@50 (시연)", f"{simulation_hit:.3f}")
        b.metric("신규 로그 (시연)", f"{int(simulation_logs):,}건")
        c.metric("재학습 트리거", "대상" if outcome["training_due"] else "대기")
        st.progress(
            min(float(simulation_logs) / NEW_LOGS_THRESHOLD, 1.0),
            text=f"신규 로그 {int(simulation_logs):,} / {NEW_LOGS_THRESHOLD:,}건",
        )
        if outcome["performance_alert"]:
            st.error(
                f"⚠️ 가상 성능 저하 경고: HitRate@50 {simulation_hit:.3f} ≤ {HIT_RATE_THRESHOLD:.2f}"
            )
        else:
            st.success("가상 성능 임계값 통과")
        if outcome["training_due"]:
            st.warning(
                "가상 신규 로그 10,000건 도달 → 재학습 트리거 조건 충족 (실제 실행 없음)"
            )
        else:
            st.caption("가상 재학습 트리거 기준 미도달")
        st.code(
            "현재 버전 v1.0 → [실제 학습 스크립트 성공 확인 후] v1.1", language="text"
        )
    else:
        state_path = ROOT / "artifacts/ct_state.json"
        candidate_logs = ROOT / "data/behavior_logs_enriched.csv"
        logs_path = (
            candidate_logs if candidate_logs.is_file() else ROOT / "data/events.csv"
        )
        candidate_metrics = ROOT / "data/recommendation_metrics.json"
        metrics_path = (
            candidate_metrics
            if candidate_metrics.is_file()
            else ROOT / "data/latest_metrics.json"
        )
        state = read_state(state_path)
        st.caption(
            f"읽을 로그: {logs_path.relative_to(ROOT)} / 추천 지표: {metrics_path.relative_to(ROOT)}"
        )
        st.caption(
            "파일은 자동으로 검증된 실험 결과가 아닙니다. 실제 모델 담당자의 평가 데이터인지 출처를 확인하세요."
        )
        try:
            observed_hit = read_hit_rate(metrics_path)
        except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
            st.error(f"추천 지표 오류: {exc}")
            observed_hit = None
        if observed_hit is None:
            st.info(
                "실제 HitRate@50 평가 파일이 없습니다. 임의 수치로 성능 경고를 만들지 않습니다."
            )
        elif observed_hit <= HIT_RATE_THRESHOLD:
            st.error(
                f"⚠️ 파일 기반 HitRate@50={observed_hit:.3f} ≤ 0.20: 성능 저하 경고"
            )
        else:
            st.success(f"파일 기반 HitRate@50={observed_hit:.3f}: 임계값 통과")
        a, b, c = st.columns(3)
        a.metric("모델 버전", state.get("model_version", "v1.0"))
        b.metric(
            "재학습 이후 신규 로그",
            f"{state.get('new_logs_since_training', 0):,}건"
            if state["baseline_initialized"]
            else "기준 미등록",
        )
        c.metric("재학습 필요", "대상" if state.get("training_due") else "대기")
        if state["baseline_initialized"]:
            progress = state.get("new_logs_since_training", 0) / NEW_LOGS_THRESHOLD
            st.progress(
                min(max(progress, 0.0), 1.0),
                text=f"{state.get('new_logs_since_training', 0):,} / {NEW_LOGS_THRESHOLD:,} 신규 로그",
            )
        elif logs_path.is_file():
            st.warning(
                "초기 기준 로그가 등록되지 않았습니다. 기존 100만 건을 신규 로그로 오인하지 않도록 최초 한 번 기준 등록이 필요합니다."
            )
        else:
            st.warning(
                "로그 CSV가 없습니다. data/behavior_logs_enriched.csv 파일을 확인하세요."
            )
        left, right = st.columns(2)
        with left:
            if st.button(
                "현재 로그를 초기 기준으로 등록",
                disabled=bool(state["baseline_initialized"] or not logs_path.is_file()),
                use_container_width=True,
            ):
                try:
                    state = monitor_once(
                        logs_path, metrics_path, state_path, initialize_baseline=True
                    )
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    st.error(f"등록 실패: {exc}")
                else:
                    st.success(
                        "기존 로그를 초기 기준으로 등록했습니다. 아래 화면은 새로고침하면 변경됩니다."
                    )
                    st.rerun()
        with right:
            if st.button(
                "현재 로그/지표 상태 갱신 (학습 실행 안 함)",
                disabled=not bool(state["baseline_initialized"]),
                use_container_width=True,
            ):
                try:
                    state = monitor_once(logs_path, metrics_path, state_path)
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    st.error(f"갱신 실패: {exc}")
                else:
                    st.rerun()
        if state.get("checked_at_utc"):
            st.caption(f"마지막 점검(UTC): {state['checked_at_utc']}")
        for alert in state.get("alerts", []):
            st.warning(alert)
        if state.get("training_error"):
            st.error(f"마지막 학습 오류: {state['training_error']}")
        if state.get("history"):
            st.markdown("**CT 이벤트 및 버전 변경 이력**")
            st.dataframe(
                pd.DataFrame(list(reversed(state["history"]))),
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "CT 이력 JSON 다운로드",
                json.dumps(state["history"], ensure_ascii=False, indent=2),
                file_name="codisay_ct_history.json",
                mime="application/json",
            )
        else:
            st.caption("아직 CT 이벤트 이력이 없습니다.")
        with st.expander("정기 실행 및 실제 모델 재학습 연결 방법"):
            st.code(
                "python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/recommendation_metrics.json --init-baseline",
                language="powershell",
            )
            st.code(
                "python ct_monitor.py --logs data/behavior_logs_enriched.csv --metrics data/recommendation_metrics.json --watch --interval-sec 60",
                language="powershell",
            )
            st.caption(
                "실제 학습을 실행하려면 검증된 스크립트 준비 후 CLI에 --train-command 옵션을 지정해야 합니다. 웹 화면에서는 임의 명령을 실행하지 않습니다."
            )
            st.caption(
                "명령이 종료 코드 0으로 끝나면 버전을 v1.0→v1.1로 갱신합니다. 새 모델의 품질 평가와 배포 승인까지 완료했다는 뜻은 아닙니다."
            )

st.divider()
st.caption(
    "출처: 팀원이 제공한 ZIP의 docs 보고서 / 샘플 로그. 평가 파일이 없으면 '미제공'으로 표시합니다."
)
