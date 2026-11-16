"""Validate per-user randomized A/B test CSV and generate honest reports.

Input is one row per experimental user, NOT raw click event rows:
user_id,group,converted   (optional: event_date)
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from typing import Any

from src.dashboard.ab_test import ABTestResult, analyze_ab_test

EXPECTED = {"user_id", "group", "converted"}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000


def summarize_experiment_csv(
    source: bytes, *, max_rows: int = MAX_ROWS
) -> tuple[dict, ABTestResult]:
    """Reject ambiguous/duplicated subjects and malformed conversion flags."""
    if not source:
        raise ValueError("비어 있는 CSV입니다.")
    if len(source) > MAX_FILE_BYTES:
        raise ValueError(
            "CSV가 너무 큽니다(최대 8MB). 사용자별 요약 데이터를 올려주세요."
        )
    try:
        contents = source.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("UTF-8 인코딩 CSV가 필요합니다.") from exc
    reader = csv.DictReader(io.StringIO(contents, newline=""))
    if reader.fieldnames is None:
        raise ValueError("헤더가 없는 CSV입니다.")
    fields = [field.strip() for field in reader.fieldnames if field is not None]
    if len(fields) != len(set(fields)):
        raise ValueError("CSV에 중복된 컬럼명이 있습니다.")
    if not EXPECTED.issubset(set(fields)):
        raise ValueError("필수 컬럼: user_id, group, converted")
    if any(field is None for field in reader.fieldnames):
        raise ValueError("CSV 컬럼명이 올바르지 않습니다.")
    reader.fieldnames = fields  # Allow harmless spaces around CSV header names.

    users: set[str] = set()
    groups = {
        "control": {"users": 0, "converted": 0},
        "treatment": {"users": 0, "converted": 0},
    }
    rows = 0
    for line, record in enumerate(reader, start=2):
        rows += 1
        if rows > max_rows:
            raise ValueError(f"최대 {max_rows:,}명까지 업로드할 수 있습니다.")
        if None in record:
            raise ValueError(f"{line}행: 컬럼 수가 헤더보다 많습니다.")
        user_id = (record.get("user_id") or "").strip()
        group = (record.get("group") or "").strip().lower()
        converted = (record.get("converted") or "").strip()
        if not user_id:
            raise ValueError(f"{line}행: user_id가 비어 있습니다.")
        if user_id in users:
            raise ValueError(
                f"{line}행: user_id가 중복되었습니다. 사용자별 1행만 사용하세요."
            )
        if group not in groups:
            raise ValueError(f"{line}행: group은 control 또는 treatment여야 합니다.")
        if converted not in {"0", "1"}:
            raise ValueError(f"{line}행: converted는 0 또는 1이어야 합니다.")
        users.add(user_id)
        groups[group]["users"] += 1
        groups[group]["converted"] += int(converted)

    if not rows:
        raise ValueError("데이터 행이 없습니다.")
    if not (groups["control"]["users"] and groups["treatment"]["users"]):
        raise ValueError("Control과 Treatment 양쪽 그룹에 최소 1명씩 필요합니다.")

    result = analyze_ab_test(
        groups["control"]["users"],
        groups["control"]["converted"],
        groups["treatment"]["users"],
        groups["treatment"]["converted"],
    )
    return {"row_count": rows, "groups": groups}, result


def check_normal_approximation(summary: dict) -> bool:
    """Whether each independent group's successes and failures are >= 5."""
    return all(
        min(item["converted"], item["users"] - item["converted"]) >= 5
        for item in summary["groups"].values()
    )


def make_report(
    summary: dict,
    result: ABTestResult,
    *,
    data_source: str,
    experimental_design_verified: bool,
) -> dict[str, Any]:
    """`experimental_design_verified` is supplied by the experiment owner, not inferred."""
    return {
        "title": "RE:SEARCH A/B 테스트 분석",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": data_source,
        "status": "실험 설계 확인(사용자 선언)"
        if experimental_design_verified
        else "시뮬레이션 또는 미검증",
        "randomization_verified_independently": False,
        "sample_counts": summary,
        "results": result.to_dict(),
        "test": "two-sided pooled two-proportion Z-test",
        "confidence_interval": "Newcombe-Wilson for (treatment - control), percentage points",
        "alpha": 0.05,
        "normal_approximation_sufficient": check_normal_approximation(summary),
        "notes": "관측 데이터만으로 무작위 배정/인과관계를 검증할 수 없음. p<0.05는 인과관계의 보증이 아님.",
    }


def report_markdown(report: dict[str, Any]) -> str:
    r = report["results"]
    g = report["sample_counts"]["groups"]
    lift = (
        "정의되지 않음 (Control CVR 0%)"
        if r["lift_pct"] is None
        else f"{r['lift_pct']:+.2f}%"
    )
    return (
        "# RE:SEARCH A/B 테스트 결과 리포트\n\n"
        f"- 생성 시각 (UTC): {report['generated_at_utc']}\n"
        f"- 출처: {report['source']}\n"
        f"- 분류: **{report['status']}**\n"
        f"- Control: {g['control']['users']:,}명 / 전환 {g['control']['converted']:,}명\n"
        f"- Treatment: {g['treatment']['users']:,}명 / 전환 {g['treatment']['converted']:,}명\n"
        f"- Control CVR: {r['control_cvr_pct']:.2f}%\n"
        f"- Treatment CVR: {r['treatment_cvr_pct']:.2f}%\n"
        f"- 상대 Lift: {lift}\n"
        f"- 양측 p-value: {r['p_value']:.6g}\n"
        f"- 전환율 차이(Treatment - Control) 95% CI: {r['ci_low_pp']:+.2f}%p ~ {r['ci_high_pp']:+.2f}%p\n"
        f"- 유의수준 0.05 미만: {'예' if r['significant_at_005'] else '아니오'}\n"
        f"- Z-test 정규 근사 표본수 권고 충족: {'예' if report['normal_approximation_sufficient'] else '아니오'}\n\n"
        "## 해석 유의 사항\n\n"
        "양측 두 비율 Z-test 및 Newcombe–Wilson 신뢰구간을 사용했습니다. "
        "실제 무작위 배정 여부와 전환 측정 규칙은 실험 책임자가 별도로 확인해야 합니다. "
        "이 파일 자체가 인과관계를 증명하지 않습니다.\n"
    )


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"
