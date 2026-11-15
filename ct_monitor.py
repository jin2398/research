"""CodiSay CT monitor (CLI): log growth trigger, HitRate alarm, version history.

Safety: on first run, register existing logs with --init-baseline. Never invoke
an unspecified training command. Versions change only after command exit code 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

HIT_RATE_THRESHOLD = 0.20
NEW_LOGS_THRESHOLD = 10_000
DEFAULT_VERSION = "v1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def count_log_rows(path: Path) -> int:
    """Count CSV data rows, ignoring the header; handles UTF-8 BOM."""
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def read_hit_rate(path: Path) -> Optional[float]:
    """Accept verified external recommendation metrics, never fabricate scores."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Metrics JSON must be an object")
    value = data.get("hit_rate_at_50", data.get("hit_rate"))
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("HitRate must be numeric")
    rate = float(value)
    if not 0 <= rate <= 1:
        raise ValueError("HitRate must be in [0, 1]")
    return rate


def read_state(path: Path) -> dict:
    if not path.is_file():
        return {"model_version": DEFAULT_VERSION, "last_trained_log_count": 0,
                "baseline_initialized": False, "history": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("CT state must be a JSON object")
    state.setdefault("model_version", DEFAULT_VERSION)
    state.setdefault("last_trained_log_count", 0)
    state.setdefault("history", [])
    # Backwards compatibility with state files from previous app versions.
    state.setdefault("baseline_initialized", "last_observed_log_count" in state)
    return state


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def bump_minor_version(version: str) -> str:
    match = re.fullmatch(r"v(\d+)\.(\d+)", version)
    if not match:
        raise ValueError("Expected version format: v1.0")
    return f"v{match.group(1)}.{int(match.group(2)) + 1}"


def _event(state: dict, kind: str, detail: str) -> None:
    state.setdefault("history", []).append({"at_utc": _now(), "event": kind,
                                              "detail": detail,
                                              "model_version": state["model_version"]})
    state["history"] = state["history"][-100:]


def assess_ct(hit_rate: Optional[float], new_logs: int, *, baseline_initialized: bool = True) -> dict:
    """Pure assessment used by CLI and in-memory dashboard simulation."""
    if hit_rate is not None and (isinstance(hit_rate, bool) or not 0 <= hit_rate <= 1):
        raise ValueError("HitRate must be within [0, 1] or missing")
    if new_logs < 0:
        raise ValueError("new_logs cannot be negative")
    return {
        "performance_alert": hit_rate is not None and hit_rate <= HIT_RATE_THRESHOLD,
        "training_due": bool(baseline_initialized and new_logs >= NEW_LOGS_THRESHOLD),
        "hit_rate": hit_rate,
        "new_logs_since_training": new_logs,
    }


def monitor_once(
    logs_path: Path,
    metrics_path: Path,
    state_path: Path,
    train_command: Optional[str] = None,
    initialize_baseline: bool = False,
    training_timeout_sec: int = 1800,
) -> dict:
    """Persist a single CT check. Real training only via CLI trusted command.

    The first check never treats pre-existing 1,000,000 synthetic logs as new:
    baseline registration is required before counting any new records.
    """
    if training_timeout_sec < 1:
        raise ValueError("training_timeout_sec must be >= 1")
    state = read_state(state_path)
    count = count_log_rows(logs_path)
    hit_rate = read_hit_rate(metrics_path)
    notifications: list[str] = []
    previously_due = bool(state.get("training_due"))
    previously_alert = bool(state.get("performance_alert"))
    last_trained = int(state["last_trained_log_count"])

    if not logs_path.is_file():
        notifications.append(f"Log CSV is missing: {logs_path}")
    if not state["baseline_initialized"] and not initialize_baseline:
        notifications.append("Baseline not initialized. Run --init-baseline first; no training executed.")
    if initialize_baseline:
        if not logs_path.is_file():
            notifications.append("Cannot initialize baseline: log CSV is missing.")
        elif state["baseline_initialized"]:
            notifications.append("Baseline already initialized; reset blocked to avoid losing new-log progress.")
        else:
            last_trained = count
            state["baseline_initialized"] = True
            _event(state, "baseline_initialized", f"Existing {count:,} log rows registered")
            notifications.append(f"Initialized baseline at {count:,} existing rows.")

    if state["baseline_initialized"] and count < last_trained:
        notifications.append("Log row count decreased; resetting baseline. Check log rotation/replacement.")
        last_trained = count
        _event(state, "log_count_reset", f"Log row count decreased to {count:,}")

    new_logs = max(0, count - last_trained) if state["baseline_initialized"] else 0
    assessment = assess_ct(hit_rate, new_logs, baseline_initialized=state["baseline_initialized"])
    if assessment["performance_alert"]:
        notifications.append(f"ALERT: HitRate@50 {hit_rate:.3f} <= {HIT_RATE_THRESHOLD:.2f}")
        if not previously_alert:
            _event(state, "performance_alert", f"HitRate@50={hit_rate:.3f}")
    if assessment["training_due"] and not previously_due:
        _event(state, "training_due", f"New log rows={new_logs:,}")

    trained = False
    training_error = None
    if assessment["training_due"]:
        if train_command and train_command.strip():
            try:
                command = shlex.split(train_command)
                if not command:
                    raise ValueError("empty train command")
                # Never shell=True; do not execute arbitrary uploaded commands.
                subprocess.run(command, check=True, timeout=training_timeout_sec)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
                training_error = str(exc)
                notifications.append(f"Training failed; version unchanged: {training_error}")
                _event(state, "training_failed", training_error)
            else:
                state["model_version"] = bump_minor_version(state["model_version"])
                last_trained = count
                new_logs = 0
                trained = True
                state["last_trained_at_utc"] = _now()
                notifications.append(f"Training command succeeded: version -> {state['model_version']}")
                _event(state, "training_completed", f"Command exit code 0; {count:,} rows total")
        else:
            notifications.append("Training due, but --train-command not set; no training executed.")

    state.update({
        "last_trained_log_count": last_trained,
        "last_observed_log_count": count,
        "new_logs_since_training": new_logs,
        "hit_rate": hit_rate,
        "performance_alert": assessment["performance_alert"],
        "training_due": bool(state["baseline_initialized"] and new_logs >= NEW_LOGS_THRESHOLD),
        "training_error": training_error,
        "alerts": notifications,
        "checked_at_utc": _now(),
    })
    write_state(state_path, state)
    return {**state, "trained": trained}


def main() -> None:
    parser = argparse.ArgumentParser(description="CodiSay continuous training monitor")
    parser.add_argument("--logs", type=Path, default=Path("data/behavior_logs_enriched.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("data/recommendation_metrics.json"))
    parser.add_argument("--state", type=Path, default=Path("artifacts/ct_state.json"))
    parser.add_argument("--train-command", type=str, default=None, help="Locally trusted model training command")
    parser.add_argument("--init-baseline", action="store_true", help="Register existing CSV rows as baseline (one-time)")
    parser.add_argument("--watch", action="store_true", help="Poll periodically")
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--training-timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if args.interval_sec < 1 or args.training_timeout_sec < 1:
        parser.error("--interval-sec and --training-timeout-sec must be >= 1")
    while True:
        status = monitor_once(args.logs, args.metrics, args.state, args.train_command,
                              args.init_baseline, args.training_timeout_sec)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        if not args.watch:
            break
        args.init_baseline = False
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
