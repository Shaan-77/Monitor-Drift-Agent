#!/usr/bin/env python3
"""Build ER-051 coverage evidence from pytest-cov coverage.json (seed-parameterized)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = 80.0
BASELINE = 80.0
SCENARIO_ID = 'runtime_not_ready'
EMIT_COVERAGE_DROPPED = False
RELEASE_CRITICAL = None
SIMULATOR_FAULT = json.loads('{}')
EVENT_TYPE_ID = "ci.coverage.dropped"
EVIDENCE_PATH = Path("coverage-evidence.json")
COVERAGE_PATH = Path("coverage.json")
SCENARIO_PATH = Path("../scenario.json") if False else Path("../../scenario.json")


def parse_coverage_percent(coverage_data: dict) -> float:
    totals = coverage_data.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("Coverage totals missing or invalid")
    percent = totals.get("percent_covered")
    if percent is not None:
        return float(percent)
    num_statements = totals.get("num_statements")
    covered_lines = totals.get("covered_lines")
    if num_statements is not None and covered_lines is not None:
        num = float(num_statements)
        if num <= 0:
            raise ValueError("Coverage num_statements must be positive")
        return float(covered_lines) * 100.0 / num
    raise ValueError("Cannot derive coverage percent from totals")


def main() -> None:
    if SIMULATOR_FAULT:
        kind = str(SIMULATOR_FAULT.get("kind") or "")
        # Simulator-only controlled faults — never used for normal provider traffic.
        payload = {
            "simulator_fault": True,
            "fault_kind": kind,
            "scenario_id": SCENARIO_ID,
            "event_type_id": EVENT_TYPE_ID,
            "failure_code": SIMULATOR_FAULT.get("failure_code"),
            "failure_stage": SIMULATOR_FAULT.get("failure_stage"),
            "coverage_dropped": False,
            "scan_passed": False,
            "tests_passed": True,
            "reason": f"simulator_fault:{kind}",
        }
        EVIDENCE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"ZeroUI ER-051 simulator fault evidence: {kind}")
        # Exit non-zero so the workflow surfaces the controlled fault path.
        raise SystemExit(2)

    if not COVERAGE_PATH.is_file():
        print(f"Missing coverage report: {COVERAGE_PATH}", file=sys.stderr)
        raise SystemExit(1)
    try:
        coverage_data = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Malformed coverage report {COVERAGE_PATH}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    try:
        coverage_percent = parse_coverage_percent(coverage_data)
    except Exception as exc:
        print(f"Cannot parse coverage percent: {exc}", file=sys.stderr)
        raise SystemExit(1)

    below = coverage_percent < REQUIRED
    if EMIT_COVERAGE_DROPPED and not below:
        print(
            f"Coverage {coverage_percent:.2f}% meets required {REQUIRED:.2f}%; "
            "selected breach scenario cannot proceed.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    if not EMIT_COVERAGE_DROPPED and below:
        print(
            f"Coverage {coverage_percent:.2f}% is below required {REQUIRED:.2f}%; "
            "pass scenario cannot emit a breach event.",
            file=sys.stderr,
        )
        raise SystemExit(3)

    drop_percent = round(max(0.0, BASELINE - coverage_percent), 2)
    evidence = {
        "event_type_id": EVENT_TYPE_ID,
        "metric_type": "coverage",
        "coverage_tool": "coverage.py",
        "coverage_report_format": "coverage-json",
        "coverage_scope": "python_unit_tests",
        "quality_gate": "required_coverage_gate",
        "covered_file": "zeroui_uat/er051_discount_calculator.py",
        "observed_value": coverage_percent,
        "coverage_percent": coverage_percent,
        "current_coverage_percent": coverage_percent,
        "threshold": REQUIRED,
        "required_coverage_percent": REQUIRED,
        "baseline": BASELINE,
        "baseline_coverage_percent": BASELINE,
        "coverage_drop_percent": drop_percent,
        "coverage_dropped": bool(EMIT_COVERAGE_DROPPED and below),
        "quality_gate_release_critical": RELEASE_CRITICAL,
        "scan_passed": not (EMIT_COVERAGE_DROPPED and below),
        "tests_passed": True,
        "test_failure_count": 0,
        "reason": "coverage_below_required_threshold" if below else "coverage_meets_required_threshold",
        "scenario_id": SCENARIO_ID,
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("ZeroUI ER-051 coverage evidence:")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
