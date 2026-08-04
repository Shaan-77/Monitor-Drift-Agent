#!/usr/bin/env python3
"""Build ER-051 coverage evidence from pytest-cov coverage.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED = 85.0
BASELINE = 90.0
EVENT_TYPE_ID = "ci.coverage.dropped"
EVIDENCE_PATH = Path("coverage-evidence.json")
COVERAGE_PATH = Path("coverage.json")


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


def compute_coverage_drop_percent(coverage_percent: float, baseline: float = BASELINE) -> float:
    return round(max(0.0, baseline - coverage_percent), 2)


def build_coverage_evidence(coverage_percent: float) -> dict:
    drop_percent = compute_coverage_drop_percent(coverage_percent)
    return {
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
        "coverage_dropped": True,
        "scan_passed": False,
        "tests_passed": True,
        "test_failure_count": 0,
        "reason": "coverage_below_required_threshold",
    }


def main() -> None:
    if not COVERAGE_PATH.is_file():
        print(f"Missing coverage report: {COVERAGE_PATH}", file=sys.stderr)
        raise SystemExit(2)
    try:
        coverage_data = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Malformed coverage report {COVERAGE_PATH}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    try:
        coverage_percent = parse_coverage_percent(coverage_data)
    except (TypeError, ValueError) as exc:
        print(f"Cannot parse coverage percent: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if coverage_percent >= REQUIRED:
        print(
            f"Coverage {coverage_percent:.2f}% meets required {REQUIRED:.2f}%; "
            "blocker-only scenario cannot proceed.",
            file=sys.stderr,
        )
        raise SystemExit(3)

    evidence = build_coverage_evidence(coverage_percent)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ZeroUI ER-051 coverage evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
