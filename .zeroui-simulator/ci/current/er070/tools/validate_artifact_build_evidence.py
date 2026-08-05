#!/usr/bin/env python3
"""Validate ER-070 controlled artifact build failure evidence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


EVENT_TYPE_ID = "ci.artifact.build_failed"
EVIDENCE_PATH = Path(".zeroui-simulator/ci/current/er070/artifact/artifact-build-evidence.json")
BUILD_LOG_PATH = Path(".zeroui-simulator/ci/current/er070/artifact/build.log")
ARTIFACT_REF = "python_wheel:monitor-drift-agent:0.1.0"
ARTIFACT_TYPE = "python_wheel"
BUILD_FAMILY = "artifact_build"
CHECK_FAMILY = "artifact_build"
DRIFT_MONITOR_PATH = ".zeroui-simulator/ci/current/er070/fixture/project/src/monitor_drift_agent/drift_monitor.py"
EVIDENCE_FORMAT = "github-artifact-build-failure"


def extract_failure_message(log_text: str) -> str:
    for line in str(log_text or "").splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        if "SyntaxError" in normalized:
            return normalized
        if "invalid syntax" in normalized.lower():
            return normalized
        if "drift_monitor.py" in normalized:
            return normalized
    lines = [row.strip() for row in str(log_text or "").splitlines() if row.strip()]
    if lines:
        return lines[-1][:500]
    return "artifact build failed with syntax error"


def validate_controlled_artifact_build_failure(
    evidence: dict[str, Any],
    *,
    build_log_text: str,
    build_exit_code: int,
) -> dict[str, Any]:
    try:
        exit_code = int(build_exit_code)
    except (TypeError, ValueError) as exc:
        raise ValueError("ER070_BUILD_EXECUTION_ERROR") from exc

    if exit_code == 0:
        raise ValueError("ER070_EXPECTED_BUILD_FAILURE_NOT_OBSERVED")
    if not isinstance(evidence, dict):
        raise ValueError("ER070_BUILD_EVIDENCE_INVALID")

    log_text = str(build_log_text or "")
    if "syntax" not in log_text.lower() and "SyntaxError" not in log_text:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")

    if str(evidence.get("event_type_id") or "") != EVENT_TYPE_ID:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if str(evidence.get("evidence_format") or "") != EVIDENCE_FORMAT:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if evidence.get("required_gate") is not True:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if str(evidence.get("build_family") or "") != BUILD_FAMILY:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if str(evidence.get("check_family") or "") != CHECK_FAMILY:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if str(evidence.get("artifact_ref") or "") != ARTIFACT_REF:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if str(evidence.get("artifact_type") or "") != ARTIFACT_TYPE:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if evidence.get("build_passed") is not False:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if evidence.get("artifact_build_failed") is not True:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if evidence.get("scan_passed") is not False:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if evidence.get("artifact_digest"):
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if str(evidence.get("workflow_family") or "") == "release_as_code":
        raise ValueError("ER070_ROUTING_CONTEXT_INVALID")
    if evidence.get("artifact_present") is True:
        raise ValueError("ER070_ARTIFACT_EVIDENCE_CONTRADICTORY")
    if evidence.get("source_built_in_ci") is not True:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")
    if evidence.get("validation_passed") is True:
        raise ValueError("ER070_UNEXPECTED_BUILD_EVIDENCE")

    failure_message = str(evidence.get("failure_message") or "").strip()
    if not failure_message:
        failure_message = extract_failure_message(log_text)
    if not failure_message:
        raise ValueError("ER070_BUILD_EVIDENCE_INVALID")

    return {
        "event_type_id": EVENT_TYPE_ID,
        "evidence_format": EVIDENCE_FORMAT,
        "build_family": BUILD_FAMILY,
        "check_family": CHECK_FAMILY,
        "required_gate": True,
        "artifact_ref": ARTIFACT_REF,
        "artifact_type": ARTIFACT_TYPE,
        "build_passed": False,
        "artifact_build_failed": True,
        "scan_passed": False,
        "failure_deterministic": True,
        "failure_type": "syntax_error",
        "failed_source_file": str(evidence.get("failed_source_file") or DRIFT_MONITOR_PATH),
        "failure_message": failure_message,
        "reason": "ci_artifact_build_failed",
    }


def main() -> None:
    if not EVIDENCE_PATH.exists():
        print(f"Missing evidence file: {EVIDENCE_PATH}", file=sys.stderr)
        raise SystemExit(2)
    if not BUILD_LOG_PATH.exists():
        print(f"Missing build log file: {BUILD_LOG_PATH}", file=sys.stderr)
        raise SystemExit(2)

    try:
        with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Invalid evidence file: {EVIDENCE_PATH} ({exc})", file=sys.stderr)
        raise SystemExit(2)

    try:
        build_log_text = BUILD_LOG_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Unreadable build log file: {BUILD_LOG_PATH} ({exc})", file=sys.stderr)
        raise SystemExit(2)

    raw_compile_exit = os.getenv("COMPILE_EXIT_CODE", os.getenv("BUILD_EXIT_CODE", "1"))
    raw_build_exit = os.getenv("BUILD_EXIT_CODE", raw_compile_exit)
    try:
        compile_exit_code = int(raw_compile_exit)
        build_exit_code = int(raw_build_exit)
    except ValueError as exc:
        print(f"Invalid build exit codes: compile={raw_compile_exit} build={raw_build_exit}", file=sys.stderr)
        raise SystemExit(2) from exc

    effective_exit = compile_exit_code if compile_exit_code != 0 else build_exit_code
    try:
        validated = validate_controlled_artifact_build_failure(
            evidence,
            build_log_text=build_log_text,
            build_exit_code=effective_exit,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    print("ZeroUI ER-070 artifact build evidence:")
    print(json.dumps(validated, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
