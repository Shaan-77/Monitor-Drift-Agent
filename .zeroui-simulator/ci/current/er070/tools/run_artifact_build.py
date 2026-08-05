#!/usr/bin/env python3
"""Run ER-070 controlled artifact build and emit failure evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EVENT_TYPE_ID = "ci.artifact.build_failed"
FIXTURE_PROJECT_ROOT = Path(".zeroui-simulator/ci/current/er070/fixture/project")
EVIDENCE_PATH = Path(".zeroui-simulator/ci/current/er070/artifact/artifact-build-evidence.json")
BUILD_LOG_PATH = Path(".zeroui-simulator/ci/current/er070/artifact/build.log")
ARTIFACT_REF = "python_wheel:monitor-drift-agent:0.1.0"
ARTIFACT_TYPE = "python_wheel"
BUILD_FAMILY = "artifact_build"
CHECK_FAMILY = "artifact_build"
DRIFT_MONITOR_PATH = ".zeroui-simulator/ci/current/er070/fixture/project/src/monitor_drift_agent/drift_monitor.py"
EVIDENCE_FORMAT = "github-artifact-build-failure"
FAILURE_TYPE = "syntax_error"
ROUTING_REASON = "ci_artifact_build_failed"


def build_artifact_build_failed_evidence(*, failure_message: str) -> dict[str, Any]:
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
        "failure_type": FAILURE_TYPE,
        "failed_source_file": DRIFT_MONITOR_PATH,
        "failure_message": failure_message,
        "reason": ROUTING_REASON,
    }


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


FIXTURE_SRC = Path(".zeroui-simulator/ci/current/er070/fixture/project/src")
ARTIFACT_ROOT = Path(".zeroui-simulator/ci/current/er070/artifact")
ARTIFACT_DIST = Path(".zeroui-simulator/ci/current/er070/artifact/dist")
ARTIFACT_BUILD = Path(".zeroui-simulator/ci/current/er070/artifact/build")
ARTIFACT_EGG_INFO = Path(".zeroui-simulator/ci/current/er070/artifact/egg-info")
EXPECTED_WHEEL_NAME = "monitor_drift_agent-0.1.0-py3-none-any.whl"


def _clean_managed_runtime() -> None:
    import shutil

    for path in (ARTIFACT_DIST, ARTIFACT_BUILD, ARTIFACT_EGG_INFO):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for path in (BUILD_LOG_PATH, EVIDENCE_PATH):
        if path.exists():
            path.unlink()


def _count_wheels() -> int:
    if not ARTIFACT_DIST.exists():
        return 0
    return sum(1 for row in ARTIFACT_DIST.glob("*.whl") if row.is_file())


def run_build() -> tuple[int, int, str]:
    if not FIXTURE_PROJECT_ROOT.is_dir():
        raise ValueError("ER070_BUILD_FIXTURE_NOT_FOUND")
    if not FIXTURE_SRC.is_dir():
        raise ValueError("ER070_BUILD_FIXTURE_NOT_FOUND")

    _clean_managed_runtime()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIST.mkdir(parents=True, exist_ok=True)

    compile_cmd = [sys.executable, "-m", "compileall", "-q", str(FIXTURE_SRC)]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    compile_exit = int(compile_result.returncode)

    log_parts = [f"compileall exit={compile_exit}"]
    if compile_result.stdout:
        log_parts.append(compile_result.stdout)
    if compile_result.stderr:
        log_parts.append(compile_result.stderr)

    build_exit = 0
    if compile_exit == 0:
        build_cmd = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(ARTIFACT_DIST),
            str(FIXTURE_PROJECT_ROOT),
        ]
        build_result = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        build_exit = int(build_result.returncode)
        log_parts.append(f"build exit={build_exit}")
        if build_result.stdout:
            log_parts.append(build_result.stdout)
        if build_result.stderr:
            log_parts.append(build_result.stderr)
    else:
        build_exit = compile_exit

    log_text = "\n".join(part for part in log_parts if part)
    BUILD_LOG_PATH.write_text(log_text, encoding="utf-8")
    return compile_exit, build_exit, log_text


def validate_controlled_build_failure(
    compile_exit_code: int,
    build_exit_code: int,
    log_text: str,
) -> dict[str, Any]:
    compile_exit = int(compile_exit_code)
    build_exit = int(build_exit_code)
    effective_exit = compile_exit if compile_exit != 0 else build_exit

    if effective_exit == 0:
        raise ValueError("ER070_EXPECTED_ARTIFACT_BUILD_FAILURE_NOT_OBSERVED")
    if _count_wheels() > 0:
        raise ValueError("ER070_ARTIFACT_EVIDENCE_CONTRADICTORY")

    log_text = str(log_text or "")
    if "syntax" not in log_text.lower() and "SyntaxError" not in log_text:
        raise ValueError("ER070_UNEXPECTED_BUILD_FAILURE")
    if "drift_monitor.py" not in log_text:
        raise ValueError("ER070_UNEXPECTED_BUILD_FAILURE")

    failure_message = extract_failure_message(log_text)
    if not failure_message:
        raise ValueError("ER070_BUILD_EVIDENCE_INVALID")

    failure_stage = "source_compile" if compile_exit != 0 else "artifact_build"
    evidence = build_artifact_build_failed_evidence(failure_message=failure_message)
    evidence.update(
        {
            "artifact_present": False,
            "validation_passed": False,
            "source_built_in_ci": True,
            "artifact_build_system": "python_build",
            "build_tool": "python_m_build",
            "failure_stage": failure_stage,
            "compile_exit_code": compile_exit,
            "build_exit_code": build_exit,
            "expected_artifact_name": EXPECTED_WHEEL_NAME,
            "artifact_directory": str(ARTIFACT_DIST),
            "artifact_count": _count_wheels(),
            "build_command": "python -m compileall -q fixture/src; python -m build --wheel",
        }
    )
    return evidence


def _export_build_exit_codes(*, compile_exit_code: int, build_exit_code: int) -> None:
    github_env = os.getenv("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(f"COMPILE_EXIT_CODE={int(compile_exit_code)}\n")
            handle.write(f"BUILD_EXIT_CODE={int(build_exit_code)}\n")


def main() -> None:
    try:
        compile_exit, build_exit, log_text = run_build()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    _export_build_exit_codes(compile_exit_code=compile_exit, build_exit_code=build_exit)
    try:
        evidence = validate_controlled_build_failure(compile_exit, build_exit, log_text)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("ZeroUI ER-070 artifact build evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
