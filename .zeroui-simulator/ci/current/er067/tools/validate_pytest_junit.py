#!/usr/bin/env python3
"""Validate ER-067 controlled pytest JUnit evidence."""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Literal


EVENT_TYPE_ID = "ci.test.failed"
CONTROLLED_TEST_NAME = "test_er067_controlled_assertion_failure"
FIXTURE_PATH = ".zeroui-simulator/ci/current/er067/fixture/test_er067_failure.py"


def parse_junit_xml(text: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(str(text or ""))
    except ET.ParseError as exc:
        raise ValueError("ER067_JUNIT_EVIDENCE_INVALID") from exc

    testsuites: list[ET.Element] = []
    if root.tag == "testsuites":
        testsuites = [row for row in root.findall("testsuite") if isinstance(row.tag, str)]
    elif root.tag == "testsuite":
        testsuites = [root]
    else:
        raise ValueError("ER067_JUNIT_EVIDENCE_INVALID")

    if not testsuites:
        raise ValueError("ER067_JUNIT_EVIDENCE_INVALID")

    testcases: list[dict[str, Any]] = []
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

    for suite in testsuites:
        for key in totals:
            raw = suite.attrib.get(key)
            if raw is not None:
                try:
                    totals[key] += int(raw)
                except (TypeError, ValueError):
                    pass
        for testcase in suite.findall("testcase"):
            name = str(testcase.attrib.get("name") or "").strip()
            classname = str(testcase.attrib.get("classname") or "").strip()
            file_path = str(testcase.attrib.get("file") or "").strip()
            line_number = str(testcase.attrib.get("line") or "").strip()
            failure_node = testcase.find("failure")
            error_node = testcase.find("error")
            skipped_node = testcase.find("skipped")
            if failure_node is not None:
                status: Literal["failed", "error", "skipped", "passed"] = "failed"
            elif error_node is not None:
                status = "error"
            elif skipped_node is not None:
                status = "skipped"
            else:
                status = "passed"
            message_node = failure_node or error_node or skipped_node
            message = ""
            if message_node is not None:
                message = str(message_node.attrib.get("message") or "").strip()
                if not message:
                    message = str(message_node.text or "").strip()
            testcases.append(
                {
                    "name": name,
                    "classname": classname,
                    "file": file_path,
                    "line": line_number,
                    "status": status,
                    "failure_message": message,
                }
            )

    if not testcases and totals["tests"] <= 0:
        raise ValueError("ER067_NO_TESTS_COLLECTED")

    computed_total = len(testcases) if testcases else totals["tests"]
    computed_failures = sum(1 for row in testcases if row["status"] == "failed")
    computed_errors = sum(1 for row in testcases if row["status"] == "error")
    computed_skipped = sum(1 for row in testcases if row["status"] == "skipped")
    if testcases:
        computed_passed = sum(1 for row in testcases if row["status"] == "passed")
    else:
        computed_passed = max(0, totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"])

    return {
        "test_total": computed_total if testcases else totals["tests"],
        "test_failures": computed_failures if testcases else totals["failures"],
        "test_errors": computed_errors if testcases else totals["errors"],
        "test_skipped": computed_skipped if testcases else totals["skipped"],
        "test_passed": computed_passed,
        "testcases": testcases,
    }


def build_pytest_test_evidence(*, failed_test_name: str, failed_test_file: str, failure_message: str) -> dict[str, Any]:
    return {
        "event_type_id": EVENT_TYPE_ID,
        "test_framework": "pytest",
        "test_result_format": "junit_xml",
        "test_total": 1,
        "test_passed": 0,
        "test_failures": 1,
        "test_errors": 0,
        "test_skipped": 0,
        "tests_failed": 1,
        "failure_count": 1,
        "scan_passed": False,
        "check_family": "required_test_gate",
        "required_test_gate": True,
        "flaky": False,
        "retry_count": 0,
        "failure_deterministic": True,
        "failure_type": "assertion_failure",
        "failed_test_name": failed_test_name,
        "failed_test_file": failed_test_file,
        "failure_message": failure_message,
        "reason": "ci_test_failed",
    }


def validate_controlled_pytest_junit(junit_text: str, pytest_exit_code: int) -> dict[str, Any]:
    try:
        exit_code = int(pytest_exit_code)
    except (TypeError, ValueError) as exc:
        raise ValueError("ER067_PYTEST_EXECUTION_ERROR") from exc

    if exit_code == 5:
        raise ValueError("ER067_NO_TESTS_COLLECTED")
    if exit_code in {2, 3, 4}:
        raise ValueError("ER067_PYTEST_EXECUTION_ERROR")
    if exit_code == 0:
        raise ValueError("ER067_EXPECTED_TEST_FAILURE_NOT_OBSERVED")
    if exit_code != 1:
        raise ValueError("ER067_PYTEST_EXECUTION_ERROR")

    parsed = parse_junit_xml(junit_text)
    if int(parsed.get("test_total") or 0) != 1:
        raise ValueError("ER067_UNEXPECTED_TEST_EVIDENCE")
    if int(parsed.get("test_failures") or 0) != 1:
        raise ValueError("ER067_UNEXPECTED_TEST_EVIDENCE")
    if int(parsed.get("test_errors") or 0) != 0:
        raise ValueError("ER067_UNEXPECTED_TEST_EVIDENCE")

    matches = [
        row
        for row in parsed.get("testcases") or []
        if str(row.get("name") or "") == CONTROLLED_TEST_NAME
    ]
    if len(matches) != 1 or matches[0].get("status") != "failed":
        raise ValueError("ER067_UNEXPECTED_TEST_EVIDENCE")

    case = matches[0]
    failed_test_file = str(case.get("file") or case.get("classname") or FIXTURE_PATH).strip()
    failure_message = str(case.get("failure_message") or "assertion failure").strip()
    if not failure_message:
        raise ValueError("ER067_JUNIT_EVIDENCE_INVALID")

    return build_pytest_test_evidence(
        failed_test_name=CONTROLLED_TEST_NAME,
        failed_test_file=failed_test_file,
        failure_message=failure_message,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: validate_pytest_junit.py <junit-xml-path>", file=sys.stderr)
        raise SystemExit(2)

    junit_path = sys.argv[1]
    try:
        with open(junit_path, "r", encoding="utf-8") as handle:
            junit_text = handle.read()
    except OSError as exc:
        print(f"Missing junit evidence file: {junit_path} ({exc})", file=sys.stderr)
        raise SystemExit(2)

    raw_exit_code = os.getenv("PYTEST_EXIT_CODE", "")
    try:
        evidence = validate_controlled_pytest_junit(junit_text, int(raw_exit_code))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    print("ZeroUI ER-067 pytest evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
