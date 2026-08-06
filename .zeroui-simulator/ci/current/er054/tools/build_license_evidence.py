#!/usr/bin/env python3
"""Build ER-054 license evidence from pip-licenses JSON output."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Literal


EVENT_TYPE_ID = "ci.license_policy.failed"
TARGET_PACKAGE_NAME = "zeroui-uat-gpl-demo"
TARGET_PACKAGE_VERSION = "0.1.0"
EVIDENCE_PATH = Path("license-evidence.json")
DENIED_LICENSE_TOKENS = ("GPL", "AGPL")


def normalize_license_text(license_text: str) -> str:
    text = str(license_text or "").strip()
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text)
    return collapsed.upper()


def is_lgpl_license(license_text: str) -> bool:
    normalized = normalize_license_text(license_text)
    return "LGPL" in normalized


def is_agpl_license(license_text: str) -> bool:
    normalized = normalize_license_text(license_text)
    return "AGPL" in normalized


def is_gpl_license(license_text: str) -> bool:
    normalized = normalize_license_text(license_text)
    if not normalized:
        return False
    if is_lgpl_license(license_text) or is_agpl_license(license_text):
        return False
    if "GPL" in normalized:
        return True
    if "GENERAL PUBLIC LICENSE" in normalized and "LESSER" not in normalized:
        return True
    return False


def is_denied_license(license_text: str) -> bool:
    normalized = normalize_license_text(license_text)
    if not normalized or normalized in {"UNKNOWN", "N/A", "NONE"}:
        return False
    if is_agpl_license(license_text):
        return True
    if is_gpl_license(license_text):
        return True
    for token in DENIED_LICENSE_TOKENS:
        if token == "GPL":
            continue
        if token in normalized:
            return True
    return False


def classify_license_policy(license_text: str) -> Literal["denied", "allowed", "unknown"]:
    normalized = normalize_license_text(license_text)
    if not normalized or normalized in {"UNKNOWN", "N/A", "NONE"}:
        return "unknown"
    if is_denied_license(license_text):
        return "denied"
    allowed_markers = (
        "MIT",
        "APACHE",
        "BSD",
        "ISC",
        "UNLICENSE",
        "CC0",
        "PUBLIC DOMAIN",
        "PYTHON SOFTWARE FOUNDATION",
        "MPL",
    )
    if any(marker in normalized for marker in allowed_markers):
        return "allowed"
    return "unknown"


def parse_pip_licenses_output(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("pip-licenses output must be a JSON array")
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("pip-licenses rows must be JSON objects")
        rows.append(item)
    return rows


def find_target_package(scan_results: list[dict[str, Any]], package_name: str) -> dict[str, Any]:
    target = str(package_name or "").strip().lower()
    if not target:
        raise ValueError("package_name is required")
    matches = [
        row
        for row in scan_results
        if str(row.get("Name") or row.get("name") or "").strip().lower() == target
    ]
    if not matches:
        raise ValueError(f"Target package not found in pip-licenses output: {package_name}")
    if len(matches) > 1:
        raise ValueError(f"Target package matched more than once: {package_name}")
    return matches[0]


def extract_package_license(row: dict[str, Any]) -> str:
    for key in ("License", "license", "License-Expression", "license_expression"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def build_license_evidence(
    *,
    package_name: str,
    package_version: str,
    license_text: str,
) -> dict[str, Any]:
    classification = classify_license_policy(license_text)
    denied = classification == "denied"
    policy_verdict = "denied" if denied else classification
    license_id = str(license_text or "").strip()
    dependency_entry = {
        "package": package_name,
        "version": package_version,
        "license": license_id,
        "policy_verdict": policy_verdict,
    }
    return {
        "event_type_id": EVENT_TYPE_ID,
        "scan_type": "license",
        "license_scanner": "pip-licenses",
        "license_report_format": "pip-licenses-json",
        "dependency_scope": "python_application_dependencies",
        "policy_id": "zeroui-denied-license-policy",
        "denied_license_tokens": list(DENIED_LICENSE_TOKENS),
        "package_name": package_name,
        "package_version": package_version,
        "license_id": license_id,
        "license_text": license_id,
        "policy_verdict": policy_verdict,
        "denied_license_count": 1 if denied else 0,
        "allowed_license_count": 1 if policy_verdict == "allowed" else 0,
        "unknown_license_count": 1 if policy_verdict == "unknown" else 0,
        "denied_dependencies": [dependency_entry] if denied else [],
        "scanned_dependencies": [dependency_entry],
        "scan_passed": not denied,
        "license_policy_passed": not denied,
        "reason": "denied_license_detected" if denied else "license_policy_passed",
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: build_license_evidence.py <pip-licenses.json>", file=sys.stderr)
        raise SystemExit(2)

    licenses_path = Path(sys.argv[1])
    if not licenses_path.is_file():
        print(f"Missing pip-licenses report: {licenses_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        payload = json.loads(licenses_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Malformed pip-licenses report {licenses_path}: {exc}", file=sys.stderr)
        raise SystemExit(2)

    try:
        scan_results = parse_pip_licenses_output(payload)
        row = find_target_package(scan_results, TARGET_PACKAGE_NAME)
        license_text = extract_package_license(row)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    if not license_text or normalize_license_text(license_text) in {"UNKNOWN", "N/A", "NONE"}:
        print("Target package license is empty or UNKNOWN.", file=sys.stderr)
        raise SystemExit(2)

    policy_verdict = classify_license_policy(license_text)
    if policy_verdict == "allowed":
        print(
            f"License {license_text!r} is allowed; blocker-only scenario cannot proceed.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    if policy_verdict == "unknown":
        print(
            f"License {license_text!r} is unknown; insufficient evidence for ER-054.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    evidence = build_license_evidence(
        package_name=TARGET_PACKAGE_NAME,
        package_version=str(row.get("Version") or row.get("version") or TARGET_PACKAGE_VERSION),
        license_text=license_text,
    )
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ZeroUI ER-054 license evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
