#!/usr/bin/env python3
"""Build ER-055 SBOM presence evidence from the release artifact manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST_PATH = Path(".zeroui-simulator/ci/current/er055/artifact/release-artifact-manifest.json")
EVIDENCE_PATH = Path(".zeroui-simulator/ci/current/er055/artifact/sbom-evidence.json")
MANAGED_ARTIFACT_PREFIX = ".zeroui-simulator/ci/current/er055/artifact/"


def normalize_posix_path(path: str) -> str:
    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def is_safe_managed_sbom_path(path: str) -> bool:
    normalized = normalize_posix_path(path)
    if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
        return False
    if not normalized.startswith(MANAGED_ARTIFACT_PREFIX):
        return False
    return normalized.endswith("sbom.spdx.json")


def is_valid_spdx_json_payload(payload) -> bool:
    return isinstance(payload, dict) and bool(payload.get("spdxVersion") or payload.get("SPDXID"))


def classify_sbom_file_content(content: bytes) -> str:
    if content[:2] == b"PK":
        return "docx_like"
    text = content.decode("utf-8", errors="replace").strip()
    if not text:
        return "malformed"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "malformed"
    if is_valid_spdx_json_payload(payload):
        return "valid_spdx"
    return "malformed"


def validate_release_manifest(manifest) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    errors = []
    if manifest.get("release_bound") is not True:
        errors.append("release_bound must be true")
    if manifest.get("sbom_required") is not True:
        errors.append("sbom_required must be true")
    if str(manifest.get("expected_format") or "").strip().upper() != "SPDX":
        errors.append("expected_format must be SPDX")
    expected_path = normalize_posix_path(str(manifest.get("expected_sbom_path") or ""))
    if not expected_path:
        errors.append("expected_sbom_path is required")
    elif not is_safe_managed_sbom_path(expected_path):
        errors.append("expected_sbom_path must be inside the managed ER-055 artifact directory")
    for key in ("artifact_name", "artifact_version", "artifact_ref"):
        if not str(manifest.get(key) or "").strip():
            errors.append(f"{key} is required")
    return errors


def build_sbom_presence_evidence(*, manifest: dict, sbom_present: bool) -> dict:
    expected_path = normalize_posix_path(str(manifest.get("expected_sbom_path") or ".zeroui-simulator/ci/current/er055/artifact/sbom.spdx.json"))
    missing = not sbom_present
    return {
        "event_type_id": "ci.sbom.missing",
        "sbom_scanner": "zeroui-sbom-presence-check",
        "sbom_required": True,
        "sbom_present": sbom_present,
        "expected_format": "SPDX",
        "sbom_format": "spdx-json",
        "artifact_release_bound": True,
        "artifact_ref": str(manifest.get("artifact_ref") or ""),
        "artifact_name": str(manifest.get("artifact_name") or ""),
        "artifact_version": str(manifest.get("artifact_version") or ""),
        "expected_sbom_path": expected_path,
        "validation_passed": not missing,
        "scan_passed": not missing,
        "policy_verdict": "missing_required_sbom" if missing else "sbom_present",
        "missing_artifact_count": 1 if missing else 0,
        "reason": "required_sbom_missing" if missing else "required_sbom_present",
        "sbom_metadata": {},
    }


def main() -> None:
    if not MANIFEST_PATH.is_file():
        print(f"Missing release artifact manifest: {MANIFEST_PATH}", file=sys.stderr)
        raise SystemExit(2)
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Malformed manifest {MANIFEST_PATH}: {exc}", file=sys.stderr)
        raise SystemExit(2)

    errors = validate_release_manifest(manifest)
    if errors:
        print("; ".join(errors), file=sys.stderr)
        raise SystemExit(2)

    expected_path = normalize_posix_path(str(manifest.get("expected_sbom_path") or ""))
    if not is_safe_managed_sbom_path(expected_path):
        print("Unsafe expected SBOM path.", file=sys.stderr)
        raise SystemExit(2)

    sbom_path = Path(expected_path)
    if not sbom_path.is_file():
        evidence = build_sbom_presence_evidence(manifest=manifest, sbom_present=False)
        EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("ZeroUI ER-055 SBOM evidence:")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        raise SystemExit(0)

    content = sbom_path.read_bytes()
    classification = classify_sbom_file_content(content)
    if classification == "valid_spdx":
        print("Required SBOM is present and valid; blocker-only scenario cannot proceed.", file=sys.stderr)
        raise SystemExit(3)
    print("SBOM file is present but malformed; cannot claim required SBOM is missing.", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
