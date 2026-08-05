#!/usr/bin/env python3
"""Build ER-057 promotion-rejection evidence from an exact GitHub workflow run."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

EVIDENCE_PATH = Path(".zeroui-simulator/ci/current/er057/artifact/promotion-rejection-evidence.json")
DETECTOR_TRIGGER_PATH = Path(".zeroui-simulator/ci/current/er057/detector-trigger.json")
TARGET_WORKFLOW_PATH = ".github/workflows/zeroui-fm1-simulator-er057-promotion-gate.yml"
TARGET_ENVIRONMENT = "dev"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Detector trigger file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("Detector trigger file must be a JSON object")
    return payload


def trigger_required(name: str, trigger: dict) -> str:
    value = trigger.get(name)
    if value in (None, ""):
        raise SystemExit(f"Missing required detector trigger field: {name}")
    return str(value).strip() if not isinstance(value, (int, float)) else value


def fetch_workflow_run(run_id: str) -> dict:
    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    repository = str(os.getenv("GITHUB_REPOSITORY") or "").strip()
    endpoint_path = (
        f"/repos/{repository}/actions/runs/{run_id}" if repository else f"/actions/runs/{run_id}"
    )
    print(f"github_token_present={'YES' if token else 'NO'}")
    print(f"github_repository_present={'YES' if repository else 'NO'}")
    print(f"target_run_id={run_id}")
    print(f"requested_api_endpoint={endpoint_path}")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    if not repository:
        raise SystemExit("GITHUB_REPOSITORY is required")
    url = f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"http_status={exc.code}")
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub workflow run fetch failed ({exc.code}): {body}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("GitHub workflow run response must be a JSON object")
    return payload


def build_evidence(run: dict, trigger: dict, repository: str) -> dict:
    conclusion = str(run.get("conclusion") or "").strip().lower()
    if conclusion not in {"failure", "failed"}:
        raise SystemExit(f"Target workflow conclusion must be failure, got {conclusion!r}")
    if str(trigger.get("review_state") or "") != "rejected":
        raise SystemExit("review_state must be rejected")
    if trigger.get("review_api_verified") is not True:
        raise SystemExit("review_api_verified must be true")
    if int(trigger.get("review_http_status") or 0) != 200:
        raise SystemExit("review_http_status must be 200")
    rejection_reason = str(trigger.get("rejection_reason") or "").strip()
    if not rejection_reason:
        raise SystemExit("rejection_reason is required")
    run_id = int(run.get("id") or 0)
    environment_id = int(trigger.get("environment_id") or 0)
    promotion_decision_id = f"gha-env-review-{run_id}-{environment_id}-rejected"
    evidence = {
        "event_type_id": "ci.promotion.rejected",
        "promotion_status": "rejected",
        "target_environment": str(trigger.get("target_environment") or TARGET_ENVIRONMENT),
        "promotion_family": "release_promotion",
        "promotion_gate": "github_environment_review",
        "promotion_rejected": True,
        "promotion_approved": False,
        "rejection_reason": rejection_reason,
        "review_state": "rejected",
        "review_api_verified": True,
        "review_http_status": 200,
        "workflow_completed": True,
        "workflow_conclusion": "failure",
        "scan_passed": False,
        "promotion_decision_id": promotion_decision_id,
        "target_workflow_name": str(trigger.get("target_workflow_name") or run.get("name") or ""),
        "target_workflow_path": TARGET_WORKFLOW_PATH,
        "target_workflow_run_id": run_id,
        "target_workflow_run_attempt": int(run.get("run_attempt") or 1),
        "target_workflow_html_url": str(trigger.get("target_workflow_html_url") or run.get("html_url") or ""),
        "target_branch": str(trigger.get("target_branch") or ""),
        "target_commit_sha": str(trigger.get("target_commit_sha") or ""),
        "target_repository": repository,
        "environment_id": environment_id,
        "review_comment": str(trigger.get("review_comment") or ""),
        "reason": "promotion_rejected",
    }
    if trigger.get("requested_by"):
        evidence["requested_by"] = str(trigger.get("requested_by"))
    if trigger.get("reviewed_by"):
        evidence["reviewed_by"] = str(trigger.get("reviewed_by"))
    return evidence


def main() -> None:
    trigger = read_json(DETECTOR_TRIGGER_PATH)
    expected_run_id = str(trigger_required("target_workflow_run_id", trigger))
    expected_attempt = str(trigger_required("target_workflow_run_attempt", trigger))
    expected_path = str(trigger_required("target_workflow_path", trigger))
    expected_branch = str(trigger_required("target_branch", trigger))
    expected_commit_sha = str(trigger_required("target_commit_sha", trigger))
    expected_repository = str(os.getenv("GITHUB_REPOSITORY") or "").strip()
    if not expected_repository:
        raise SystemExit("GITHUB_REPOSITORY is required")
    if expected_path != TARGET_WORKFLOW_PATH:
        raise SystemExit("TARGET_WORKFLOW_PATH is not the managed ER-057 promotion gate")
    target_conclusion = str(trigger.get("target_workflow_conclusion") or "").strip().lower()
    if target_conclusion not in {"failure", "failed"}:
        raise SystemExit("detector trigger target_workflow_conclusion must be failure")

    detector_run_id = str(os.getenv("GITHUB_RUN_ID") or "").strip()
    if detector_run_id and detector_run_id == expected_run_id:
        raise SystemExit("Detector workflow run ID must not be used as the target workflow run ID")

    run = fetch_workflow_run(expected_run_id)
    if str(run.get("id") or "") != expected_run_id:
        raise SystemExit("Fetched workflow run ID does not match input")
    if str(run.get("path") or "") != expected_path:
        raise SystemExit("Target workflow path mismatch")
    if str(run.get("head_branch") or "") != expected_branch:
        raise SystemExit("Target workflow branch mismatch")
    if str(run.get("head_sha") or "").strip() != expected_commit_sha.strip():
        raise SystemExit("Target workflow commit SHA mismatch")
    if str(run.get("run_attempt") or "1") != expected_attempt:
        raise SystemExit("Target workflow run attempt mismatch")
    if str(run.get("status") or "") != "completed":
        raise SystemExit("Target workflow must be completed before evidence generation")

    conclusion = str(run.get("conclusion") or "").strip().lower()
    if conclusion == "cancelled":
        raise SystemExit("Target workflow cancelled; cannot treat cancellation as rejection")
    if conclusion in {"timed_out", "timeout", "stale"}:
        raise SystemExit("Target workflow timed out; cannot treat timeout as rejection")
    if conclusion in {"success", "passed"}:
        raise SystemExit("Target workflow succeeded; blocker-only scenario cannot proceed")

    evidence = build_evidence(run, trigger, repository=expected_repository)
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ZeroUI ER-057 promotion-rejection evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
