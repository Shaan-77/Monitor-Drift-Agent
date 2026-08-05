#!/usr/bin/env python3
"""Build ER-056 workflow-cancelled evidence from an exact GitHub workflow run."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

EVIDENCE_PATH = Path(".zeroui-simulator/ci/current/er056/artifact/workflow-cancelled-evidence.json")
DETECTOR_TRIGGER_PATH = Path(".zeroui-simulator/ci/current/er056/detector-trigger.json")
TARGET_WORKFLOW_PATH = ".github/workflows/zeroui-fm1-simulator-er056-required-gate.yml"


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


def build_evidence(run: dict, *, branch: str, commit_sha: str, repository: str) -> dict:
    conclusion = str(run.get("conclusion") or "").strip().lower()
    if conclusion != "cancelled":
        raise SystemExit(f"Target workflow conclusion must be cancelled, got {conclusion!r}")
    return {
        "event_type_id": "ci.workflow.cancelled",
        "required_gate": True,
        "workflow_family": "release_as_code",
        "stage": "release",
        "workflow_completed": False,
        "workflow_cancelled": True,
        "workflow_conclusion": "cancelled",
        "scan_passed": False,
        "target_workflow_name": str(run.get("name") or ""),
        "target_workflow_path": TARGET_WORKFLOW_PATH,
        "target_workflow_run_id": int(run.get("id") or 0),
        "target_workflow_run_attempt": int(run.get("run_attempt") or 1),
        "target_workflow_html_url": str(run.get("html_url") or ""),
        "target_branch": branch,
        "target_commit_sha": commit_sha,
        "target_repository": repository,
        "cancellation_source": "simulator_github_provider",
        "cancellation_verified_from": "github_actions_api",
        "reason": "required_workflow_cancelled",
    }


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
    target_conclusion = str(trigger.get("target_conclusion") or "").strip().lower()
    if target_conclusion != "cancelled":
        raise SystemExit("detector trigger target_conclusion must be cancelled")
    if expected_path != TARGET_WORKFLOW_PATH:
        raise SystemExit("TARGET_WORKFLOW_PATH is not the managed ER-056 required gate")

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
    if conclusion == "failure":
        raise SystemExit("Target workflow failed; cannot treat failure as cancellation")
    if conclusion in {"timed_out", "timeout", "stale"}:
        raise SystemExit("Target workflow timed out; cannot treat timeout as cancellation")
    if conclusion in {"success", "passed"}:
        raise SystemExit("Target workflow succeeded; blocker-only scenario cannot proceed")

    evidence = build_evidence(
        run,
        branch=expected_branch,
        commit_sha=expected_commit_sha,
        repository=expected_repository,
    )
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ZeroUI ER-056 workflow-cancelled evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
