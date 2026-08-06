#!/usr/bin/env python3
"""Build ER-068 workflow-failure evidence from the exact GitHub target job."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


EVENT_TYPE_ID = "ci.workflow.failed"
TARGET_JOB_DISPLAY_NAME = "ZeroUI FM-1 ER-068 Required Release Workflow"
WORKFLOW_PATH = ".github/workflows/zeroui-fm1-simulator-er068.yml"
WORKFLOW_NAME = "ZeroUI FM-1 ER-068 Workflow Failure"
EVIDENCE_PATH = Path(".zeroui-simulator/ci/current/er068/artifact/workflow-failure-evidence.json")


def classify_required_workflow_needs_result(result: str | None) -> str:
    normalized = str(result or "").strip().lower()
    if normalized in {"failure", "failed"}:
        return "failure"
    if normalized in {"success", "passed"}:
        return "success"
    if normalized == "cancelled":
        return "cancelled"
    if normalized == "skipped":
        return "skipped"
    if normalized in {"timed_out", "timeout", "stale"}:
        return "timed_out"
    return "unknown"


def select_target_jobs_by_display_name(jobs: list[dict[str, Any]], display_name: str) -> list[dict[str, Any]]:
    expected = str(display_name or "").strip()
    return [
        job
        for job in jobs
        if isinstance(job, dict) and str(job.get("name") or "").strip() == expected
    ]


def validate_target_job_completed_failure(job: dict[str, Any]) -> None:
    status = str(job.get("status") or "").strip().lower()
    conclusion = str(job.get("conclusion") or "").strip().lower()
    if conclusion == "cancelled":
        raise SystemExit("ER068_TARGET_WORKFLOW_CANCELLED")
    if conclusion == "skipped":
        raise SystemExit("ER068_TARGET_WORKFLOW_SKIPPED")
    if conclusion in {"timed_out", "timeout", "stale"}:
        raise SystemExit("ER068_TARGET_WORKFLOW_TIMED_OUT")
    if status != "completed":
        raise SystemExit(f"target job status must be completed, got {status!r}")
    if conclusion in {"success", "passed"}:
        raise SystemExit("ER068_EXPECTED_WORKFLOW_FAILURE_NOT_OBSERVED")
    if conclusion not in {"failure", "failed"}:
        raise SystemExit("ER068_EXPECTED_WORKFLOW_FAILURE_NOT_OBSERVED")


def build_workflow_failure_evidence(
    *,
    target_job: dict[str, Any],
    workflow_run_id: int | str,
    workflow_run_attempt: int | str,
    workflow_html_url: str,
    workflow_path: str,
    workflow_name: str,
    target_branch: str,
    target_commit_sha: str,
    target_repository: str,
) -> dict[str, Any]:
    validate_target_job_completed_failure(target_job)
    return {
        "event_type_id": EVENT_TYPE_ID,
        "required_gate": True,
        "workflow_family": "release_as_code",
        "stage": "release",
        "workflow_completed": False,
        "workflow_failed": True,
        "workflow_cancelled": False,
        "workflow_conclusion": "failure",
        "scan_passed": False,
        "failure_kind": "required_release_workflow_failure",
        "reason": "required_workflow_failed",
        "target_workflow_name": workflow_name,
        "target_workflow_path": workflow_path,
        "target_workflow_run_id": int(workflow_run_id),
        "target_workflow_run_attempt": int(workflow_run_attempt),
        "target_workflow_html_url": str(workflow_html_url or ""),
        "target_branch": str(target_branch or ""),
        "target_commit_sha": str(target_commit_sha or ""),
        "target_repository": str(target_repository or ""),
        "target_job_name": str(target_job.get("name") or TARGET_JOB_DISPLAY_NAME),
        "target_job_id": int(target_job.get("id") or 0),
        "target_job_conclusion": "failure",
        "target_job_status": "completed",
        "target_job_html_url": str(target_job.get("html_url") or ""),
    }


def fetch_workflow_jobs(run_id: str) -> list[dict[str, Any]]:
    token = str(os.getenv("GITHUB_TOKEN") or "").strip()
    repository = str(os.getenv("GITHUB_REPOSITORY") or "").strip()
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    if not repository:
        raise SystemExit("GITHUB_REPOSITORY is required")
    url = f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/jobs?per_page=20"
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
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub workflow jobs fetch failed ({exc.code}): {body}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("GitHub workflow jobs response must be a JSON object")
    jobs = payload.get("jobs") or []
    if not isinstance(jobs, list):
        raise SystemExit("GitHub workflow jobs response jobs must be a list")
    return [row for row in jobs if isinstance(row, dict)]


def main() -> None:
    run_id = str(os.getenv("GITHUB_RUN_ID") or "").strip()
    run_attempt = str(os.getenv("GITHUB_RUN_ATTEMPT") or "1").strip()
    repository = str(os.getenv("GITHUB_REPOSITORY") or "").strip()
    branch = str(os.getenv("GITHUB_REF_NAME") or "").strip()
    commit_sha = str(os.getenv("GITHUB_SHA") or "").strip()
    workflow_html_url = (
        f"https://github.com/{repository}/actions/runs/{run_id}" if repository and run_id else ""
    )
    if not run_id:
        raise SystemExit("GITHUB_RUN_ID is required")

    needs_result = str(os.getenv("REQUIRED_WORKFLOW_RESULT") or "").strip()
    classified = classify_required_workflow_needs_result(needs_result)
    if classified == "success":
        raise SystemExit("ER068_EXPECTED_WORKFLOW_FAILURE_NOT_OBSERVED")
    if classified == "cancelled":
        raise SystemExit("ER068_TARGET_WORKFLOW_CANCELLED")
    if classified == "skipped":
        raise SystemExit("ER068_TARGET_WORKFLOW_SKIPPED")
    if classified == "timed_out":
        raise SystemExit("ER068_TARGET_WORKFLOW_TIMED_OUT")
    if classified != "failure":
        raise SystemExit("ER068_TARGET_JOB_VERIFICATION_FAILED")

    jobs = fetch_workflow_jobs(run_id)
    matches = select_target_jobs_by_display_name(jobs, TARGET_JOB_DISPLAY_NAME)
    if len(matches) == 0:
        raise SystemExit("ER068_TARGET_JOB_NOT_IDENTIFIED")
    if len(matches) != 1:
        raise SystemExit("ER068_TARGET_JOB_IDENTITY_AMBIGUOUS")

    target_job = matches[0]
    evidence = build_workflow_failure_evidence(
        target_job=target_job,
        workflow_run_id=run_id,
        workflow_run_attempt=run_attempt,
        workflow_html_url=workflow_html_url,
        workflow_path=WORKFLOW_PATH,
        workflow_name=WORKFLOW_NAME,
        target_branch=branch,
        target_commit_sha=commit_sha,
        target_repository=repository,
    )

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("ZeroUI ER-068 workflow failure evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main()
