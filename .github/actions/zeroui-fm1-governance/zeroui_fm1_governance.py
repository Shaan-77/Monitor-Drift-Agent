from __future__ import annotations

import http.client
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


SUPPORTED_EVENT_BY_FORMAT = {
    "bandit": "ci.security_scan.failed",
}


def read_json_file(path: str) -> Any:
    evidence_path = Path(path)
    if not evidence_path.exists():
        raise SystemExit(f"Evidence file not found: {path}")
    with evidence_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bandit_context(evidence: Any) -> Dict[str, Any]:
    if not isinstance(evidence, dict):
        raise SystemExit("Bandit evidence must be a JSON object.")

    results = evidence.get("results", [])
    if not isinstance(results, list):
        results = []

    findings: List[Dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        findings.append({
            "file": item.get("filename"),
            "line": item.get("line_number"),
            "test_id": item.get("test_id"),
            "test_name": item.get("test_name"),
            "severity": item.get("issue_severity"),
            "confidence": item.get("issue_confidence"),
            "message": item.get("issue_text"),
        })

    high_or_critical = [
        item for item in findings
        if str(item.get("severity", "")).upper() in {"HIGH", "CRITICAL"}
    ]
    medium = [
        item for item in findings
        if str(item.get("severity", "")).upper() == "MEDIUM"
    ]

    return {
        "scan_type": "sast",
        "scanner": "bandit",
        "source_evidence_format": "bandit",
        "severity": "critical" if high_or_critical else ("medium" if medium else "none"),
        "finding_count": len(findings),
        "high_or_critical_count": len(high_or_critical),
        "medium_count": len(medium),
        "scan_passed": len(findings) == 0,
        "findings": findings,
    }


def context_from_evidence(evidence_file: str, evidence_format: str) -> Dict[str, Any]:
    normalized_format = evidence_format.strip().lower()
    evidence = read_json_file(evidence_file)

    if normalized_format == "bandit":
        return bandit_context(evidence)

    raise SystemExit(f"Unsupported evidence_format: {evidence_format}")


def default_branch_from_event() -> Optional[str]:
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        with open(event_path, "r", encoding="utf-8") as handle:
            event = json.load(handle)
    except Exception:
        return None

    repository = event.get("repository") if isinstance(event, dict) else None
    if isinstance(repository, dict):
        value = repository.get("default_branch")
        if value:
            return str(value)
    return None


def release_branch() -> str:
    return (
        os.getenv("GITHUB_BASE_REF")
        or default_branch_from_event()
        or os.getenv("GITHUB_REF_NAME")
        or "main"
    )


def build_payload(event_type_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    run_id = os.getenv("GITHUB_RUN_ID") or "unknown-run"
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT") or "1"
    repository = os.getenv("GITHUB_REPOSITORY") or "unknown-repository"
    branch = release_branch()
    workflow_name = os.getenv("GITHUB_WORKFLOW") or "unknown-workflow"
    job_id = os.getenv("GITHUB_JOB") or "zeroui-governance-gate"

    return {
        "schema_version": 1,
        "trace_id": f"trace-gha-{event_type_id}-{run_id}-{run_attempt}",
        "source_event_id": f"gha-{event_type_id}-{run_id}-{job_id}-{run_attempt}",
        "provider": "github_actions",
        "platform": "GitHub Actions",
        "source_system": "ci_cd",
        "signal_method": "pipeline_job",
        "event_type_id": event_type_id,
        "repository": repository,
        "branch": branch,
        "commit_hash": os.getenv("GITHUB_SHA") or "unknown-commit",
        "workflow_name": workflow_name,
        "workflow_run_id": run_id,
        "pipeline_id": run_id,
        "job_id": job_id,
        "run_id": run_id,
        "payload": context,
        "raw_metadata": {
            "github_actions": True,
            "github_event_name": os.getenv("GITHUB_EVENT_NAME"),
            "github_actor": os.getenv("GITHUB_ACTOR"),
            "github_actor_id": os.getenv("GITHUB_ACTOR_ID"),
            "source_branch": os.getenv("GITHUB_REF_NAME"),
            "release_branch": branch,
            "repository": repository,
        },
    }


def _fm1_signal_connection(base_url: str) -> Tuple[http.client.HTTPConnection, str]:
    parsed = urlparse(base_url)

    if parsed.scheme not in {"http", "https"}:
        raise SystemExit("FM1_BASE_URL must use http or https.")
    if not parsed.hostname:
        raise SystemExit("FM1_BASE_URL must include a hostname.")
    if parsed.username or parsed.password:
        raise SystemExit("FM1_BASE_URL must not include credentials.")
    if parsed.params or parsed.query or parsed.fragment:
        raise SystemExit("FM1_BASE_URL must not include params, query, or fragment.")

    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    base_path = parsed.path.rstrip("/")
    signal_path = f"{base_path}/fm1/v1/ci/signals" if base_path else "/fm1/v1/ci/signals"

    connection = connection_class(parsed.hostname, port=parsed.port, timeout=60)
    return connection, signal_path


def post_signal(payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = (os.getenv("FM1_BASE_URL") or "").rstrip("/")
    token = os.getenv("FM1_SIGNAL_INTAKE_AUTH_TOKEN") or ""

    if not base_url:
        raise SystemExit("FM1_BASE_URL is missing.")
    if not token:
        raise SystemExit("FM1_SIGNAL_INTAKE_AUTH_TOKEN is missing.")

    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    connection, signal_path = _fm1_signal_connection(base_url)

    try:
        connection.request("POST", signal_path, body=body_bytes, headers=headers)
        response = connection.getresponse()
        status = response.status
        body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise SystemExit(f"FM-1 request failed: {exc}") from exc
    finally:
        connection.close()

    print(f"FM-1 HTTP status: {status}")

    if status < 200 or status >= 300:
        print("FM-1 error response:")
        print(body)
        raise SystemExit(1)

    try:
        return json.loads(body)
    except Exception as exc:
        print("FM-1 non-JSON response:")
        print(body)
        raise SystemExit(f"FM-1 response was not JSON: {exc}") from exc


def pick(response: Dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = response
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, "", []):
            return current
    return None


def write_summary(status_line: str, result_line: str, fields: Dict[str, Any]) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("## ZeroUI Governance Check\n\n")
        summary.write(f"{status_line}\n\n")
        summary.write("| Field | Value |\n")
        summary.write("|---|---|\n")
        for key, value in fields.items():
            summary.write(f"| {key} | {value} |\n")
        summary.write(f"| Result | {result_line} |\n")


def enforce_response(response: Dict[str, Any]) -> None:
    er = pick(response, "mapped_trigger_id", "trigger_id", "routing.mapped_trigger_id", "projections.ci_response.mapped_trigger_id")
    decision = pick(response, "decision_outcome", "decision.decision_outcome", "projections.ci_response.decision_outcome")
    policy_source = pick(response, "policy_source", "projections.ci_response.policy_source")
    receipt_status = pick(response, "receipt_status", "intake_receipt_status", "decision_receipt_status", "projections.ci_response.receipt_status")
    receipt_ref = pick(response, "receipt_ref", "projections.ci_response.receipt_ref", "decision.receipt_ref")
    blockers = pick(response, "active_blocker_ids", "projections.ci_response.active_blocker_ids") or []

    if not isinstance(blockers, list):
        blockers = [blockers]

    receipt_present = receipt_status == "receipt_written" or bool(receipt_ref)
    blocker_count = len(blockers)

    print(f"er={er}")
    print(f"decision_outcome={decision}")
    print(f"policy_source={policy_source}")
    print(f"receipt_status={receipt_status}")
    print(f"receipt_ref_present={bool(receipt_ref)}")
    print(f"active_blocker_count={blocker_count}")

    summary_fields = {
        "ER": er,
        "Decision": decision,
        "Policy source": policy_source,
        "Receipt status": receipt_status or "n/a",
        "Receipt present": receipt_present,
        "Active blocker count": blocker_count,
    }

    if policy_source != "DB_POLICY_SOURCE":
        write_summary("❌ ZeroUI governance check failed.", "Invalid policy source", summary_fields)
        print("::error title=ZeroUI governance check failed::policy_source is not DB_POLICY_SOURCE.")
        raise SystemExit(1)

    if not receipt_present:
        write_summary("❌ ZeroUI governance check failed.", "Receipt was not written", summary_fields)
        print("::error title=ZeroUI governance check failed::Receipt was not written.")
        raise SystemExit(1)

    if decision in {"hard_block", "soft_block"}:
        write_summary("❌ ZeroUI governance blocked this pipeline.", "Pipeline blocked by ZeroUI", summary_fields)
        print(f"::error title=ZeroUI governance blocked::ZeroUI returned {decision} for {er}. Active blockers: {blocker_count}.")
        print("ZeroUI governance blocked this pipeline.")
        raise SystemExit(1)

    if decision == "pass":
        write_summary("✅ ZeroUI governance check passed.", "Pipeline unblocked by ZeroUI", summary_fields)
        print(f"::notice title=ZeroUI governance check passed::ZeroUI returned pass for {er}. Receipt written. Active blockers: {blocker_count}.")
        print("ZeroUI governance check passed. Pipeline may continue.")
        return

    write_summary("❌ ZeroUI governance check failed.", "Unknown decision", summary_fields)
    print("::error title=ZeroUI governance check failed::Unknown or missing decision_outcome.")
    raise SystemExit(1)


def main() -> None:
    evidence_file = os.getenv("INPUT_EVIDENCE_FILE") or ""
    evidence_format = (os.getenv("INPUT_EVIDENCE_FORMAT") or "").strip().lower()
    explicit_event_type = (os.getenv("INPUT_EVENT_TYPE_ID") or "").strip()

    if not evidence_file:
        raise SystemExit("evidence_file input is required.")
    if not evidence_format:
        raise SystemExit("evidence_format input is required.")

    event_type_id = explicit_event_type or SUPPORTED_EVENT_BY_FORMAT.get(evidence_format)
    if not event_type_id:
        raise SystemExit(f"Cannot infer event_type_id for evidence_format: {evidence_format}")

    context = context_from_evidence(evidence_file, evidence_format)
    print("ZeroUI evidence context summary:")
    print(json.dumps({
        "finding_count": context.get("finding_count"),
        "severity": context.get("severity"),
        "scan_passed": context.get("scan_passed"),
    }, indent=2, sort_keys=True))

    payload = build_payload(event_type_id, context)
    print("ZeroUI FM-1 governance gate prepared signal.")
    print(f"event_type_id={event_type_id}")
    print(f"repository={payload.get('repository')}")
    print(f"branch={payload.get('branch')}")
    print(f"workflow_run_id={payload.get('workflow_run_id')}")

    response = post_signal(payload)
    enforce_response(response)


if __name__ == "__main__":
    main()
