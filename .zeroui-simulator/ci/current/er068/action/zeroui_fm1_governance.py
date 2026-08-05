from __future__ import annotations

import http.client
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


SCENARIO_PATH = Path(".zeroui-simulator/ci/current/er068/scenario.json")
EXPECTED_ER = "ER-068"
TARGET_JOB_DISPLAY_NAME = "ZeroUI FM-1 ER-068 Required Release Workflow"
WORKFLOW_PATH = ".github/workflows/zeroui-fm1-simulator-er068.yml"


def build_github_actions_source_event_id(event_type_id, workflow_run_id, github_job_key, run_attempt):
    return f"gha-{event_type_id}-{workflow_run_id}-{github_job_key}-{run_attempt}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def github_workflow_failure_context(evidence: Any) -> Dict[str, Any]:
    data = evidence if isinstance(evidence, dict) else {}
    required_keys = [
        "required_gate",
        "workflow_family",
        "stage",
        "workflow_failed",
        "workflow_conclusion",
        "workflow_cancelled",
        "scan_passed",
        "failure_kind",
        "reason",
        "target_workflow_path",
        "target_workflow_run_id",
        "target_workflow_run_attempt",
        "target_branch",
        "target_commit_sha",
        "target_repository",
        "target_job_name",
        "target_job_id",
        "target_job_conclusion",
        "target_job_status",
    ]
    for key in required_keys:
        if key not in data:
            raise SystemExit(f"Missing required evidence field: {key}")

    if not isinstance(data["required_gate"], bool) or data["required_gate"] is not True:
        raise SystemExit("required_gate must be true")
    if str(data["workflow_family"]) != "release_as_code":
        raise SystemExit("workflow_family must be release_as_code")
    if str(data["stage"]) != "release":
        raise SystemExit("stage must be release")
    if not isinstance(data["workflow_failed"], bool) or data["workflow_failed"] is not True:
        raise SystemExit("workflow_failed must be true")
    if str(data["workflow_conclusion"]).strip().lower() != "failure":
        raise SystemExit("workflow_conclusion must be failure")
    if not isinstance(data["workflow_cancelled"], bool) or data["workflow_cancelled"] is not False:
        raise SystemExit("workflow_cancelled must be false")
    if not isinstance(data["scan_passed"], bool) or data["scan_passed"] is not False:
        raise SystemExit("scan_passed must be false")
    if str(data["failure_kind"]) != "required_release_workflow_failure":
        raise SystemExit("failure_kind must be required_release_workflow_failure")
    if str(data["reason"]) != "required_workflow_failed":
        raise SystemExit("reason must be required_workflow_failed")
    if str(data["target_workflow_path"]) != WORKFLOW_PATH:
        raise SystemExit("target_workflow_path must match the managed ER-068 workflow")
    if str(data["target_job_name"]) != TARGET_JOB_DISPLAY_NAME:
        raise SystemExit("target_job_name must match the managed ER-068 target job display name")
    if str(data["target_job_conclusion"]).strip().lower() != "failure":
        raise SystemExit("target_job_conclusion must be failure")
    if str(data["target_job_status"]).strip().lower() != "completed":
        raise SystemExit("target_job_status must be completed")

    try:
        target_run_id = int(data["target_workflow_run_id"])
        target_run_attempt = int(data["target_workflow_run_attempt"])
        target_job_id = int(data["target_job_id"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"target workflow/job identity must be numeric: {exc}") from exc

    return {
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
        "target_workflow_name": str(data.get("target_workflow_name") or ""),
        "target_workflow_path": str(data["target_workflow_path"]),
        "target_workflow_run_id": target_run_id,
        "target_workflow_run_attempt": target_run_attempt,
        "target_workflow_html_url": str(data.get("target_workflow_html_url") or ""),
        "target_branch": str(data["target_branch"]),
        "target_commit_sha": str(data["target_commit_sha"]),
        "target_repository": str(data["target_repository"]),
        "target_job_name": str(data["target_job_name"]),
        "target_job_id": target_job_id,
        "target_job_conclusion": "failure",
        "target_job_status": "completed",
        "target_job_html_url": str(data.get("target_job_html_url") or ""),
    }


SUPPORTED_EVENT_BY_FORMAT = {
    "github-workflow-failure": "ci.workflow.failed",
}


def context_from_evidence(evidence_file: Path, evidence_format: str) -> Tuple[str, Dict[str, Any]]:
    fmt = evidence_format.strip().lower()
    if fmt == "github-workflow-failure":
        return "ci.workflow.failed", github_workflow_failure_context(read_json(evidence_file))
    raise SystemExit(f"Unsupported evidence_format: {evidence_format}")


def load_scenario_metadata() -> Dict[str, Any]:
    if not SCENARIO_PATH.exists():
        return {}
    try:
        payload = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def release_branch(scenario: Dict[str, Any]) -> str:
    scenario_branch = str(scenario.get("branch") or "").strip()
    if scenario_branch:
        return scenario_branch

    github_head_ref = (os.getenv("GITHUB_HEAD_REF") or "").strip()
    if github_head_ref:
        return github_head_ref

    github_ref_name = (os.getenv("GITHUB_REF_NAME") or "").strip()
    if github_ref_name:
        return github_ref_name

    github_ref = (os.getenv("GITHUB_REF") or "").strip()
    if github_ref.startswith("refs/heads/"):
        return github_ref[len("refs/heads/"):]

    return default_branch_from_event() or "main"


def build_payload(event_type_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    scenario = load_scenario_metadata()
    run_id = os.getenv("GITHUB_RUN_ID") or "unknown-run"
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT") or "1"
    repository = os.getenv("GITHUB_REPOSITORY") or str(scenario.get("repository") or "unknown-repository")
    branch = release_branch(scenario)
    workflow_name = os.getenv("GITHUB_WORKFLOW") or str(scenario.get("workflow_name") or "unknown-workflow")
    job_id = os.getenv("ZEROUI_JOB_ID") or os.getenv("GITHUB_JOB") or "zeroui-governance-gate"
    trace_id = str(scenario.get("trace_id") or f"trace-gha-{event_type_id}-{run_id}-{run_attempt}")
    simulator_run_id = str(scenario.get("simulator_run_id") or "").strip()
    source_event_id = build_github_actions_source_event_id(
        event_type_id,
        run_id,
        job_id,
        run_attempt,
    )

    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "source_event_id": source_event_id,
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
            "ci_runner_marker": "github_actions",
            "ci_runner_detected": True,
            "github_event_name": os.getenv("GITHUB_EVENT_NAME"),
            "github_actor": os.getenv("GITHUB_ACTOR"),
            "github_actor_id": os.getenv("GITHUB_ACTOR_ID"),
            "source_branch": os.getenv("GITHUB_REF_NAME"),
            "release_branch": branch,
            "repository": repository,
            "simulator_run_id": simulator_run_id or None,
            "recipe_id": scenario.get("recipe_id"),
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
        parsed = json.loads(body)
    except Exception as exc:
        print("FM-1 non-JSON response:")
        print(body)
        raise SystemExit(f"FM-1 response was not JSON: {exc}") from exc

    normalized_event_id = pick(
        parsed,
        "normalized_event_id",
        "projections.ci_response.normalized_event_id",
    )
    reason_code = pick(parsed, "reason_code", "projections.ci_response.reason_code")
    mapped_trigger_id = pick(
        parsed,
        "mapped_trigger_id",
        "trigger_id",
        "projections.ci_response.mapped_trigger_id",
    )
    decision_outcome = pick(
        parsed,
        "decision_outcome",
        "decision.decision_outcome",
        "projections.ci_response.decision_outcome",
    )
    print(f"normalized_event_id={normalized_event_id or 'n/a'}")
    print(f"reason_code={reason_code or 'n/a'}")
    print(f"mapped_trigger_id={mapped_trigger_id or 'n/a'}")
    print(f"decision_outcome={decision_outcome or 'n/a'}")
    return parsed


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
        "Normalized event ID": pick(response, "normalized_event_id", "projections.ci_response.normalized_event_id") or "n/a",
        "Reason code": pick(response, "reason_code", "projections.ci_response.reason_code") or "n/a",
    }

    if str(er or "").strip().upper() != EXPECTED_ER:
        write_summary("❌ ZeroUI governance check failed.", "Unexpected mapped ER", summary_fields)
        print(f"::error title=ZeroUI governance check failed::mapped ER must be {EXPECTED_ER}, got {er}.")
        raise SystemExit(1)

    if policy_source != "DB_POLICY_SOURCE":
        write_summary("❌ ZeroUI governance check failed.", "Invalid policy source", summary_fields)
        print("::error title=ZeroUI governance check failed::policy_source is not DB_POLICY_SOURCE.")
        raise SystemExit(1)

    if not receipt_present:
        write_summary("❌ ZeroUI governance check failed.", "Receipt was not written", summary_fields)
        print("::error title=ZeroUI governance check failed::Receipt was not written.")
        raise SystemExit(1)

    if decision in {"hard_block", "soft_block", "action_required"}:
        write_summary("❌ ZeroUI governance blocked this pipeline.", "Pipeline blocked by ZeroUI", summary_fields)
        print(f"::error title=ZeroUI governance blocked::ZeroUI returned {decision} for {er}. Active blockers: {blocker_count}.")
        print("ZeroUI governance blocked this pipeline.")
        raise SystemExit(1)

    cli_exit_raw = pick(response, "cli_exit_code", "projections.ci_response.cli_exit_code")
    try:
        cli_exit_code = int(cli_exit_raw) if cli_exit_raw is not None else None
    except (TypeError, ValueError):
        cli_exit_code = None

    if decision == "pass":
        if cli_exit_code not in (None, 0):
            write_summary("❌ ZeroUI governance check failed.", "CLI exit code indicates block", summary_fields)
            print("::error title=ZeroUI governance check failed::FM-1 cli_exit_code requires pipeline failure.")
            raise SystemExit(1)
        write_summary("✅ ZeroUI governance check passed.", "Pipeline unblocked by ZeroUI", summary_fields)
        print(f"::notice title=ZeroUI governance check passed::ZeroUI returned pass for {er}. Receipt written. Active blockers: {blocker_count}.")
        print("ZeroUI governance check passed. Pipeline may continue.")
        return

    if cli_exit_code not in (None, 0):
        write_summary("❌ ZeroUI governance blocked this pipeline.", "CLI exit code indicates block", summary_fields)
        print("::error title=ZeroUI governance blocked::FM-1 cli_exit_code requires pipeline failure.")
        raise SystemExit(1)

    write_summary("❌ ZeroUI governance check failed.", "Unknown decision", summary_fields)
    print("::error title=ZeroUI governance check failed::Unknown or missing decision_outcome.")
    raise SystemExit(1)


def main() -> None:
    evidence_file_raw = os.getenv("ZEROUI_EVIDENCE_FILE") or ""
    evidence_format = (os.getenv("ZEROUI_EVIDENCE_FORMAT") or "").strip().lower()

    if not evidence_file_raw:
        raise SystemExit("evidence_file input is required.")
    if not evidence_format:
        raise SystemExit("evidence_format input is required.")

    evidence_file = Path(evidence_file_raw)
    if not evidence_file.exists():
        raise SystemExit(f"Evidence file not found: {evidence_file}")

    event_type_id, context = context_from_evidence(evidence_file, evidence_format)
    print("ZeroUI evidence context summary:")
    print(json.dumps({
        "event_type_id": event_type_id,
        "workflow_failed": context.get("workflow_failed"),
        "workflow_conclusion": context.get("workflow_conclusion"),
        "target_job_name": context.get("target_job_name"),
        "target_job_conclusion": context.get("target_job_conclusion"),
        "scan_passed": context.get("scan_passed"),
        "reason": context.get("reason"),
    }, indent=2, sort_keys=True))

    payload = build_payload(event_type_id, context)
    forbidden = ("tenant_id", "release_id", "expected_er", "expected_decision", "should_block")
    for key in forbidden:
        if key in payload:
            raise SystemExit(f"Forbidden authoritative field in signal payload: {key}")

    print("ZeroUI FM-1 governance gate prepared signal.")
    print(f"event_type_id={event_type_id}")
    print(f"repository={payload.get('repository')}")
    print(f"branch={payload.get('branch')}")
    print(f"workflow_run_id={payload.get('workflow_run_id')}")

    response = post_signal(payload)
    enforce_response(response)


if __name__ == "__main__":
    main()
