from __future__ import annotations

import http.client
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


SCENARIO_PATH = Path(".zeroui-simulator/ci/current/er056/scenario.json")
EXPECTED_ER = "ER-056"
TARGET_WORKFLOW_PATH = ".github/workflows/zeroui-fm1-simulator-er056-required-gate.yml"


def build_github_actions_source_event_id(event_type_id, workflow_run_id, github_job_key, run_attempt):
    return f"gha-{event_type_id}-{workflow_run_id}-{github_job_key}-{run_attempt}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def github_workflow_run_context(evidence: Any) -> Dict[str, Any]:
    data = evidence if isinstance(evidence, dict) else {}
    required_keys = [
        "required_gate",
        "workflow_family",
        "stage",
        "workflow_cancelled",
        "workflow_conclusion",
        "workflow_completed",
        "scan_passed",
        "target_workflow_path",
        "target_workflow_run_id",
        "target_workflow_run_attempt",
        "target_branch",
        "target_commit_sha",
        "target_repository",
        "reason",
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
    if not isinstance(data["workflow_cancelled"], bool) or data["workflow_cancelled"] is not True:
        raise SystemExit("workflow_cancelled must be true")
    if str(data["workflow_conclusion"]).strip().lower() != "cancelled":
        raise SystemExit("workflow_conclusion must be cancelled")
    if data.get("workflow_completed") is True:
        raise SystemExit("workflow_completed must be false for cancelled required workflow evidence")
    if str(data["target_workflow_path"]) != TARGET_WORKFLOW_PATH:
        raise SystemExit("target_workflow_path must match the managed ER-056 required gate")

    try:
        target_run_id = int(data["target_workflow_run_id"])
        target_run_attempt = int(data["target_workflow_run_attempt"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"target workflow run identity must be numeric: {exc}") from exc

    return {
        "required_gate": True,
        "workflow_family": "release_as_code",
        "stage": "release",
        "workflow_completed": False,
        "workflow_cancelled": True,
        "workflow_conclusion": "cancelled",
        "scan_passed": False,
        "target_workflow_name": str(data.get("target_workflow_name") or ""),
        "target_workflow_path": str(data["target_workflow_path"]),
        "target_workflow_run_id": target_run_id,
        "target_workflow_run_attempt": target_run_attempt,
        "target_workflow_html_url": str(data.get("target_workflow_html_url") or ""),
        "target_branch": str(data["target_branch"]),
        "target_commit_sha": str(data["target_commit_sha"]),
        "target_repository": str(data["target_repository"]),
        "cancellation_source": str(data.get("cancellation_source") or "simulator_github_provider"),
        "cancellation_verified_from": str(data.get("cancellation_verified_from") or "github_actions_api"),
        "reason": str(data["reason"]),
    }


def context_from_evidence(evidence_file: Path, evidence_format: str) -> Tuple[str, Dict[str, Any]]:
    fmt = evidence_format.strip().lower()
    if fmt != "github-workflow-run":
        raise SystemExit(f"Unsupported evidence_format: {evidence_format}")
    return "ci.workflow.cancelled", github_workflow_run_context(read_json(evidence_file))


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
    source_event_id = build_github_actions_source_event_id(event_type_id, run_id, job_id, run_attempt)
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
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    base_path = parsed.path.rstrip("/")
    signal_path = f"{base_path}/fm1/v1/ci/signals" if base_path else "/fm1/v1/ci/signals"
    return connection_class(parsed.hostname, port=parsed.port, timeout=60), signal_path


def post_signal(payload: Dict[str, Any]) -> Dict[str, Any]:
    base_url = (os.getenv("FM1_BASE_URL") or "").rstrip("/")
    token = os.getenv("FM1_SIGNAL_INTAKE_AUTH_TOKEN") or ""
    if not base_url:
        raise SystemExit("FM1_BASE_URL is missing.")
    if not token:
        raise SystemExit("FM1_SIGNAL_INTAKE_AUTH_TOKEN is missing.")
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
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
    parsed = json.loads(body)
    print(f"normalized_event_id={parsed.get('normalized_event_id', 'n/a')}")
    print(f"mapped_trigger_id={parsed.get('mapped_trigger_id', parsed.get('trigger_id', 'n/a'))}")
    print(f"decision_outcome={parsed.get('decision_outcome', 'n/a')}")
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


def enforce_response(response: Dict[str, Any]) -> None:
    er = pick(response, "mapped_trigger_id", "trigger_id", "projections.ci_response.mapped_trigger_id")
    decision = pick(response, "decision_outcome", "decision.decision_outcome", "projections.ci_response.decision_outcome")
    policy_source = pick(response, "policy_source", "projections.ci_response.policy_source")
    receipt_status = pick(response, "receipt_status", "intake_receipt_status", "decision_receipt_status")
    receipt_ref = pick(response, "receipt_ref", "projections.ci_response.receipt_ref", "decision.receipt_ref")
    receipt_present = receipt_status == "receipt_written" or bool(receipt_ref)
    if str(er or "").strip().upper() != EXPECTED_ER:
        raise SystemExit(1)
    if policy_source != "DB_POLICY_SOURCE":
        raise SystemExit(1)
    if not receipt_present:
        raise SystemExit(1)
    if decision in {"hard_block", "soft_block", "action_required"}:
        print(f"::error title=ZeroUI governance blocked::ZeroUI returned {decision} for {er}.")
        raise SystemExit(1)
    if decision == "pass":
        return
    raise SystemExit(1)


def main() -> None:
    evidence_file = Path(os.getenv("ZEROUI_EVIDENCE_FILE") or "")
    evidence_format = (os.getenv("ZEROUI_EVIDENCE_FORMAT") or "").strip().lower()
    if not evidence_file.exists():
        raise SystemExit(f"Evidence file not found: {evidence_file}")
    event_type_id, context = context_from_evidence(evidence_file, evidence_format)
    payload = build_payload(event_type_id, context)
    response = post_signal(payload)
    enforce_response(response)


if __name__ == "__main__":
    main()
