from __future__ import annotations

import http.client
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


SCENARIO_PATH = Path(".zeroui-simulator/ci/current/er057/scenario.json")
EXPECTED_ER = "ER-057"
TARGET_WORKFLOW_PATH = ".github/workflows/zeroui-fm1-simulator-er057-promotion-gate.yml"


def build_github_actions_source_event_id(event_type_id, workflow_run_id, github_job_key, run_attempt):
    return f"gha-{event_type_id}-{workflow_run_id}-{github_job_key}-{run_attempt}"


def build_github_actions_source_event_revision(repository, workflow_ref, commit_sha, branch, simulator_run_id=""):
    return f"gha-rev-{repository}-{workflow_ref}-{commit_sha}-{branch}-{simulator_run_id or 'no-simulator-run'}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def promotion_rejection_evidence_context(evidence: Any) -> Dict[str, Any]:
    data = evidence if isinstance(evidence, dict) else {}
    required_keys = [
        "promotion_status",
        "target_environment",
        "rejection_reason",
        "review_state",
        "review_api_verified",
        "review_http_status",
        "promotion_rejected",
        "promotion_approved",
        "workflow_completed",
        "workflow_conclusion",
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

    if str(data["promotion_status"]).strip().lower() != "rejected":
        raise SystemExit("promotion_status must be rejected")
    if not str(data["target_environment"]).strip():
        raise SystemExit("target_environment is required")
    if not str(data["rejection_reason"]).strip():
        raise SystemExit("rejection_reason is required")
    if str(data["review_state"]).strip().lower() != "rejected":
        raise SystemExit("review_state must be rejected")
    if data.get("review_api_verified") is not True:
        raise SystemExit("review_api_verified must be true")
    try:
        review_http_status = int(data["review_http_status"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"review_http_status must be an integer: {exc}") from exc
    if review_http_status != 200:
        raise SystemExit("review_http_status must be 200")
    if not isinstance(data["promotion_rejected"], bool) or data["promotion_rejected"] is not True:
        raise SystemExit("promotion_rejected must be true")
    if not isinstance(data["promotion_approved"], bool) or data["promotion_approved"] is not False:
        raise SystemExit("promotion_approved must be false")
    if str(data["target_workflow_path"]) != TARGET_WORKFLOW_PATH:
        raise SystemExit("target_workflow_path must match the managed ER-057 promotion gate")
    conclusion = str(data["workflow_conclusion"]).strip().lower()
    if conclusion not in {"failure", "failed"}:
        raise SystemExit("workflow_conclusion must be failure for rejected promotion evidence")

    try:
        target_run_id = int(data["target_workflow_run_id"])
        target_run_attempt = int(data["target_workflow_run_attempt"])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"target workflow run identity must be numeric: {exc}") from exc

    context: Dict[str, Any] = {
        "promotion_status": "rejected",
        "target_environment": str(data["target_environment"]),
        "promotion_family": str(data.get("promotion_family") or "release_promotion"),
        "promotion_gate": str(data.get("promotion_gate") or "github_environment_review"),
        "promotion_rejected": True,
        "promotion_approved": False,
        "rejection_reason": str(data["rejection_reason"]),
        "review_state": "rejected",
        "review_api_verified": True,
        "review_http_status": 200,
        "workflow_completed": bool(data["workflow_completed"]),
        "workflow_conclusion": conclusion,
        "scan_passed": bool(data["scan_passed"]),
        "target_workflow_name": str(data.get("target_workflow_name") or ""),
        "target_workflow_path": str(data["target_workflow_path"]),
        "target_workflow_run_id": target_run_id,
        "target_workflow_run_attempt": target_run_attempt,
        "target_workflow_html_url": str(data.get("target_workflow_html_url") or ""),
        "target_branch": str(data["target_branch"]),
        "target_commit_sha": str(data["target_commit_sha"]),
        "target_repository": str(data["target_repository"]),
        "reason": str(data["reason"]),
    }
    if data.get("promotion_decision_id"):
        context["promotion_decision_id"] = str(data["promotion_decision_id"])
    if data.get("environment_id") is not None:
        try:
            context["environment_id"] = int(data["environment_id"])
        except (TypeError, ValueError):
            pass
    if data.get("requested_by"):
        context["requested_by"] = str(data["requested_by"])
    if data.get("reviewed_by"):
        context["reviewed_by"] = str(data["reviewed_by"])
    if data.get("review_comment"):
        context["review_comment"] = str(data["review_comment"])
    return context


def context_from_evidence(evidence_file: Path, evidence_format: str) -> Tuple[str, Dict[str, Any]]:
    fmt = evidence_format.strip().lower()
    if fmt != "github-promotion-rejection":
        raise SystemExit(f"Unsupported evidence_format: {evidence_format}")
    return "ci.promotion.rejected", promotion_rejection_evidence_context(read_json(evidence_file))


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
    commit_hash = os.getenv("GITHUB_SHA") or "unknown-commit"
    source_event_id = build_github_actions_source_event_id(event_type_id, run_id, job_id, run_attempt)
    source_event_revision = build_github_actions_source_event_revision(
        repository,
        workflow_name,
        commit_hash,
        branch,
        simulator_run_id,
    )
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "source_event_id": source_event_id,
        "source_event_revision": source_event_revision,
        "provider": "github_actions",
        "platform": "GitHub Actions",
        "source_system": "ci_cd",
        "signal_method": "pipeline_job",
        "event_type_id": event_type_id,
        "repository": repository,
        "branch": branch,
        "commit_hash": commit_hash,
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
