from __future__ import annotations

import hashlib
import http.client
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


SCENARIO_PATH = Path(".zeroui-simulator/ci/current/scenario.json")


def validate_er051_runtime_binding(evidence_file: Path) -> Dict[str, str]:
    commit_sha = str(os.getenv("GITHUB_SHA") or "").strip()
    if not commit_sha:
        raise SystemExit(
            "ER051_RUNTIME_CONTEXT_BINDING_FAILED: missing GITHUB_SHA commit binding before FM-1 submission."
        )
    artifact_digest = f"sha256:{hashlib.sha256(evidence_file.read_bytes()).hexdigest()}"
    return {"commitSha": commit_sha, "artifactDigest": artifact_digest}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def coverage_context(evidence: Any) -> Dict[str, Any]:
    data = evidence if isinstance(evidence, dict) else {}

    observed_raw = data.get("observed_value")
    if observed_raw is None:
        observed_raw = data.get("coverage_percent")
    if observed_raw is None:
        observed_raw = data.get("current_coverage_percent")

    threshold_raw = data.get("threshold")
    if threshold_raw is None:
        threshold_raw = data.get("required_coverage_percent")

    baseline_raw = data.get("baseline")
    if baseline_raw is None:
        baseline_raw = data.get("baseline_coverage_percent")

    if observed_raw is None:
        raise SystemExit("Missing required evidence field: observed_value")
    if threshold_raw is None:
        raise SystemExit("Missing required evidence field: threshold")
    if baseline_raw is None:
        raise SystemExit("Missing required evidence field: baseline")

    try:
        observed_value = float(observed_raw)
        threshold = float(threshold_raw)
        baseline = float(baseline_raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Coverage evidence numeric fields invalid: {exc}") from exc

    metric_type = str(data.get("metric_type") or "coverage")
    coverage_percent = float(data.get("coverage_percent", observed_value))
    current_coverage_percent = float(data.get("current_coverage_percent", observed_value))
    required_coverage_percent = float(data.get("required_coverage_percent", threshold))
    baseline_coverage_percent = float(data.get("baseline_coverage_percent", baseline))
    coverage_drop_percent = float(data.get("coverage_drop_percent", max(0.0, baseline - observed_value)))

    if "quality_gate_release_critical" not in data:
        raise SystemExit("Missing required evidence field: quality_gate_release_critical")

    return {
        "metric_type": metric_type,
        "observed_value": observed_value,
        "threshold": threshold,
        "baseline": baseline,
        "coverage_percent": coverage_percent,
        "current_coverage_percent": current_coverage_percent,
        "required_coverage_percent": required_coverage_percent,
        "baseline_coverage_percent": baseline_coverage_percent,
        "coverage_drop_percent": coverage_drop_percent,
        "coverage_dropped": bool(data.get("coverage_dropped", True)),
        "quality_gate_release_critical": bool(data.get("quality_gate_release_critical")),
        "coverage_tool": data.get("coverage_tool", "coverage.py"),
        "coverage_report_format": data.get("coverage_report_format", "coverage-json"),
        "coverage_scope": data.get("coverage_scope", "python_unit_tests"),
        "quality_gate": data.get("quality_gate", "required_coverage_gate"),
        "covered_file": data.get("covered_file", "zeroui_uat/er051_discount_calculator.py"),
        "scan_passed": bool(data.get("scan_passed", False)),
        "tests_passed": bool(data.get("tests_passed", True)),
        "test_failure_count": int(data.get("test_failure_count", 0)),
        "reason": str(data.get("reason") or "coverage_below_required_threshold"),
    }


def policy_gate_context(evidence: Any) -> Dict[str, Any]:
    data = evidence if isinstance(evidence, dict) else {}
    required_keys = [
        "policy_gate_family",
        "check_family",
        "contract_check_failed",
        "policy_gate_id",
        "policy_scope",
        "policy_engine",
        "required_gate",
        "validation_passed",
        "scan_passed",
        "reason",
    ]
    for key in required_keys:
        if key not in data:
            raise SystemExit(f"Missing required evidence field: {key}")

    violations = data.get("violations", [])
    if not isinstance(violations, list):
        violations = []

    violation_count = data.get("violation_count", len(violations))
    try:
        violation_count = int(violation_count)
    except (TypeError, ValueError):
        violation_count = len(violations)

    return {
        "policy_gate_family": data["policy_gate_family"],
        "check_family": data["check_family"],
        "contract_check_failed": data["contract_check_failed"],
        "policy_gate_id": data["policy_gate_id"],
        "policy_scope": data["policy_scope"],
        "policy_engine": data["policy_engine"],
        "required_gate": data["required_gate"],
        "validation_passed": data["validation_passed"],
        "scan_passed": data["scan_passed"],
        "policy_gate_passed": bool(data.get("policy_gate_passed", data["validation_passed"])),
        "contract_check_passed": bool(data.get("contract_check_passed", data["validation_passed"])),
        "reason": data["reason"],
        "violation_count": violation_count,
        "violations": violations,
    }


def db_migration_context(evidence: Any) -> Dict[str, Any]:
    data = evidence if isinstance(evidence, dict) else {}
    findings = data.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    destructive_count = data.get("destructive_operation_count", len(findings))
    try:
        destructive_count = int(destructive_count)
    except (TypeError, ValueError):
        destructive_count = len(findings)
    validation_passed = bool(data.get("validation_passed", destructive_count == 0))
    scan_passed = bool(data.get("scan_passed", validation_passed and destructive_count == 0))
    return {
        "migration_checker": data.get("migration_checker", "zeroui-db-migration-guard"),
        "migration_scope": data.get("migration_scope", "app_schema"),
        "migration_files": data.get("migration_files", []),
        "destructive_operation_count": destructive_count,
        "destructive_operation_approved": False,
        "approval_evidence_present": False,
        "backup_plan_present": False,
        "rollback_plan_present": False,
        "application_dependency_check_passed": False,
        "migration_window_approved": False,
        "validation_passed": validation_passed,
        "scan_passed": scan_passed,
        "reason": data.get(
            "reason",
            "db_migration_check_passed" if scan_passed else "unsafe_db_migration_detected_without_required_approval_evidence",
        ),
        "findings": findings,
    }


def bandit_context(evidence: Any) -> Dict[str, Any]:
    results = evidence.get("results", []) if isinstance(evidence, dict) else []
    high_or_critical = [
        item for item in results
        if str(item.get("issue_severity", "")).upper() in {"HIGH", "CRITICAL"}
    ]
    medium = [
        item for item in results
        if str(item.get("issue_severity", "")).upper() == "MEDIUM"
    ]
    findings = []
    for item in results:
        findings.append({
            "file": item.get("filename"),
            "line": item.get("line_number"),
            "test_id": item.get("test_id"),
            "test_name": item.get("test_name"),
            "severity": item.get("issue_severity"),
            "confidence": item.get("issue_confidence"),
            "message": item.get("issue_text"),
        })
    return {
        "scan_type": "sast",
        "scanner": "bandit",
        "severity": "critical" if high_or_critical else ("medium" if medium else "none"),
        "finding_count": len(results),
        "high_or_critical_count": len(high_or_critical),
        "medium_count": len(medium),
        "scan_passed": len(results) == 0,
        "findings": findings,
    }


SUPPORTED_EVENT_BY_FORMAT = {
    "bandit": "ci.security_scan.failed",
    "db-migration-evidence": "ci.db_migration.check_failed",
    "policy-gate-evidence": "ci.policy_gate.failed",
    "coverage-evidence": "ci.coverage.dropped",
}


def context_from_evidence(evidence_file: Path, evidence_format: str) -> Tuple[str, Dict[str, Any]]:
    fmt = evidence_format.strip().lower()
    if fmt == "bandit":
        return "ci.security_scan.failed", bandit_context(read_json(evidence_file))
    if fmt == "db-migration-evidence":
        return "ci.db_migration.check_failed", db_migration_context(read_json(evidence_file))
    if fmt == "policy-gate-evidence":
        return "ci.policy_gate.failed", policy_gate_context(read_json(evidence_file))
    if fmt == "coverage-evidence":
        return "ci.coverage.dropped", coverage_context(read_json(evidence_file))
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


def base_release_branch(scenario: Dict[str, Any], head_branch: str) -> str:
    base = str(scenario.get("base_branch") or "").strip()
    if base:
        return base
    return head_branch


def build_payload(event_type_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    scenario = load_scenario_metadata()
    run_id = os.getenv("GITHUB_RUN_ID") or "unknown-run"
    run_attempt = os.getenv("GITHUB_RUN_ATTEMPT") or "1"
    repository = os.getenv("GITHUB_REPOSITORY") or str(scenario.get("repository") or "unknown-repository")
    head_branch = release_branch(scenario)
    base_branch = base_release_branch(scenario, head_branch)
    workflow_name = os.getenv("GITHUB_WORKFLOW") or str(scenario.get("workflow_name") or "unknown-workflow")
    job_id = os.getenv("ZEROUI_JOB_ID") or os.getenv("GITHUB_JOB") or "zeroui-governance-gate"
    trace_id = str(scenario.get("trace_id") or f"trace-gha-{event_type_id}-{run_id}-{run_attempt}")
    simulator_run_id = str(scenario.get("simulator_run_id") or "").strip()
    commit_hash = os.getenv("GITHUB_SHA") or "unknown-commit"
    source_event_id = f"gha-{event_type_id}-{run_id}-{job_id}-{run_attempt}"
    source_event_revision = (
        f"gha-rev-{repository}-{workflow_name}-{commit_hash}-{head_branch}-"
        f"{simulator_run_id or 'no-simulator-run'}"
    )
    release_key = str(scenario.get("release_key") or "").strip()

    envelope: Dict[str, Any] = {
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
        "branch": head_branch,
        "commit_hash": commit_hash,
        "commit_sha": commit_hash,
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
            "head_branch": head_branch,
            "base_branch": base_branch,
            "release_branch": head_branch,
            "repository": repository,
            "simulator_run_id": simulator_run_id or None,
            "recipe_id": scenario.get("recipe_id"),
        },
    }
    if release_key:
        envelope["release_key"] = release_key
        envelope["raw_metadata"]["release_key"] = release_key
    return envelope


def attach_recheck_correlation(envelope: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Bind CI re-check to original ER-051 blocker; never invent unrelated correlation."""
    recheck = scenario.get("recheck") if isinstance(scenario.get("recheck"), dict) else {}
    if not recheck:
        return envelope
    out = dict(envelope)
    nev = str(recheck.get("recheck_of_normalized_event_id") or "").strip()
    blocker_id = str(recheck.get("recheck_of_blocker_id") or "").strip()
    if nev:
        out["recheck_of_normalized_event_id"] = nev
    if blocker_id:
        out["recheck_of_blocker_id"] = blocker_id
    out["recheck_reason"] = str(recheck.get("recheck_reason") or "er051_coverage_remediation_recheck")
    out["ci_recheck"] = True
    risk_key = str(recheck.get("risk_equivalence_key") or "").strip()
    if risk_key:
        out["risk_equivalence_key"] = risk_key
    return out


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
        try:
            err_body = json.loads(body)
            reason_code = err_body.get("reason_code")
            message = err_body.get("message")
            if reason_code:
                print(f"reason_code={reason_code}")
            if message:
                print(f"fm1_error_message={message}")
        except json.JSONDecodeError:
            pass
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

    if decision == "warn":
        warn_raw = pick(
            response,
            "warn_continuation_allowed",
            "projections.ci_response.warn_continuation_allowed",
        )
        blocked_state = pick(
            response,
            "blocked_state",
            "projections.ci_response.blocked_state",
        )
        warn_allowed = warn_raw in {True, "true", "True", 1, "1"}
        if warn_raw in {False, "false", "False", 0, "0"}:
            warn_allowed = False
        not_blocked = str(blocked_state or "not_blocked").strip().lower() in {
            "",
            "not_blocked",
            "none",
            "null",
        }
        if warn_allowed and not_blocked and blocker_count == 0:
            if cli_exit_code not in (None, 0):
                write_summary("❌ ZeroUI governance check failed.", "CLI exit code indicates block", summary_fields)
                print("::error title=ZeroUI governance check failed::FM-1 cli_exit_code requires pipeline failure.")
                raise SystemExit(1)
            write_summary(
                "⚠️ ZeroUI governance issued a warning.",
                "Pipeline may continue with warning",
                summary_fields,
            )
            print(
                f"::warning title=ZeroUI governance warning::ZeroUI returned warn for {er}. "
                f"Receipt written. Active blockers: {blocker_count}."
            )
            print("ZeroUI governance warning recorded. Pipeline may continue.")
            return
        write_summary("❌ ZeroUI governance blocked this pipeline.", "Warning requires resolution", summary_fields)
        print(
            f"::error title=ZeroUI governance blocked::ZeroUI warn for {er} is not continuation-safe. "
            f"Active blockers: {blocker_count}."
        )
        raise SystemExit(1)

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

    runtime_binding = validate_er051_runtime_binding(evidence_file)
    print("ZeroUI ER-051 runtime binding:")
    print(json.dumps(runtime_binding, indent=2, sort_keys=True))

    event_type_id, context = context_from_evidence(evidence_file, evidence_format)
    print("ZeroUI evidence context summary:")
    print(json.dumps({
        "event_type_id": event_type_id,
        "metric_type": context.get("metric_type"),
        "observed_value": context.get("observed_value"),
        "threshold": context.get("threshold"),
        "baseline": context.get("baseline"),
        "coverage_dropped": context.get("coverage_dropped"),
        "scan_passed": context.get("scan_passed"),
    }, indent=2, sort_keys=True))

    payload = build_payload(event_type_id, context)
    payload = attach_recheck_correlation(payload, load_scenario_metadata())
    scenario = load_scenario_metadata()
    try:
        tools_dir = Path(__file__).resolve().parents[1] / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        from er051_uat_context import attach_trusted_ci_cd_uat_context_for_governance

        payload = attach_trusted_ci_cd_uat_context_for_governance(payload, scenario)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        if "CI_CD_UAT_CONTEXT_ATTACH_FAILED" not in message:
            message = f"CI_CD_UAT_CONTEXT_ATTACH_FAILED:{message}"
        raise SystemExit(message) from exc
    forbidden = ("tenant_id", "release_id", "expected_er", "expected_decision", "should_block")
    for key in forbidden:
        if key in payload:
            raise SystemExit(f"Forbidden authoritative field in signal payload: {key}")

    print("ZeroUI FM-1 governance gate prepared signal.")
    print(f"event_type_id={event_type_id}")
    print(f"repository={payload.get('repository')}")
    print(f"branch={payload.get('branch')}")
    print(f"workflow_run_id={payload.get('workflow_run_id')}")
    if payload.get("recheck_of_normalized_event_id") or payload.get("recheck_of_blocker_id"):
        print(
            "recheck_of_normalized_event_id="
            f"{payload.get('recheck_of_normalized_event_id')}"
        )
        print(f"recheck_of_blocker_id={payload.get('recheck_of_blocker_id')}")

    response = post_signal(payload)
    enforce_response(response)


if __name__ == "__main__":
    main()
