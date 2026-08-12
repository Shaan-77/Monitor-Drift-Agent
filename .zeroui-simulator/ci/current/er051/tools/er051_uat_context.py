"""Simulator-side trusted CI/CD UAT scenario context signing (governance → FM-1)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping

EVALUATION_CONTEXT_UAT_SCENARIO = "uat_scenario"
FIXTURE_SOURCE_SYNTHETIC_SEED = "synthetic_seed"
GOVERNANCE_PATH_LIVE_FM1 = "live_fm1"
ER051_RECIPE_ID = "ci-009"
ER051_EVENT_TYPE_ID = "ci.coverage.dropped"
ER051_MAPPED_ER_ID = "ER-051"

_SIGNED_FIELDS = (
    "evaluation_context",
    "fixture_source",
    "governance_path",
    "scenario_id",
    "simulator_run_id",
    "trace_id",
    "workflow_run_id",
    "commit_sha",
    "repository",
    "tenant_id",
    "recipe_id",
    "mapped_er_id",
    "event_type_id",
    "uat_operational_fault",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def resolve_uat_hmac_secret() -> str:
    keys = (
        "FM1_CI_CD_UAT_HMAC_SECRET",
        "FM1_ER051_UAT_HMAC_SECRET",
        "TSIM_FM1_INTAKE_TOKEN",
        "FM1_SIGNAL_INTAKE_AUTH_TOKEN",
        "VITE_FM1_SIGNAL_INTAKE_AUTH_TOKEN",
        "ZEROUI_FM1_TOKEN",
        "FM1_INTAKE_TOKEN",
    )
    for key in keys:
        secret = _text(os.getenv(key))
        if secret:
            return secret
    try:
        from live_bootstrap import find_repository_root, load_root_env, resolve_env_value

        _, root_env = load_root_env(find_repository_root())
        for key in keys:
            secret = _text(resolve_env_value(key, root_env))
            if secret:
                return secret
    except Exception:
        return ""
    return ""


def canonical_uat_signing_payload(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: fields.get(key)
        for key in _SIGNED_FIELDS
        if fields.get(key) is not None and _text(fields.get(key))
    }


def sign_ci_cd_uat_fields(fields: Mapping[str, Any], *, secret: str) -> str:
    payload = canonical_uat_signing_payload(fields)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()


def build_ci_cd_uat_scenario_evaluation(
    *,
    scenario_id: str,
    simulator_run_id: str,
    trace_id: str,
    workflow_run_id: str,
    commit_sha: str,
    repository: str,
    tenant_id: str,
    mapped_er_id: str,
    event_type_id: str,
    recipe_id: str,
    fixture_source: str = FIXTURE_SOURCE_SYNTHETIC_SEED,
    governance_path: str = GOVERNANCE_PATH_LIVE_FM1,
    uat_operational_fault: str | None = None,
    hmac_secret: str | None = None,
) -> dict[str, Any]:
    secret = _text(hmac_secret) or resolve_uat_hmac_secret()
    if not secret:
        raise ValueError("CI_CD_UAT_HMAC_SECRET_MISSING")

    fields: dict[str, Any] = {
        "evaluation_context": EVALUATION_CONTEXT_UAT_SCENARIO,
        "fixture_source": fixture_source,
        "governance_path": governance_path,
        "scenario_id": _text(scenario_id).lower(),
        "simulator_run_id": _text(simulator_run_id),
        "trace_id": _text(trace_id),
        "workflow_run_id": _text(workflow_run_id),
        "commit_sha": _text(commit_sha),
        "repository": _text(repository),
        "tenant_id": _text(tenant_id),
        "recipe_id": _text(recipe_id),
        "mapped_er_id": _text(mapped_er_id).upper(),
        "event_type_id": _text(event_type_id),
    }
    fault = _text(uat_operational_fault)
    if fault:
        fields["uat_operational_fault"] = fault

    signed = dict(fields)
    signed["signature"] = sign_ci_cd_uat_fields(fields, secret=secret)
    signed["signed_at"] = datetime.now(timezone.utc).isoformat()
    signed["validated"] = False
    return signed


def attach_ci_cd_uat_evaluation_to_envelope(
    envelope: dict[str, Any],
    scenario: Mapping[str, Any],
    *,
    tenant_id: str | None = None,
    mapped_er_id: str | None = None,
    event_type_id: str | None = None,
    recipe_id: str | None = None,
) -> dict[str, Any]:
    """Stamp signed UAT evaluation context on the FM-1 signal envelope payload."""
    out = dict(envelope)
    scenario_id = _text(scenario.get("scenario_id") or scenario.get("scenario"))
    if not scenario_id:
        return out

    resolved_recipe = _text(recipe_id or scenario.get("recipe_id"))
    resolved_er = _text(mapped_er_id or scenario.get("mapped_er_id") or scenario.get("er_id"))
    if not resolved_er and resolved_recipe:
        try:
            from er_demo_seed_loader import recipe_to_er_id

            resolved_er = _text(recipe_to_er_id(scenario, recipe_id=resolved_recipe))
        except Exception:
            resolved_er = ""
    resolved_event = _text(
        event_type_id or scenario.get("event_type_id") or out.get("event_type_id")
    )
    if not resolved_er or not resolved_event or not resolved_recipe:
        return out

    fixture_source = _text(scenario.get("fixture_source")) or FIXTURE_SOURCE_SYNTHETIC_SEED
    governance_path = _text(scenario.get("governance_path")) or GOVERNANCE_PATH_LIVE_FM1
    simulator_run_id = _text(scenario.get("simulator_run_id"))
    trace_id = _text(out.get("trace_id") or scenario.get("trace_id"))
    workflow_run_id = _text(out.get("workflow_run_id") or os.getenv("GITHUB_RUN_ID"))
    commit_sha = _text(out.get("commit_hash") or os.getenv("GITHUB_SHA"))
    repository = _text(out.get("repository") or scenario.get("repository"))
    resolved_tenant = _text(tenant_id or scenario.get("tenant_id"))

    if not simulator_run_id or not trace_id or not workflow_run_id or not commit_sha:
        return out

    uat_fault = _text(scenario.get("uat_operational_fault"))
    try:
        evaluation = build_ci_cd_uat_scenario_evaluation(
            scenario_id=scenario_id,
            simulator_run_id=simulator_run_id,
            trace_id=trace_id,
            workflow_run_id=workflow_run_id,
            commit_sha=commit_sha,
            repository=repository,
            tenant_id=resolved_tenant,
            mapped_er_id=resolved_er,
            event_type_id=resolved_event,
            recipe_id=resolved_recipe,
            fixture_source=fixture_source,
            governance_path=governance_path,
            uat_operational_fault=uat_fault or None,
        )
    except ValueError:
        return out

    payload = out.get("payload") if isinstance(out.get("payload"), dict) else {}
    payload = dict(payload)
    payload["ci_cd_uat_scenario_evaluation"] = evaluation
    payload["er051_uat_scenario_evaluation"] = evaluation
    out["payload"] = payload
    return out


def requires_trusted_ci_cd_uat_scenario_context(scenario: Mapping[str, Any]) -> bool:
    """True when governance must attach signed synthetic-seed UAT evaluation."""
    scenario_id = _text(scenario.get("scenario_id") or scenario.get("scenario"))
    if not scenario_id:
        return False
    fixture = _text(scenario.get("fixture_source")) or FIXTURE_SOURCE_SYNTHETIC_SEED
    governance = _text(scenario.get("governance_path")) or GOVERNANCE_PATH_LIVE_FM1
    return fixture == FIXTURE_SOURCE_SYNTHETIC_SEED and governance == GOVERNANCE_PATH_LIVE_FM1


def _uat_evaluation_present(envelope: Mapping[str, Any]) -> bool:
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    evaluation = payload.get("ci_cd_uat_scenario_evaluation")
    return isinstance(evaluation, dict) and bool(_text(evaluation.get("signature")))


def diagnose_ci_cd_uat_attach_failure(
    envelope: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> str:
    if not _text(scenario.get("scenario_id") or scenario.get("scenario")):
        return "scenario_id_missing"
    if not _text(scenario.get("simulator_run_id")):
        return "simulator_run_id_missing"
    resolved_recipe = _text(scenario.get("recipe_id"))
    resolved_er = _text(scenario.get("mapped_er_id") or scenario.get("er_id"))
    resolved_event = _text(scenario.get("event_type_id"))
    if not resolved_er or not resolved_event or not resolved_recipe:
        return "scenario_identity_incomplete"
    envelope_copy = dict(envelope)
    if not _text(envelope_copy.get("trace_id") or scenario.get("trace_id")):
        return "trace_id_missing"
    if not _text(envelope_copy.get("workflow_run_id") or os.getenv("GITHUB_RUN_ID")):
        return "workflow_run_id_missing"
    if not _text(envelope_copy.get("commit_hash") or os.getenv("GITHUB_SHA")):
        return "commit_sha_missing"
    if not resolve_uat_hmac_secret():
        return "CI_CD_UAT_HMAC_SECRET_MISSING"
    return "CI_CD_UAT_CONTEXT_ATTACH_FAILED"


def attach_trusted_ci_cd_uat_context_for_governance(
    envelope: dict[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach signed UAT context; fail closed for explicit trusted scenario mode."""
    if not requires_trusted_ci_cd_uat_scenario_context(scenario):
        try:
            return attach_ci_cd_uat_evaluation_to_envelope(envelope, scenario)
        except Exception:
            return envelope
    try:
        out = attach_ci_cd_uat_evaluation_to_envelope(envelope, scenario)
    except Exception as exc:
        cause = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(f"CI_CD_UAT_CONTEXT_ATTACH_FAILED:{cause}") from exc
    if not _uat_evaluation_present(out):
        reason = diagnose_ci_cd_uat_attach_failure(out, scenario)
        raise RuntimeError(f"CI_CD_UAT_CONTEXT_ATTACH_FAILED:{reason}")
    return out


def uat_mechanism_preflight_status() -> dict[str, Any]:
    secret = resolve_uat_hmac_secret()
    if not secret:
        return {
            "available": False,
            "reason_code": "CI_CD_UAT_HMAC_SECRET_MISSING",
            "message": "FM-1 intake token / UAT HMAC secret is required for signed UAT scenario context.",
        }
    return {
        "available": True,
        "reason_code": "READY",
        "message": "Signed UAT scenario evaluation context is available.",
    }


def evaluate_ci_cd_uat_runtime_preflight(
    *,
    fm1_base_url: str | None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Verify FM-1 trusted UAT scenario runtime is enabled before scenario trigger."""
    if not _text(scenario_id):
        return {
            "preflight_status": "READY",
            "reason_code": "SCENARIO_NOT_SELECTED",
            "message": "No scenario selected; FM-1 UAT runtime gate not required.",
            "trigger_enabled": True,
        }

    mechanism = uat_mechanism_preflight_status()
    if not mechanism.get("available"):
        return {
            "preflight_status": "NOT_READY",
            "reason_code": mechanism.get("reason_code"),
            "message": mechanism.get("message"),
            "trigger_enabled": False,
        }

    base = _text(fm1_base_url).rstrip("/")
    if not base:
        return {
            "preflight_status": "NOT_READY",
            "reason_code": "FM1_BASE_URL_MISSING",
            "message": "FM-1 base URL is required to verify trusted UAT scenario runtime.",
            "trigger_enabled": False,
        }

    try:
        import httpx

        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{base}/fm1/v1/health/deep")
        if response.status_code != 200:
            return {
                "preflight_status": "NOT_READY",
                "reason_code": "FM1_HEALTH_UNAVAILABLE",
                "message": "FM-1 deep health probe failed; cannot verify UAT scenario runtime.",
                "trigger_enabled": False,
            }
        payload = response.json()
    except Exception as exc:
        return {
            "preflight_status": "NOT_READY",
            "reason_code": "FM1_HEALTH_UNAVAILABLE",
            "message": f"FM-1 deep health probe failed: {exc}",
            "trigger_enabled": False,
        }

    runtime = payload.get("ci_cd_uat_scenario_runtime")
    enabled = bool(isinstance(runtime, dict) and runtime.get("enabled"))
    if not enabled:
        return {
            "preflight_status": "NOT_READY",
            "reason_code": "CI_CD_UAT_SCENARIO_RUNTIME_DISABLED",
            "message": (
                "FM-1 trusted CI/CD UAT scenario mode is disabled. "
                "Start FM-1 with FM1_CI_CD_UAT_SCENARIO_ENABLED=1 for local/UAT."
            ),
            "trigger_enabled": False,
            "fm1_runtime": runtime,
        }

    return {
        "preflight_status": "READY",
        "reason_code": "READY",
        "message": "FM-1 trusted CI/CD UAT scenario runtime is enabled.",
        "trigger_enabled": True,
        "fm1_runtime": runtime,
    }


def build_er051_uat_scenario_evaluation(
    *,
    scenario_id: str,
    simulator_run_id: str,
    trace_id: str,
    workflow_run_id: str,
    commit_sha: str,
    repository: str,
    tenant_id: str,
    fixture_source: str = FIXTURE_SOURCE_SYNTHETIC_SEED,
    governance_path: str = GOVERNANCE_PATH_LIVE_FM1,
    recipe_id: str = ER051_RECIPE_ID,
    uat_operational_fault: str | None = None,
    hmac_secret: str | None = None,
) -> dict[str, Any]:
    return build_ci_cd_uat_scenario_evaluation(
        scenario_id=scenario_id,
        simulator_run_id=simulator_run_id,
        trace_id=trace_id,
        workflow_run_id=workflow_run_id,
        commit_sha=commit_sha,
        repository=repository,
        tenant_id=tenant_id,
        mapped_er_id=ER051_MAPPED_ER_ID,
        event_type_id=ER051_EVENT_TYPE_ID,
        recipe_id=recipe_id or ER051_RECIPE_ID,
        fixture_source=fixture_source,
        governance_path=governance_path,
        uat_operational_fault=uat_operational_fault,
        hmac_secret=hmac_secret,
    )


def sign_er051_uat_fields(fields: Mapping[str, Any], *, secret: str) -> str:
    return sign_ci_cd_uat_fields(fields, secret=secret)


def attach_er051_uat_evaluation_to_envelope(
    envelope: dict[str, Any],
    scenario: Mapping[str, Any],
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    return attach_ci_cd_uat_evaluation_to_envelope(
        envelope,
        scenario,
        tenant_id=tenant_id,
        mapped_er_id=ER051_MAPPED_ER_ID,
        event_type_id=ER051_EVENT_TYPE_ID,
        recipe_id=_text(scenario.get("recipe_id")) or ER051_RECIPE_ID,
    )
