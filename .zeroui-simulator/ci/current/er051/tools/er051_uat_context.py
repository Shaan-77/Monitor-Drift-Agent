"""Authenticated ER-051 UAT scenario evaluation context (simulator → governance → FM-1)."""

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
    "uat_operational_fault",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def resolve_uat_hmac_secret() -> str:
    for key in (
        "FM1_ER051_UAT_HMAC_SECRET",
        "FM1_SIGNAL_INTAKE_AUTH_TOKEN",
        "ZEROUI_FM1_TOKEN",
        "FM1_INTAKE_TOKEN",
    ):
        secret = _text(os.getenv(key))
        if secret:
            return secret
    return ""


def canonical_uat_signing_payload(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: fields.get(key)
        for key in _SIGNED_FIELDS
        if fields.get(key) is not None and _text(fields.get(key))
    }


def sign_er051_uat_fields(fields: Mapping[str, Any], *, secret: str) -> str:
    payload = canonical_uat_signing_payload(fields)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()


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
    secret = _text(hmac_secret) or resolve_uat_hmac_secret()
    if not secret:
        raise ValueError("ER051_UAT_HMAC_SECRET_MISSING")

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
        "recipe_id": _text(recipe_id) or ER051_RECIPE_ID,
    }
    fault = _text(uat_operational_fault)
    if fault:
        fields["uat_operational_fault"] = fault

    signed = dict(fields)
    signed["signature"] = sign_er051_uat_fields(fields, secret=secret)
    signed["signed_at"] = datetime.now(timezone.utc).isoformat()
    signed["validated"] = False
    return signed


def attach_er051_uat_evaluation_to_envelope(
    envelope: dict[str, Any],
    scenario: Mapping[str, Any],
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Stamp signed UAT evaluation context on the FM-1 signal envelope payload."""
    out = dict(envelope)
    scenario_id = _text(scenario.get("scenario_id") or scenario.get("scenario"))
    if not scenario_id:
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
        evaluation = build_er051_uat_scenario_evaluation(
            scenario_id=scenario_id,
            simulator_run_id=simulator_run_id,
            trace_id=trace_id,
            workflow_run_id=workflow_run_id,
            commit_sha=commit_sha,
            repository=repository,
            tenant_id=resolved_tenant,
            fixture_source=fixture_source,
            governance_path=governance_path,
            recipe_id=_text(scenario.get("recipe_id")) or ER051_RECIPE_ID,
            uat_operational_fault=uat_fault or None,
        )
    except ValueError:
        return out

    payload = out.get("payload") if isinstance(out.get("payload"), dict) else {}
    payload = dict(payload)
    payload["er051_uat_scenario_evaluation"] = evaluation
    out["payload"] = payload
    return out


def uat_mechanism_preflight_status() -> dict[str, Any]:
    secret = resolve_uat_hmac_secret()
    if not secret:
        return {
            "available": False,
            "reason_code": "ER051_UAT_HMAC_SECRET_MISSING",
            "message": "FM-1 intake token / UAT HMAC secret is required for signed UAT scenario context.",
        }
    return {
        "available": True,
        "reason_code": "READY",
        "message": "Signed UAT scenario evaluation context is available.",
    }
