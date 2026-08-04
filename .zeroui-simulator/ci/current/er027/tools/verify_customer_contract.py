#!/usr/bin/env python3
"""Verify provider/consumer contract compatibility for ER-027 simulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROVIDER_PATH = Path(
    ".zeroui-simulator/ci/current/er027/contracts/provider/customer-service/openapi.json"
)
CONSUMER_PATH = Path(
    ".zeroui-simulator/ci/current/er027/contracts/consumer/order-service/customer-service.consumer-contract.json"
)
EVIDENCE_PATH = Path("contract-verification-evidence.json")

ROUTING_FIELDS: dict[str, Any] = {
    "event_type_id": "ci.policy_gate.failed",
    "policy_gate_family": "contract",
    "check_family": "contract",
    "contract_check_failed": True,
    "policy_gate_id": "contract_verification",
    "policy_scope": "provider_consumer_contracts",
    "policy_engine": "zeroui-contract-schema-verifier",
    "required_gate": True,
}


def load_json(path: Path) -> Any:
    if not path.is_file():
        print(f"Missing contract file: {path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Malformed contract file {path}: {exc}", file=sys.stderr)
        raise SystemExit(2)


def extract_provider_customer_schema(provider: Any) -> dict[str, Any]:
    if not isinstance(provider, dict):
        print("Provider contract must be a JSON object.", file=sys.stderr)
        raise SystemExit(2)
    try:
        schema = (
            provider["paths"]["/customers/{id}"]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
        )
    except (KeyError, TypeError) as exc:
        print(f"Provider contract missing expected customer schema path: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(schema, dict):
        print("Provider customer schema must be an object.", file=sys.stderr)
        raise SystemExit(2)
    return schema


def extract_consumer_contract(consumer: Any) -> dict[str, Any]:
    if not isinstance(consumer, dict):
        print("Consumer contract must be a JSON object.", file=sys.stderr)
        raise SystemExit(2)
    required_fields = consumer.get("required_response_fields")
    if not isinstance(required_fields, list):
        print("Consumer contract required_response_fields must be an array.", file=sys.stderr)
        raise SystemExit(2)
    return consumer


def compare_contracts(provider_schema: dict[str, Any], consumer_contract: dict[str, Any]) -> list[dict[str, str]]:
    provider_properties = provider_schema.get("properties")
    consumer_required = consumer_contract.get("required_response_fields")
    if not isinstance(provider_properties, dict):
        print("Provider schema properties must be an object.", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(consumer_required, list):
        print("Consumer contract required_response_fields must be an array.", file=sys.stderr)
        raise SystemExit(2)

    violations: list[dict[str, str]] = []
    for field in consumer_required:
        field_name = str(field)
        if field_name not in provider_properties:
            violations.append(
                {
                    "rule": "required_consumer_field_missing_in_provider",
                    "field": field_name,
                }
            )
    return violations


def write_evidence(*, violations: list[dict[str, str]], consumer_contract: dict[str, Any]) -> dict[str, Any]:
    passed = len(violations) == 0
    evidence = {
        **ROUTING_FIELDS,
        "policy_gate_name": "Provider/consumer contract verification",
        "provider": str(consumer_contract.get("provider") or "customer-service"),
        "consumer": str(consumer_contract.get("consumer") or "order-service"),
        "contract_check_failed": not passed,
        "contract_verification_passed": passed,
        "schema_check_passed": passed,
        "breaking_change_count": len(violations),
        "violation_count": len(violations),
        "policy_violations": violations,
        "violations": violations,
        "validation_passed": passed,
        "scan_passed": passed,
        "policy_gate_passed": passed,
        "contract_check_passed": passed,
        "reason": "contract_verification_passed" if passed else "contract_verification_failed",
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ZeroUI ER-027 contract verification evidence:")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return evidence


def main() -> None:
    provider = load_json(PROVIDER_PATH)
    consumer = load_json(CONSUMER_PATH)
    provider_schema = extract_provider_customer_schema(provider)
    consumer_contract = extract_consumer_contract(consumer)
    violations = compare_contracts(provider_schema, consumer_contract)
    write_evidence(violations=violations, consumer_contract=consumer_contract)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
