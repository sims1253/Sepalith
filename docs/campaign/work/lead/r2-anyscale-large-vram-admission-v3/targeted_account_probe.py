#!/usr/bin/env python3
"""Read-only account probe for large-GPU admission evidence.

The GPU status endpoints are fleet snapshots. Their empty result is deliberately
reported as absence of active-fleet evidence, never as lack of entitlement or
future capacity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

TARGETS = ("g6e.2xlarge", "p4de.24xlarge", "p5.4xlarge")
ALLOWED_CALLS = (
    "list_clouds_GET",
    "get_cloud_GET",
    "credits_GET",
    "gpu_fleet_instance_types_GET",
    "gpu_fleet_status_GET_filtered",
)


def _date(value):
    return None if value is None else str(value)


def _error(exc):
    # Never serialize exception text or HTTP response bodies: either can include
    # URLs, request metadata, or credentials.
    return {"type": type(exc).__name__, "status": getattr(exc, "status", None)}


def collect(client):
    api = client._internal_api_client
    credit = api.get_credits_v2_api_v2_organization_billing_credits_v2_get(
        _request_timeout=20
    ).to_dict()
    windows = []
    for item in credit.get("in_use_credits") or []:
        windows.append(
            {
                "effective_date_start": _date(item.get("effective_date_start")),
                "effective_date_end": _date(item.get("effective_date_end")),
                "total_granted_usd": item.get("total_granted_usd"),
                "total_balance_usd": item.get("total_balance_usd"),
                "amount_consumed_usd": item.get("amount_consumed_usd"),
            }
        )
    result = {
        "schema": "sepalith.anyscale-large-vram-account-evidence.v3",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "allowed_calls": list(ALLOWED_CALLS),
        "credits": {
            "current_balance_usd": credit.get("current_balance_usd"),
            "amount_spent_usd": credit.get("amount_spent_usd"),
            "total_granted_usd": credit.get("total_granted_usd"),
            "active_windows": windows,
        },
        "clouds": [],
        "interpretation": {
            "fleet_snapshot_is_entitlement_catalog": False,
            "zero_target_nodes_means": "no target-shaped nodes in current fleet snapshot",
            "zero_target_nodes_does_not_mean": [
                "instance type is unavailable to this account",
                "future allocation would fail",
                "future capacity is zero",
            ],
        },
        "credentials_accessed_or_serialized": False,
        "mutations": 0,
    }
    for cloud in client.list_clouds(count=20).results:
        full = client.get_cloud(cloud_id=cloud.id)
        entry = {
            "name": cloud.name,
            "provider": str(cloud.provider),
            "region": cloud.region,
            "state": str(cloud.state),
            "status": str(cloud.status),
            "is_anyscale_hosted": bool(full.is_aioa),
            "target_fleet_snapshots": {},
        }
        try:
            snap = api.get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(
                cloud.id, _request_timeout=20
            ).to_dict()
            entry["fleet_snapshot_instance_types"] = (
                (snap.get("result") or {}).get("instance_types") or []
            )
        except Exception as exc:
            entry["fleet_snapshot_instance_types_error"] = _error(exc)
        for instance_type in TARGETS:
            try:
                snap = api.get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get(
                    cloud.id, instance_type=instance_type, _request_timeout=20
                ).to_dict()
                body = snap.get("result") or snap
                rollup = body.get("node_rollup") or {}
                entry["target_fleet_snapshots"][instance_type] = {
                    "nodes_total": rollup.get("nodes_total", 0),
                    "nodes_allocated": rollup.get("nodes_allocated", 0),
                    "nodes_idle": rollup.get("nodes_idle", 0),
                    "group_count": len(body.get("groups") or []),
                }
            except Exception as exc:
                entry["target_fleet_snapshots"][instance_type] = {"error": _error(exc)}
        result["clouds"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("fresh output required")
    import anyscale

    value = collect(anyscale.Anyscale()._anyscale_client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cloud_count": len(value["clouds"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
