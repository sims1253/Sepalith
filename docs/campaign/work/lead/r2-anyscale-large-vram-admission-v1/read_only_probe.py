#!/usr/bin/env python3
"""Read-only Anyscale hosted-cloud and GPU evidence probe.

This script calls only list/search/GET SDK methods. It cannot create a compute
config, cluster, job, workspace, quota, or cloud resource.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


SDK = "/home/m0hawk/.local/share/uv/tools/anyscale/lib/python3.14/site-packages"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_client():
    import anyscale
    return anyscale.Anyscale()._anyscale_client


def collect(client) -> dict:
    clouds = client.list_clouds(count=20).results
    result = {
        "schema": "sepalith.anyscale-large-vram-read-only-evidence.v1",
        "observed_at": now(),
        "sdk": SDK,
        "methods": [
            "list_clouds",
            "list_resource_quotas",
            "get_cloud_additional_instance_types...GET",
            "get_cloud_gpu_status_instance_types...GET",
            "search_cluster_computes",
        ],
        "clouds": [],
    }
    api = client._internal_api_client
    for cloud in clouds:
        entry = {
            "id": cloud.id,
            "name": cloud.name,
            "provider": str(cloud.provider),
            "region": cloud.region,
            "state": str(cloud.state),
            "status": str(cloud.status),
            "compute_stack": str(cloud.compute_stack),
        }
        quotas = client.list_resource_quotas(cloud_id=cloud.id, max_items=50)
        entry["explicit_resource_quotas"] = [
            {
                "name": quota.name,
                "enabled": bool(quota.is_enabled),
                "soft": bool(quota.is_soft_quota),
                "quota": quota.quota.to_dict() if hasattr(quota.quota, "to_dict") else quota.quota,
            }
            for quota in quotas
        ]
        additional = api.get_cloud_additional_instance_types_api_v2_clouds_cloud_id_additional_instance_types_get(cloud.id)
        entry["additional_instance_types"] = additional.to_dict()
        fleet = api.get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(cloud.id)
        entry["observed_gpu_fleet_instance_types"] = fleet.to_dict()
        computes = client.search_cluster_computes({
            "cloud_id": cloud.id, "include_anonymous": True, "version": -1,
        }).results
        entry["registered_compute_configs"] = [
            {
                "id": item.id,
                "name": item.name,
                "head_instance_type": item.config.head_node_type.instance_type,
                "worker_instance_types": [node.instance_type for node in item.config.worker_node_types],
            }
            for item in computes
        ]
        result["clouds"].append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("fresh output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = collect(load_client())
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"clouds": len(value["clouds"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
