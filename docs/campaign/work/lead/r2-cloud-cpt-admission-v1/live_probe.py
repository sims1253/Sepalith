#!/usr/bin/env python3
"""Read-only cloud/provider admission probe; never prints credential values."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
CLOUD_ID = "cld_mh2xgqnvkq5squguqzf5wwwqzm"


def main() -> None:
    observed = dt.datetime.now(dt.timezone.utc).isoformat()
    result: dict = {"schema": "sepalith.cloud-cpt.live-readonly.v1", "observed_at": observed}

    try:
        import anyscale

        api = anyscale.Anyscale()._anyscale_client._internal_api_client
        fleet_raw = api.get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get(
            CLOUD_ID, _request_timeout=20
        ).to_dict()
        fleet = fleet_raw.get("result") or fleet_raw
        credits_raw = api.get_credits_v2_api_v2_organization_billing_credits_v2_get(
            _request_timeout=20
        ).to_dict()
        credits = credits_raw.get("result") or credits_raw
        result["anyscale"] = {
            "cloud_id": CLOUD_ID,
            "gpu_status": {
                "snapshot_time": fleet.get("snapshot_time", fleet.get("time")),
                "node_rollup": fleet.get("node_rollup"),
                "groups_count": len(fleet.get("groups") or []),
                "interpretation": "current fleet occupancy only; this endpoint does not prove allocatable provider capacity",
            },
            "credits": {
                key: credits.get(key)
                for key in ("current_balance_usd", "amount_spent_usd", "total_granted_usd")
            },
        }
    except ModuleNotFoundError:
        # The Anyscale CLI is isolated in its uv tool environment. Query it
        # there without copying authentication material into this process.
        helper = r'''import anyscale,json
api=anyscale.Anyscale()._anyscale_client._internal_api_client
cid="cld_mh2xgqnvkq5squguqzf5wwwqzm"
fleet_raw=api.get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get(cid,_request_timeout=20).to_dict();fleet=fleet_raw.get("result") or fleet_raw
credits_raw=api.get_credits_v2_api_v2_organization_billing_credits_v2_get(_request_timeout=20).to_dict();credits=credits_raw.get("result") or credits_raw
types_raw=api.get_cloud_gpu_status_instance_types_api_v2_clouds_cloud_id_gpu_status_instance_types_get(cid,_request_timeout=20).to_dict();types=types_raw.get("result") or types_raw
active=credits.get("in_use_credits") or []
print(json.dumps({"cloud_id":cid,"gpu_status":{"snapshot_time":fleet.get("snapshot_time",fleet.get("time")),"node_rollup":fleet.get("node_rollup"),"groups_count":len(fleet.get("groups") or []),"active_instance_types":types.get("instance_types") or [],"interpretation":"current fleet occupancy only; this endpoint does not prove allocatable provider capacity"},"credits":{**{k:credits.get(k) for k in ("current_balance_usd","amount_spent_usd","total_granted_usd")},"active_credit_expiry_max":max((str(r.get("effective_date_end")) for r in active),default=None)}}))'''
        probe = subprocess.run(
            ["/home/m0hawk/.local/share/uv/tools/anyscale/bin/python", "-c", helper],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode == 0:
            result["anyscale"] = json.loads(probe.stdout)
        else:
            result["anyscale"] = {"error_type": "AnyscaleToolProbeFailed"}
    except Exception as error:  # preserve a safe blocker without exception text
        result["anyscale"] = {"error_type": type(error).__name__}

    try:
        from huggingface_hub import HfApi, get_token

        token = get_token()
        info = HfApi(token=token).repo_info("scholzmx/sepalith-lora", repo_type="model")
        result["private_transport"] = {
            "provider": "huggingface_hub",
            "repo_id": info.id,
            "token_available": bool(token),
            "repo_accessible": True,
            "repo_private": bool(info.private),
            "head_sha": info.sha,
            "probe_mode": "read_only_repo_info",
        }
    except Exception as error:
        result["private_transport"] = {
            "repo_id": "scholzmx/sepalith-lora",
            "repo_accessible": False,
            "error_type": type(error).__name__,
        }

    # Kaggle has no public quota endpoint. Authenticated list/status calls still
    # establish account access and whether an already uploaded private dataset
    # can supply this run; they do not consume accelerator time.
    env = dict(os.environ)
    proc = subprocess.run(
        ["kaggle", "datasets", "list", "--mine", "--csv"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    refs = []
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        refs = [line.split(",", 1)[0] for line in lines[1:]]
    result["kaggle"] = {
        "authenticated_dataset_list_ok": proc.returncode == 0,
        "owned_dataset_refs": refs,
        "remaining_gpu_quota": None,
        "quota_reason": "Kaggle's public API/CLI exposes kernels and datasets but no remaining GPU-hours endpoint",
        "artifact_ready": any("sepalith-cpt-remaining" in ref for ref in refs),
    }

    destination = HERE / "live-provider-status.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
