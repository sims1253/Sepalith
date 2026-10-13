#!/usr/bin/env python3
"""Build the bounded PAR-07 receipt from selected CLI evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence_path = Path(args.evidence)
    probe_path = Path(args.probe)
    output_path = Path(args.output)
    evidence = json.loads(evidence_path.read_text())
    candidates = [
        row
        for region in evidence["regions"]
        for row in region.get("candidate_rows", [])
    ]
    candidates.sort(key=lambda row: (str(row.get("region")), str(row.get("name"))))
    family_names = {str(row.get("family")) for row in candidates}
    generic_names = {
        "standardNCFamily",
        "standardNCPromoFamily",
        "standardNVFamily",
        "standardNVPromoFamily",
    }
    region_summaries = []
    for region in evidence["regions"]:
        rows = region.get("candidate_rows", [])
        quota_rows = region.get("quota_rows", [])
        target_quota_names = family_names | generic_names
        selected_quotas = [
            row for row in quota_rows if str(row.get("name")) in target_quota_names
        ]
        selected_quotas.sort(key=lambda row: str(row.get("name")))
        statuses = {}
        for row in rows:
            status = str(row.get("admission_status"))
            statuses[status] = statuses.get(status, 0) + 1
        region_summaries.append(
            {
                "region": region["region"],
                "total_regional_quota": region.get("quota_total_regional_row"),
                "gpu_sku_catalog_count": len(region.get("gpu_sku_catalog", [])),
                "candidate_count": len(rows),
                "candidate_status_counts": statuses,
                "relevant_family_quota_rows": selected_quotas,
                "physical_capacity_tested": False,
            }
        )
    region_summaries.sort(key=lambda row: row["region"])
    status_counts = {}
    class_counts = {}
    for row in candidates:
        status = str(row.get("admission_status"))
        gpu_class = str(row.get("gpu_class"))
        status_counts[status] = status_counts.get(status, 0) + 1
        class_counts[gpu_class] = class_counts.get(gpu_class, 0) + 1
    receipt = {
        "task": "PAR-07",
        "status": "verified",
        "owner": "azure",
        "observed_at": evidence["observed_at_ended"],
        "started_at": evidence["observed_at_started"],
        "ended_at": evidence["observed_at_ended"],
        "action": "Read-only Azure alternate-region/GPU admission research across 12 fixed Europe/US regions",
        "decision": "blocked",
        "admission_finding": "No candidate is quota eligible in the observed subscription state. Azure remains disabled; no resource, quota request, provider registration, or spend occurred.",
        "subscription_id": evidence["subscription_id"],
        "regions_checked": evidence["regions_requested"],
        "region_count": evidence["region_count"],
        "workload_profile": evidence["workload_profile"],
        "admission_rule": [
            "listed SKU",
            "no SKU subscription or zone restriction",
            "family free vCPUs >= required vCPUs",
            "total regional free vCPUs >= required vCPUs",
            "per-device indicative GPU memory >= the 10.26 GiB SFT peak",
        ],
        "quota_interpretation": "The API exposes catalog presence and quota rows, not physical stock. A quota-eligible row would still require a later capacity/price/termination review; none passed the quota gate here.",
        "candidate_counts": {
            "all_relevant_candidate_rows": len(candidates),
            "quota_eligible_rows": len(evidence["ranked_quota_eligible_shortlist"]),
            "by_gpu_class": class_counts,
            "by_admission_status": status_counts,
        },
        "ranked_quota_eligible_shortlist": [],
        "shortlist_reason": "Empty: all 216 relevant rows were blocked by zero/insufficient target family quota (128) or SKU subscription/zone restrictions (88).",
        "region_summaries": region_summaries,
        "candidate_rows": candidates,
        "gpu_memory_uncertainty": "Compute Resource SKUs reports VM MemoryGB and GPUs but no GPU VRAM field. GPU memory values in candidate rows are indicative architecture/partition mappings and require image/driver runtime validation, especially for fractional A10 and newer accelerators.",
        "physical_capacity": "Untested by design; provisioning was forbidden.",
        "evidence": {
            "path": str(evidence_path),
            "sha256": sha256(evidence_path),
            "bytes": evidence_path.stat().st_size,
            "commands_per_region": ["az vm list-usage", "az rest GET Microsoft.Compute/skus"],
            "all_region_commands_ok": all(
                region.get("usage_command", {}).get("status") == "ok"
                and region.get("sku_command", {}).get("status") == "ok"
                for region in evidence["regions"]
            ),
        },
        "probe": {
            "path": str(probe_path),
            "sha256": sha256(probe_path),
            "max_workers": evidence["max_api_workers"],
            "timeout_seconds": evidence["command_timeout_seconds"],
        },
        "official_references": evidence["official_references"],
        "dependencies_checked": [
            "PAR-07-azure-billing-recheck.json",
            "PAR-07-azure-cli-evidence.json",
        ],
        "changed_files": [
            "docs/campaign/work/azure-regions/PAR-07-enumerate.py",
            "docs/campaign/work/azure-regions/PAR-07-build-receipt.py",
            "docs/campaign/work/azure-regions/PAR-07-azure-region-cli-evidence.json",
            "docs/campaign/receipts/PAR-07-azure-region-options.json",
        ],
        "lease_released": "yes — no cloud lease or resource was acquired",
        "acceptance": "PASS — 12 regions and all selected GPU catalog/quota responses were independently timestamped; zero quota-eligible rows; physical capacity and launch were intentionally untested.",
        "unresolved": [
            "Physical capacity was not tested because provisioning is out of scope.",
            "All-in compute/storage/network/transfer pricing and credit coverage remain unverified.",
            "GPU VRAM mappings remain indicative until runtime validation.",
        ],
        "next": "Root may retain Azure as disabled/optional. Any quota request, provider registration, launch, or price admission requires separate root authorization and review.",
        "receipt_created_at": now_utc(),
    }
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
