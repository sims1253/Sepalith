#!/usr/bin/env python3
"""Read-only Azure GPU admission probe for PAR-07.

The probe reads the Compute Resource SKUs and per-region VM usage endpoints for
a fixed Europe/US region set.  It never provisions, registers, requests quota,
or starts/deletes a resource.  The JSON printed on stdout contains selected
fields from the API responses, command timestamps, and response hashes; full
provider responses are intentionally not retained.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


SUBSCRIPTION_ID = "db41dff1-3bf8-4f89-8465-5425ca316fd8"
API_VERSION = "2021-07-01"
TIMEOUT_SECONDS = 60
MAX_WORKERS = 2
REGIONS = [
    "westeurope",
    "northeurope",
    "francecentral",
    "germanywestcentral",
    "swedencentral",
    "uksouth",
    "polandcentral",
    "italynorth",
    "eastus",
    "eastus2",
    "westus3",
    "southcentralus",
]

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def number(value: object) -> int | float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        parsed = float(text)
        return int(parsed) if parsed.is_integer() else parsed
    except (TypeError, ValueError):
        return None


def capabilities(item: dict[str, Any]) -> dict[str, object]:
    return {
        normalized(cap.get("name")): cap.get("value")
        for cap in item.get("capabilities", [])
        if isinstance(cap, dict)
    }


def safe_restrictions(item: dict[str, Any]) -> list[dict[str, object]]:
    result = []
    for restriction in item.get("restrictions", []) or []:
        if not isinstance(restriction, dict):
            continue
        info = restriction.get("restrictionInfo")
        result.append(
            {
                "type": restriction.get("type"),
                "reasonCode": restriction.get("reasonCode"),
                "values": restriction.get("values"),
                "restrictionInfo": info if isinstance(info, dict) else None,
            }
        )
    return result


def command_result(argv: list[str], parse_json: bool = True) -> dict[str, object]:
    command = shlex.join(argv)
    started = now_utc()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        stdout = completed.stdout or ""
        result: dict[str, object] = {
            "command": command,
            "started_at": started,
            "ended_at": now_utc(),
            "exit_code": completed.returncode,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stdout_bytes": len(stdout.encode()),
        }
        if completed.returncode != 0:
            # Do not retain arbitrary CLI diagnostics: they can contain tenant
            # or identity material.  A short, redacted status is sufficient.
            result["status"] = "error"
            result["stderr_present"] = bool(completed.stderr)
            return result
        result["status"] = "ok"
        if parse_json:
            try:
                result["json"] = json.loads(stdout)
            except json.JSONDecodeError:
                result["status"] = "invalid_json"
        return result
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "started_at": started,
            "ended_at": now_utc(),
            "status": "timeout",
            "exit_code": None,
        }
    except OSError as exc:
        return {
            "command": command,
            "started_at": started,
            "ended_at": now_utc(),
            "status": "exec_error",
            "exit_code": None,
            "error_type": type(exc).__name__,
        }


def gpu_classification(name: str, family: str) -> str | None:
    text = f"{name} {family}".lower()
    if "t4" in text or "ncasv3_t4" in text:
        return "T4_NCasT4v3"
    if "ncs" in text and "v3" in text:
        return "V100_NCv3"
    if "a100" in text or "ncadsa100" in text or "ndasv4_a100" in text:
        return "A100_NCadsA100_or_NDA100"
    # Check A100 before A10 because the string "a100" starts with "a10".
    if "a10" in text or "nvadsa10v5" in text:
        return "A10_NVadsA10v5"
    if "h100" in text:
        return "H100_NCadsH100_or_NDH100"
    if "h200" in text:
        return "H200_newer_GPU"
    if "mi300" in text:
        return "MI300X_newer_GPU"
    if "gb200" in text or "gb300" in text or "b200" in text:
        return "Blackwell_newer_GPU"
    return None


def gpu_memory_map(name: str, family: str, gpu_count: int | float | None) -> dict[str, object]:
    """Return an indicative architecture map; the SKU API has no GPU VRAM field."""
    text = name.lower()
    count = float(gpu_count) if gpu_count is not None else None
    per_gpu: float | None = None
    basis = "unknown; API reports GPUs but not GPU VRAM"
    if "t4" in text or "ncasv3_t4" in family.lower():
        per_gpu, basis = 16.0, "NVIDIA T4 specification mapped from NCasT4v3 family"
    elif "a100" in text or "ncadsa100" in family.lower() or "ndasv4_a100" in family.lower():
        per_gpu, basis = 40.0, "NVIDIA A100 40-GiB specification mapped from A100 family"
    elif "a10" in text or "nvadsa10v5" in family.lower():
        # NVadsA10v5 exposes GPU partitions for the first three sizes.
        if "nv6ads_a10" in text:
            per_gpu, basis = 4.0, "A10 24-GiB GPU, 1/6 partition"
        elif "nv12ads_a10" in text:
            per_gpu, basis = 8.0, "A10 24-GiB GPU, 1/3 partition"
        elif "nv18ads_a10" in text:
            per_gpu, basis = 12.0, "A10 24-GiB GPU, 1/2 partition"
        else:
            per_gpu, basis = 24.0, "NVIDIA A10 specification mapped from full NVadsA10v5 partition"
    elif ("ncs" in text and "v3" in text) or "ncs" in family.lower():
        per_gpu, basis = 16.0, "NVIDIA V100 specification mapped from NCv3 family"
    elif "h100" in text:
        per_gpu, basis = 80.0, "NVIDIA H100 80-GiB specification mapped from H100 family"
    elif "h200" in text:
        per_gpu, basis = 141.0, "NVIDIA H200 specification; confirm SKU partition and image"
    elif "mi300" in text:
        per_gpu, basis = 192.0, "AMD MI300X specification; accelerator software compatibility is unverified"
    total = per_gpu * count if per_gpu is not None and count is not None else None
    return {
        "gpu_memory_gib_per_device": per_gpu,
        "gpu_memory_gib_total": total,
        "gpu_memory_basis": basis,
        "gpu_memory_precision": "indicative architecture map, not an Azure API capability; verify at runtime",
    }


def workload_note(
    gpu_class: str,
    per_device_memory: float | None,
    device_count: int | float | None,
) -> str:
    if per_device_memory is None:
        return "Unclassified GPU; no admission recommendation without accelerator/software verification"
    if per_device_memory >= 21.6:
        if device_count is not None and device_count > 1:
            return "RL G4 policy8 nominally fits per device; multi-GPU SKU is oversized for the one-GPU campaign profile"
        return "SFT and RL G4 policy8 nominally fit; RL reserved profile is about 21.6 GiB; validate real sequence length"
    if per_device_memory >= 10.26:
        return "SFT nominally fits the 10.26 GiB peak; 2048–3064 sequence headroom is uncertain; RL reserved profile does not fit"
    return "Below the reported SFT peak; unsuitable for this campaign profile"


def selected_sku(item: dict[str, Any]) -> dict[str, object] | None:
    caps = capabilities(item)
    gpu_count = number(caps.get("gpus"))
    if gpu_count is None or gpu_count <= 0:
        return None
    name = str(item.get("name", ""))
    family = str(item.get("family", ""))
    gpu_class = gpu_classification(name, family)
    # Keep the complete GPU catalog as selected fields, but mark only the
    # families relevant to this campaign as candidate rows.
    memory = number(caps.get("memorygb"))
    vcpus = number(caps.get("vcpusavailable"))
    if vcpus is None:
        vcpus = number(caps.get("vcpus"))
    result: dict[str, object] = {
        "name": name,
        "size": item.get("size"),
        "family": family,
        "locations": item.get("locations", []),
        "gpus": gpu_count,
        "required_vcpus": vcpus,
        "vm_memory_gb": memory,
        "restrictions": safe_restrictions(item),
        "gpu_class": gpu_class,
    }
    result.update(gpu_memory_map(name, family, gpu_count))
    result["useful_workload"] = workload_note(
        str(gpu_class or "unknown"),
        result["gpu_memory_gib_per_device"] if isinstance(result["gpu_memory_gib_per_device"], (int, float)) else None,
        gpu_count,
    )
    return result


def usage_rows(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        return []
    selected = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name_obj = item.get("name")
        if not isinstance(name_obj, dict):
            continue
        key = str(name_obj.get("value", ""))
        key_norm = normalized(key)
        localized = str(name_obj.get("localizedValue", ""))
        # Avoid false positives such as StandardDnv6Family.  GPU quota keys
        # are emitted with the StandardNC/StandardNV/StandardND (or newer
        # NPS/NGA) prefix; the complete matching row is retained even when its
        # limit is zero.
        is_gpu_family = key_norm.endswith("family") and key_norm.startswith(
            ("standardnc", "standardnv", "standardnd", "standardnps", "standardnga", "internalnd")
        )
        if key_norm == "cores" or is_gpu_family:
            selected.append(
                {
                    "name": key,
                    "localized_name": localized,
                    "current": number(item.get("currentValue")),
                    "limit": number(item.get("limit")),
                }
            )
    return selected


def quota_index(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        key = normalized(row.get("name"))
        if key and key not in result:
            result[key] = row
    return result


def add_quota_fields(
    sku: dict[str, object],
    quotas: dict[str, dict[str, object]],
    core_quota: dict[str, object] | None,
) -> dict[str, object]:
    family = str(sku.get("family", ""))
    family_row = quotas.get(normalized(family))
    if family_row is None:
        # Some API records use a family spelling not emitted by usage.  Keep a
        # bounded alias check, preserving missing as unknown rather than zero.
        for alias, row in quotas.items():
            if alias == normalized(family) or (
                alias.endswith("family") and normalized(family).endswith("family") and alias == normalized(family)
            ):
                family_row = row
                break
    required = sku.get("required_vcpus")
    family_current = family_row.get("current") if family_row else None
    family_limit = family_row.get("limit") if family_row else None
    core_current = core_quota.get("current") if core_quota else None
    core_limit = core_quota.get("limit") if core_quota else None
    free_family = (
        max(0, float(family_limit) - float(family_current))
        if isinstance(family_limit, (int, float)) and isinstance(family_current, (int, float))
        else None
    )
    free_core = (
        max(0, float(core_limit) - float(core_current))
        if isinstance(core_limit, (int, float)) and isinstance(core_current, (int, float))
        else None
    )
    restrictions = sku.get("restrictions") or []
    restricted = bool(restrictions)
    enough_family = free_family is not None and isinstance(required, (int, float)) and free_family >= required
    enough_core = free_core is not None and isinstance(required, (int, float)) and free_core >= required
    memory = sku.get("gpu_memory_gib_per_device")
    useful = isinstance(memory, (int, float)) and memory >= 10.26
    if restricted:
        admission = "blocked_subscription_or_zone_restriction"
    elif family_row is None:
        admission = "unknown_family_quota"
    elif not enough_family:
        admission = "blocked_zero_or_insufficient_family_quota"
    elif not enough_core:
        admission = "blocked_zero_or_insufficient_total_regional_quota"
    elif not useful:
        admission = "blocked_insufficient_gpu_memory"
    else:
        admission = "quota_eligible_physical_capacity_untested"
    return {
        **sku,
        "family_quota_name": family_row.get("name") if family_row else None,
        "family_quota_current_vcpus": family_current,
        "family_quota_limit_vcpus": family_limit,
        "free_family_vcpus": int(free_family) if isinstance(free_family, float) and free_family.is_integer() else free_family,
        "total_regional_quota_current_vcpus": core_current,
        "total_regional_quota_limit_vcpus": core_limit,
        "free_core_vcpus": int(free_core) if isinstance(free_core, float) and free_core.is_integer() else free_core,
        "listed_sku": True,
        "subscription_restricted": restricted,
        "quota_eligible": admission == "quota_eligible_physical_capacity_untested",
        "admission_status": admission,
        "physical_capacity_tested": False,
    }


def region_probe(region: str) -> dict[str, object]:
    sku_url = (
        "https://management.azure.com/subscriptions/"
        f"{SUBSCRIPTION_ID}/providers/Microsoft.Compute/skus?api-version={API_VERSION}"
        f"&%24filter=location%20eq%20%27{region}%27"
    )
    usage_argv = [
        "az",
        "vm",
        "list-usage",
        "--location",
        region,
        "--subscription",
        SUBSCRIPTION_ID,
        "--only-show-errors",
        "-o",
        "json",
    ]
    sku_argv = [
        "az",
        "rest",
        "--only-show-errors",
        "--method",
        "get",
        "--url",
        sku_url,
        "-o",
        "json",
    ]
    usage_call = command_result(usage_argv)
    usage_raw = usage_call.pop("json", None)
    sku_call = command_result(sku_argv)
    sku_raw = sku_call.pop("json", None)
    usage = usage_rows(usage_raw) if usage_call.get("status") == "ok" else []
    usage_lookup = quota_index(usage)
    core = usage_lookup.get("cores")
    raw_skus = sku_raw if sku_call.get("status") == "ok" else {}
    sku_items = raw_skus.get("value", []) if isinstance(raw_skus, dict) else []
    catalog = [selected_sku(item) for item in sku_items if isinstance(item, dict)]
    catalog = [item for item in catalog if item is not None]
    candidates = [item for item in catalog if item.get("gpu_class") is not None]
    admitted = [add_quota_fields(item, usage_lookup, core) for item in candidates]
    for item in admitted:
        item["region"] = region
    return {
        "region": region,
        "usage_command": usage_call,
        "sku_command": sku_call,
        "total_sku_records": len(sku_items),
        "gpu_sku_catalog": catalog,
        "quota_rows": usage,
        "candidate_rows": admitted,
        "quota_total_regional_row": core,
    }


def sort_key(row: dict[str, object]) -> tuple[object, ...]:
    workload = str(row.get("useful_workload", ""))
    rl = 0 if "SFT and RL" in workload else 1
    memory = row.get("gpu_memory_gib_per_device")
    memory_distance = abs(float(memory) - 24.0) if isinstance(memory, (int, float)) else 9999.0
    vcpus = row.get("required_vcpus")
    return (rl, memory_distance, float(vcpus) if isinstance(vcpus, (int, float)) else 9999.0, str(row.get("region")), str(row.get("name")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Write evidence JSON to this path")
    args = parser.parse_args()
    started = now_utc()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(region_probe, region): region for region in REGIONS}
        regions = []
        for future in concurrent.futures.as_completed(futures):
            region = futures[future]
            try:
                regions.append(future.result())
            except Exception as exc:  # preserve a bounded failed-region record
                regions.append({"region": region, "status": "probe_error", "error_type": type(exc).__name__})
    regions.sort(key=lambda item: str(item.get("region")))
    candidates = [row for region in regions for row in region.get("candidate_rows", [])]
    eligible = sorted((row for row in candidates if row.get("quota_eligible")), key=sort_key)
    result = {
        "task": "PAR-07",
        "purpose": "Read-only Azure alternate-region/GPU admission research",
        "observed_at_started": started,
        "observed_at_ended": now_utc(),
        "subscription_id": SUBSCRIPTION_ID,
        "login_action": "No az login; existing authenticated CLI session only",
        "api_version": API_VERSION,
        "regions_requested": REGIONS,
        "region_count": len(REGIONS),
        "max_api_workers": MAX_WORKERS,
        "command_timeout_seconds": TIMEOUT_SECONDS,
        "mutation_guard": {
            "read_only_commands": True,
            "provision_or_registration_called": False,
            "quota_request_called": False,
            "physical_capacity_tested": False,
        },
        "workload_profile": {
            "model": "Sepalith R editing 2.5B MiniCPM",
            "precision": "BF16",
            "sft": "LoRA rank32 batch4 peak about 10.26 GiB; real sequence lengths 2048–3064 require headroom validation",
            "rl": "G4 policy8 about 15 GiB allocated and 21.6 GiB reserved",
        },
        "regions": regions,
        "candidate_row_count": len(candidates),
        "quota_eligible_row_count": len(eligible),
        "ranked_quota_eligible_shortlist": eligible,
        "official_references": [
            "https://learn.microsoft.com/en-us/rest/api/compute/resource-skus/list",
            "https://learn.microsoft.com/en-us/rest/api/compute/virtual-machines/list-usage",
            "https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/",
            "https://www.nvidia.com/en-us/data-center/tesla-t4/",
            "https://www.nvidia.com/en-us/data-center/products/a10-gpu/",
            "https://www.nvidia.com/en-us/data-center/tesla-v100/",
            "https://www.nvidia.com/en-us/data-center/a100/",
            "https://www.nvidia.com/en-us/data-center/h100/",
            "https://www.nvidia.com/en-us/data-center/h200/",
            "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
