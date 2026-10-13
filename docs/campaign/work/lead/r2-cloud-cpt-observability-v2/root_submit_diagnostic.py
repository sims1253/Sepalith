#!/usr/bin/env python3
"""Root-controlled, fail-closed submit wrapper for one bootstrap diagnostic."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
CLOUD_ID = "cld_mh2xgqnvkq5squguqzf5wwwqzm"
PAYLOAD_MANIFEST_SHA256 = "53e1a7f4049effb79e81b15e13c70d50bf616a180c3af099a886276e716a2aa9"
ROOT_REVIEW_SHA256 = "d49797d56160ef179b5aba0a1453f9a4aaa7e8bd72d85d0941c09af8d810fe76"
RESERVATION_USD = 28.0
CAMPAIGN_CEILING_USD = 60.0
AVAILABILITY_MAX_AGE_SECONDS = 120
ENTRYPOINT = "bash payload/diagnostic-root-bound-entry.sh"
BINDING_ENV_NAME = "SEPALITH_CPT_DIAGNOSTIC_BINDING"
MARKER_NAME = "diagnostic-submission-started.json"


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp(epoch: float | None = None) -> str:
    value = time.time() if epoch is None else epoch
    return dt.datetime.fromtimestamp(value, dt.timezone.utc).isoformat()


def epoch(value: str) -> float:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "timestamp timezone required")
    return parsed.timestamp()


def load_contract(root: Path):
    path = root / "payload/diagnostic_contract.py"
    spec = importlib.util.spec_from_file_location("sepalith_diagnostic_submit_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def process_identity(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    root = proc_root / str(pid)
    stat_text = (root / "stat").read_text()
    close = stat_text.rfind(")")
    require(close > 0, "watchdog stat format differs")
    fields = stat_text[close + 2 :].split()
    require(len(fields) >= 20, "watchdog stat is incomplete")
    cmdline = (root / "cmdline").read_bytes().split(b"\0")
    argv = [item.decode("utf-8", "strict") for item in cmdline if item]
    return {"pid": pid, "start_tick": fields[19], "argv": argv}


def normalize_availability(document: dict[str, Any]) -> dict[str, Any]:
    credits = document.get("credits", {})
    fleet = document.get("fleet", {}).get("result", document.get("fleet", {}))
    return {
        "observed_epoch": epoch(document["at"]) if "at" in document else float(document["observed_epoch"]),
        "balance_usd": float(credits.get("current_balance_usd", document.get("balance_usd", -1))),
        "spent_usd": float(credits.get("amount_spent_usd", document.get("spent_usd", -1))),
        "fleet_epoch": float(fleet.get("time", fleet.get("snapshot_time", document.get("fleet_epoch", -1)))),
        "nodes_total": int((fleet.get("node_rollup") or {}).get("nodes_total", document.get("nodes_total", -1))),
        "groups_count": len(fleet.get("groups") or []) if "groups" in fleet else int(document.get("groups_count", -1)),
    }


def validate_availability(value: dict[str, Any], now: float) -> dict[str, Any]:
    current = normalize_availability(value)
    require(0 <= now - current["observed_epoch"] <= AVAILABILITY_MAX_AGE_SECONDS, "billing observation is stale or future")
    require(0 <= now - current["fleet_epoch"] <= AVAILABILITY_MAX_AGE_SECONDS, "fleet observation is stale or future")
    require(current["nodes_total"] == 0 and current["groups_count"] == 0, "provider fleet is not empty")
    require(current["balance_usd"] >= RESERVATION_USD, "USD28 funded balance unavailable")
    require(current["spent_usd"] + RESERVATION_USD <= CAMPAIGN_CEILING_USD, "campaign USD60 ceiling would be exceeded")
    return current


def validate_payload(root: Path) -> dict[str, Any]:
    manifest_path = root / "payload-manifest.json"
    require(sha(manifest_path) == PAYLOAD_MANIFEST_SHA256, "payload manifest differs")
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest.get("files", [])) == 17, "frozen payload file denominator differs")
    seen = set()
    for record in manifest["files"]:
        relative = record["path"]
        require(relative not in seen, "duplicate payload manifest path")
        seen.add(relative)
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "payload file missing or symlinked")
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "payload file hash or size differs")
    return manifest


def validate_admission(root: Path, binding: dict[str, Any], now: float) -> tuple[dict[str, Any], dict[str, Any]]:
    admission_path = root / binding["root_admission_relative_path"]
    require(admission_path.is_file() and not admission_path.is_symlink(), "root diagnostic admission missing")
    require(sha(admission_path) == binding["root_admission_sha256"], "root diagnostic admission hash differs")
    admission = json.loads(admission_path.read_text())
    required = {
        "schema": "sepalith.cloud-cpt.bootstrap-diagnostic-root-admission.v1",
        "status": "admitted_bootstrap_diagnostic",
        "admitted": True,
        "paid_launch_authorized": True,
        "run_id": binding["run_id"],
        "payload_manifest_sha256": binding["payload_manifest_sha256"],
        "root_review_sha256": ROOT_REVIEW_SHA256,
        "max_incremental_charge_usd": RESERVATION_USD,
        "campaign_ceiling_usd": CAMPAIGN_CEILING_USD,
    }
    for key, expected in required.items():
        require(admission.get(key) == expected, f"root admission {key} differs")
    require(0 <= now - epoch(admission["at"]) <= AVAILABILITY_MAX_AGE_SECONDS, "root admission is stale or future")
    resource = admission.get("resource", {})
    require(resource == {"provider": "Anyscale Hosted", "instance_type": "g5.2xlarge", "head_nodes": 1,
                         "worker_nodes": 0, "provider_seconds": 600, "watchdog_seconds": 900, "max_retries": 0},
            "root admission resource differs")
    availability_path = root / "root-availability.json"
    require(sha(availability_path) == admission.get("root_availability_sha256"), "admitted root availability differs")
    availability = json.loads(availability_path.read_text())
    return admission, validate_availability(availability, now)


def validate_watchdog(root: Path, binding: dict[str, Any], admission: dict[str, Any], now: float,
                      process_probe: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
    armed_path = root / binding["watchdog_armed_receipt_relative_path"]
    require(armed_path.is_file() and not armed_path.is_symlink(), "watchdog armed receipt missing")
    require(sha(armed_path) == binding["watchdog_armed_receipt_sha256"], "watchdog armed receipt hash differs")
    armed = json.loads(armed_path.read_text())
    name = "sepalith-cpt-" + binding["run_id"]
    require(armed.get("name") == name and armed.get("deadline") == binding["absolute_deadline_utc"], "watchdog armed name or deadline differs")
    identity = admission.get("watchdog_identity", {})
    require(identity == {"pid": armed.get("pid"), "start_tick": identity.get("start_tick"), "name": name,
                         "deadline": binding["absolute_deadline_utc"]}, "admitted watchdog identity differs")
    require(re.fullmatch(r"[0-9]+", str(identity.get("start_tick", ""))) is not None, "watchdog start tick missing")
    observed = process_probe(int(armed["pid"]))
    require(observed["pid"] == armed["pid"] and str(observed["start_tick"]) == str(identity["start_tick"]), "live watchdog PID/start tick differs")
    argv = observed["argv"]
    require(any(Path(item).name == "deadline_watchdog.py" for item in argv), "live watchdog executable differs")
    require("--name" in argv and argv[argv.index("--name") + 1] == name, "live watchdog name argument differs")
    require("--deadline-utc" in argv and argv[argv.index("--deadline-utc") + 1] == binding["absolute_deadline_utc"], "live watchdog deadline argument differs")
    require(now + 600 <= epoch(binding["absolute_deadline_utc"]), "insufficient provider diagnostic window remains")
    return observed


def build_config(job_config_cls, root: Path, binding: dict[str, Any], token: str):
    require(bool(re.fullmatch(r"hf_[A-Za-z0-9]+", token)), "private Hugging Face token unavailable")
    return job_config_cls(
        name="sepalith-cpt-" + binding["run_id"], cloud="Anyscale Cloud", working_dir=str(root),
        image_uri=binding["image_uri"], ray_version="2.57.0", entrypoint=ENTRYPOINT,
        compute_config={"head_node": {"instance_type": "g5.2xlarge"}, "worker_nodes": []},
        max_retries=0, timeout_s=600,
        env_vars={"HF_TOKEN": token, "PYTHONDONTWRITEBYTECODE": "1", BINDING_ENV_NAME: "binding.root.json"},
        excludes=["__pycache__", "*.pyc", ".env", ".git", "root_submit_diagnostic.py",
                  "root_submit_diagnostic_with_private_token.py", "test_root_submit_diagnostic.py", "*.log"],
    )


def prepare(root: Path, now: float, live_availability: dict[str, Any], process_probe=process_identity) -> tuple[dict[str, Any], dict[str, Any]]:
    binding_path = root / "binding.root.json"
    require(binding_path.is_file() and not binding_path.is_symlink(), "root binding missing")
    binding = json.loads(binding_path.read_text())
    contract = load_contract(root)
    contract.validate(binding, now=now)
    require(binding["payload_manifest_sha256"] == PAYLOAD_MANIFEST_SHA256, "binding payload hash differs")
    validate_payload(root)
    review = json.loads((root / "root-review.json").read_text())
    require(sha(root / "root-review.json") == ROOT_REVIEW_SHA256 and review.get("root_tests_passed") == 22,
            "root review or 22-test gate differs")
    admission, admitted_availability = validate_admission(root, binding, now)
    live = validate_availability(live_availability, now)
    watchdog = validate_watchdog(root, binding, admission, now, process_probe)
    marker = root / MARKER_NAME
    require(not marker.exists(), "diagnostic submission marker already exists")
    require(not (root / "diagnostic-submission-receipt.json").exists(), "diagnostic submission receipt already exists")
    return binding, {"admitted_availability": admitted_availability, "live_availability": live, "watchdog": watchdog}


def submit_once(root: Path, token: str, now: float, live_availability: dict[str, Any], job_config_cls,
    submitter: Callable[[Any], str], process_probe=process_identity) -> dict[str, Any]:
    binding, evidence = prepare(root, now, live_availability, process_probe)
    logging.disable(logging.CRITICAL)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        config = build_config(job_config_cls, root, binding, token)
    marker = root / MARKER_NAME
    with marker.open("x") as handle:
        json.dump({"at": stamp(now), "run_id": binding["run_id"], "pid": os.getpid(),
                   "payload_manifest_sha256": PAYLOAD_MANIFEST_SHA256}, handle, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            job_id = submitter(config)
        require(isinstance(job_id, str) and re.fullmatch(r"prodjob_[a-z0-9]+", job_id) is not None,
                "provider job ID differs")
        receipt = {"at": stamp(), "status": "submitted_not_runtime_accepted", "job_id": job_id,
                   "name": "sepalith-cpt-" + binding["run_id"], "run_id": binding["run_id"],
                   "provider_timeout_seconds": 600, "watchdog_timeout_seconds": 900,
                   "max_retries": 0, "max_incremental_charge_usd": RESERVATION_USD,
                   "absolute_deadline_utc": binding["absolute_deadline_utc"]}
        (root / "diagnostic-submission-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
    except BaseException as error:
        failure = {"at": stamp(), "status": "submission_failed_no_retry_authorized",
                   "error_type": type(error).__name__, "message": "Provider submission failed; raw output suppressed",
                   "run_id": binding["run_id"]}
        (root / "diagnostic-submission-error.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def query_live(anyscale_module, now: float) -> dict[str, Any]:
    api = anyscale_module.Anyscale()._anyscale_client._internal_api_client
    credits_raw = api.get_credits_v2_api_v2_organization_billing_credits_v2_get(_request_timeout=20).to_dict()
    credits = credits_raw.get("result") or credits_raw
    fleet_raw = api.get_cloud_gpu_status_api_v2_clouds_cloud_id_gpu_status_get(CLOUD_ID, _request_timeout=20).to_dict()
    fleet = fleet_raw.get("result") or fleet_raw
    return {"observed_epoch": now, "balance_usd": credits["current_balance_usd"],
            "spent_usd": credits["amount_spent_usd"], "fleet_epoch": fleet.get("time", fleet.get("snapshot_time")),
            "nodes_total": (fleet.get("node_rollup") or {})["nodes_total"], "groups_count": len(fleet.get("groups") or [])}


def main() -> None:
    import anyscale
    from anyscale.job.models import JobConfig
    now = time.time()
    logging.disable(logging.CRITICAL)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        live = query_live(anyscale, now)
    receipt = submit_once(ROOT, os.environ.get("HF_TOKEN", ""), time.time(), live, JobConfig, anyscale.job.submit)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps({"status": "pre_or_submission_rejected", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
