#!/usr/bin/env python3
"""Create the exact short-lived admission after root source review."""
import argparse, datetime, hashlib, json, os, re
from pathlib import Path

PACKET_SHA = "2b6dd8210ede1db19b00fa1aa8e0815eb4034618acfbc920a5facd31ae27cfb3"
SOURCE_MANIFEST_SHA = "fa18fefe0d771f8493b59ad08b0968ab7cc62e6532691c2e9ec73f331221db9c"
MODEL_SHA = "9b11c5275202b83c6e6299890582fc55526113586c4ae3f8daf3bc0ac1ab54bd"
MODEL_INTEGRITY_SHA = "b5a3450b052cdace5c662b4ddc74b25b40e8e90a0834faf63efd7c502194b01c"

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def build(run_id, reviewed_source_manifest_sha256, now=None):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("unsafe run ID")
    if reviewed_source_manifest_sha256 != SOURCE_MANIFEST_SHA:
        raise ValueError("root-reviewed source manifest mismatch")
    created = now or datetime.datetime.now(datetime.timezone.utc)
    if created.tzinfo is None:
        raise ValueError("created time must be timezone-aware")
    expires = created + datetime.timedelta(minutes=30)
    return {
        "schema": "sepalith.run06.e750-notebook-cpu-root-admission.v1",
        "status": "admitted",
        "packet_sha256": PACKET_SHA,
        "model_sha256": MODEL_SHA,
        "model_integrity_sha256": MODEL_INTEGRITY_SHA,
        "caps": [192, 384, 768],
        "maximum_threads": 2,
        "offline_case_deadline_seconds": 120,
        "maximum_seconds": 28800,
        "run_id": run_id,
        "created_at": created.isoformat(),
        "expires_at": expires.isoformat(),
        "host": "m0hawk@192.168.178.40",
        "purpose": "offline_DEV75_CPU_quality_not_production_latency",
        "source_reviewed": True,
        "remote_packet_source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "issued_by": "root_after_independent_review",
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--reviewed-source-manifest-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("admission output must be fresh")
    value = build(a.run_id, a.reviewed_source_manifest_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(value, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    fd = os.open(a.output.parent, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    print(json.dumps({"status":"admission_issued","path":str(a.output),"sha256":hashlib.sha256(a.output.read_bytes()).hexdigest(),"run_id":a.run_id}))

if __name__ == "__main__":
    main()
