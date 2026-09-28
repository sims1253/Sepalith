#!/usr/bin/env python3
"""Bind root-audited model hashes into the notebook quant packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "q8": "d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db",
    "q4_calibrated": "d7a8438b2ba1019f88e72f5dc6beccde3428111811ce254985556df1fe61ef9b",
    "iq3": "e465637549d52bea28cf9017d5340a32bad259b445c6670076cd7031abbf3fed",
    "iq2": "69c1f9a6c6c7331085c410c7d240fa40ce28834c82fde91acc4be81f23411c11",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(template: Path, output: Path, verify_files: bool) -> dict:
    packet = json.loads(template.read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError(f"refusing existing output: {output}")
    for arm, expected in EXPECTED.items():
        entry = packet["models"][arm]
        marker = entry["sha256"]
        if not marker.startswith("__ROOT_BIND_"):
            raise ValueError(f"{arm} template marker is missing")
        model_path = Path(entry["path"])
        if verify_files:
            if not model_path.is_file() or model_path.stat().st_size != entry["bytes"]:
                raise ValueError(f"{arm} path/size mismatch: {model_path}")
            actual = sha256(model_path)
            if actual != expected:
                raise ValueError(f"{arm} SHA mismatch: {actual}")
        entry["sha256"] = expected
        entry["binding"] = "root_export_audit_and_remote_full_sha256"
    packet["status"] = "root_hashes_bound_no_model_load"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    bind(args.template, args.out, args.verify_files)
    print(json.dumps({"status": "bound", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
