#!/usr/bin/env python3
"""Validate the step-500 editor packet and print a root-only run skeleton.

The default operation is metadata validation.  This file never invokes an
editor, SSH, a server, or a model.  It hashes only the pinned small packet
files and the VSIX; it deliberately does not open model payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE.parents[4]
CAPSULE = PLAN / "docs/campaign/work/lead/remote-auto350-b/notebook-capsule"
LAN = PLAN / "docs/campaign/work/lead/r2-step500-lan-preparation-v1"
PARENT = PLAN / "docs/campaign/work/lead/r2-step500-rl-gate-v2/parent-manifest.json"
PROFILE = Path("/home/m0hawk/.local/share/sepalith-r2-step500-checks")

VSIX_SHA = "b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1"
VSIX_BYTES = 40_832
CAPSULE_MANIFEST_SHA = "79be4167a2f9159480c8a4ad2747c7019c0e5368d69a1df9e0e474c65e6194a6"
PRIMARY_FILE_SHA = "9459f19a8a29e1158b4e91169da6a4d26c91f6e625f61c5d954a3d2b68201a92"
PRIMARY_CANONICAL_SHA = "59971733ec5a09320c991466e0cd378c26ab9b69d897d2fd192ddeeed2e3b36e"
PARENT_SHA = "05eac983926eacc45b78f59d281eeaeeecfc8c58ed03647b4a8cbe794160afa5"
MODEL_SHA = "d269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db"
MODEL_BYTES = 2_679_710_496
OLD_THETA0_SHA = "22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559"
RENDERER_SHA = "e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78"

SMALL_PINS = {
    LAN / "daily_lan.py": "d653c733832c4980724bba3bf76d38d0c4ffac2e935926d286403e449432299c",
    LAN / "notebook_setup.py": "7b2e60446ebd274080ad861f96751c784376872a3ac1f41bce423fcdc015e97e",
    LAN / "primary-manifest.json": PRIMARY_FILE_SHA,
    LAN / "source-manifest.json": "63b4795e76e61b5bf94f10a22f642166ba0067ee50567004827730a4a76dfe3c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_file(path: Path, expected: str, checks: list[dict]) -> None:
    if not path.is_file():
        checks.append({"name": str(path), "status": "fail", "reason": "missing_file"})
        return
    actual = sha256(path)
    checks.append({
        "name": str(path),
        "status": "pass" if actual == expected else "fail",
        "expected_sha256": expected,
        "actual_sha256": actual,
        "bytes": path.stat().st_size,
    })


def validate() -> dict:
    checks: list[dict] = []
    check_file(CAPSULE / "capsule-manifest.json", CAPSULE_MANIFEST_SHA, checks)
    capsule_manifest_path = CAPSULE / "capsule-manifest.json"
    if capsule_manifest_path.is_file():
        manifest = json.loads(capsule_manifest_path.read_text())
        for relative, expected in manifest.items():
            check_file(CAPSULE / relative, expected, checks)

    vsix = CAPSULE / "candidate.vsix"
    check_file(vsix, VSIX_SHA, checks)
    if vsix.is_file():
        checks.append({
            "name": "candidate.vsix size",
            "status": "pass" if vsix.stat().st_size == VSIX_BYTES else "fail",
            "expected_bytes": VSIX_BYTES,
            "actual_bytes": vsix.stat().st_size,
        })
    for path, expected in SMALL_PINS.items():
        check_file(path, expected, checks)

    primary_path = LAN / "primary-manifest.json"
    primary = json.loads(primary_path.read_text()) if primary_path.is_file() else {}
    checks.append({
        "name": "selected primary manifest canonical identity",
        "status": "pass" if canonical_sha(primary) == PRIMARY_CANONICAL_SHA else "fail",
        "expected_sha256": PRIMARY_CANONICAL_SHA,
        "actual_sha256": canonical_sha(primary) if primary else None,
    })
    model = primary.get("model", {})
    profile = primary.get("modelProfile", {})
    selected_contract = {
        "model_sha256": model.get("sha256"),
        "model_bytes": model.get("bytes"),
        "native_eog": profile.get("nativeEogTokenIds"),
        "bos": profile.get("bosTokenId"),
        "eos": profile.get("canonicalEosTokenId"),
        "vocab_size": profile.get("vocabSize"),
        "context_size": profile.get("contextSize"),
        "max_output_tokens": profile.get("maxOutputTokens"),
        "renderer": profile.get("renderer"),
    }
    expected_contract = {
        "model_sha256": MODEL_SHA,
        "model_bytes": MODEL_BYTES,
        "native_eog": [1, 130073],
        "bos": 0,
        "eos": 1,
        "vocab_size": 130560,
        "context_size": 4096,
        "max_output_tokens": 192,
        "renderer": "zeta2-prm03-v1",
    }
    checks.append({
        "name": "selected step-500 runtime contract",
        "status": "pass" if selected_contract == expected_contract else "fail",
        "expected": expected_contract,
        "actual": selected_contract,
    })

    if PARENT.is_file():
        parent = json.loads(PARENT.read_text())
        parent_ok = (
            sha256(PARENT) == PARENT_SHA
            and parent.get("status") == "accepted"
            and parent.get("kind") == "merged_sft"
            and parent.get("sft_checkpoint_manifest", {}).get("step") == 500
            and parent.get("merged_weights_sha256") == "631b97966d3432ab751785660bb3c8cfe6f984b3524be0169b5bc12d5362752c"
        )
        checks.append({
            "name": "selected accepted SFT parent metadata",
            "status": "pass" if parent_ok else "fail",
            "sha256": sha256(PARENT),
            "step": parent.get("sft_checkpoint_manifest", {}).get("step"),
            "status_field": parent.get("status"),
            "kind": parent.get("kind"),
            "merged_weights_sha256": parent.get("merged_weights_sha256"),
        })
    else:
        checks.append({"name": "selected accepted SFT parent metadata", "status": "fail", "reason": "missing_file"})

    capsule_binding_path = CAPSULE / "binding.json"
    old_binding = json.loads(capsule_binding_path.read_text()) if capsule_binding_path.is_file() else {}
    old_model = old_binding.get("manifest", {}).get("model", {}).get("sha256")
    checks.append({
        "name": "capsule binding requires fresh step-500 rebind",
        "status": "pass" if old_model == OLD_THETA0_SHA and old_model != MODEL_SHA else "fail",
        "checked_old_binding_model_sha256": old_model,
        "selected_model_sha256": MODEL_SHA,
        "action": "ignore capsule binding.json; use the fresh daily_lan binding",
    })
    return {
        "schema": "sepalith.run01.step500-editor-plan-validation.v1",
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "no_launch": True,
        "model_payload_read": False,
        "profile": str(PROFILE),
        "renderer_sha256": RENDERER_SHA,
        "checks": checks,
    }


def command_skeleton() -> list[str]:
    host = "m0hawk@192.168.178.40"
    remote = str(PROFILE)
    capsule = CAPSULE
    run = f"{remote}/runs/<RUN_ID>"
    return [
        "# Root runs after RL cleanup and desktop admission; this list is not executed by editor_plan.py.",
        f"scp -p -o BatchMode=yes -o StrictHostKeyChecking=yes {shlex.quote(str(capsule / 'candidate.vsix'))} {host}:{remote}/package/",
        f"ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes {host} '/usr/bin/code --user-data-dir {remote}/user-data --extensions-dir {remote}/extensions --install-extension {remote}/package/candidate.vsix --force'",
        f"ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes {host} 'sha256sum /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js'",
        f"ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes {host} 'node {remote}/capsule/run_remote_editor.mjs --code /usr/bin/code --vsix {remote}/package/candidate.vsix --vsix-sha256 {VSIX_SHA} --binding {remote}/current-binding.json --binding-sha256 <FRESH-BINDING-SHA256> --instance-id <FRESH-INSTANCE-UUID> --debug-port 19403 --renderer-source /usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js --renderer-sha256 {RENDERER_SHA} --run-root {run} --timeout-ms 180000 --debounce-ms 350'",
        f"ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes {host} 'node {remote}/capsule/analyze_renderer.mjs {run} {run}/renderer-analysis.json'",
        f"ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes {host} 'node {remote}/capsule/analyze_auto.mjs {run}'",
        f"ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes {host} '/usr/bin/Rscript --vanilla -e \"parse(file=commandArgs(trailingOnly=TRUE)[1], keep.source=FALSE); cat(\\\"PASS\\\\n\\\")\" {run}/workspace/auto-accept.R'",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="validate pinned local metadata (default)")
    parser.add_argument("--commands", action="store_true", help="print the deferred root command skeleton")
    args = parser.parse_args()
    if args.commands:
        print("\n".join(command_skeleton()))
        return 0
    result = validate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
