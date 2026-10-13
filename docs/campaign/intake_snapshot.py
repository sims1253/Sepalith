#!/usr/bin/env python3
"""Preserve campaign source without importing training code or using CUDA."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

PLAN = Path(__file__).resolve().parents[2]
ROOTS = {
    "plan": PLAN,
    "owner": Path("/home/m0hawk/.t3/worktrees/Sepalith/t3code-e6aed5ff"),
    "canonical": Path("/home/m0hawk/Documents/Sepalith"),
}


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path("/mnt/e/sepalith/campaign-20260915/source-snapshots") / stamp
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "task": "PRE-01", "observed_at": stamp, "snapshot": str(destination),
        "roots": {}, "artifacts": [], "secrets": "Environment and credential files excluded",
        "source_map": {
            "plan": "Task specifications, shared campaign state, contracts and research evidence",
            "owner": "Current experiment source; do not change while its O2 process is active",
            "canonical": "Asset/environment authority; dirty source preserved separately, not a launch checkout",
            "cv_reference": "/home/m0hawk/.t3/worktrees/Sepalith/t3code-3d83226e",
        },
        "protected_paths": ["/mnt/h/sepalith/runs/o2-control-1800-20260911",
                            "/mnt/h/sepalith/runs/o2-filtered-1800-20260911",
                            "/mnt/h/sepalith/runs/pft1_b4_merged",
                            "/home/m0hawk/Documents/Sepalith/.venv-sft"],
        "launch_policy": "Select reviewed files from this inventory; freeze a new recipe snapshot after campaign edits. No launch is admitted by this inventory alone.",
    }
    for role, root in ROOTS.items():
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        ).decode().split("\0")
        # Some campaign research inputs are ignored by Git. They still define launches.
        names += [str(p.relative_to(root)) for p in (root / "docs/research").glob("72h*") if p.is_file()]
        record = {"path": str(root), "head": command(["git", "rev-parse", "HEAD"], root),
                  "dirty_status": command(["git", "status", "--short", "--untracked-files=all"], root).splitlines(),
                  "files": [], "excluded": [], "missing": []}
        for name in sorted(set(filter(None, names))):
            p = root / name
            if any(part in {".env", ".env.local", "credentials.json", "auth.json", "kaggle.json"}
                   or part.endswith((".pem", ".key")) for part in Path(name).parts):
                record["excluded"].append(name)
                continue
            if not p.is_file():
                record["missing"].append(name)
                continue
            before = p.stat()
            target = destination / role / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            digest = sha256(target)
            if digest != sha256(p) or before.st_mtime_ns != p.stat().st_mtime_ns:
                raise RuntimeError(f"Source changed during capture: {p}")
            record["files"].append({"path": name, "bytes": before.st_size, "sha256": digest,
                                    "symlink_target": os.readlink(p) if p.is_symlink() else None})
        manifest["roots"][role] = record
    canonical = ROOTS["canonical"]
    for rel in ["experiments/models/b4_qwen35_2b-Q8_0.gguf", "experiments/models/packaging_b4-Q8_0.gguf"]:
        p = canonical / rel
        manifest["artifacts"].append({"path": str(p), "bytes": p.stat().st_size,
                                      "sha256": sha256(p), "disposition": "Protected in place; do not regenerate"})
    manifest["primary_model"] = {
        "repo_id": "openbmb/MiniCPM5-2B-Midtrain",
        "revision": "8dc5f6055b90fe4b9422340810b270b9569f37f3",
        "cache": "/home/m0hawk/.cache/huggingface/hub/models--openbmb--MiniCPM5-2B-Midtrain",
        "acceptance": "Pinned hypothesis only; model tensor and tokenizer verification pending SFT-01/PRE-05",
    }
    env_python = canonical / ".venv-sft/bin/python"
    env_probe = "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':sorted([(d.metadata['Name'],d.version) for d in m.distributions()])}))"
    manifest["environment"] = json.loads(command([str(env_python), "-c", env_probe]))
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    (destination / "manifest.json").write_bytes(encoded)
    (destination / "manifest.sha256").write_text(hashlib.sha256(encoded).hexdigest() + "  manifest.json\n")
    # Read-only modes are best-effort on the Windows mount; hashes enforce identity.
    for p in destination.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    receipt = PLAN / "docs/campaign/receipts/PRE-01-source-snapshot.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(encoded)
    print(json.dumps({"receipt": str(receipt), "snapshot": str(destination),
                      "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                      "source_files": {k: len(v["files"]) for k, v in manifest["roots"].items()},
                      "artifacts": manifest["artifacts"]}))


if __name__ == "__main__":
    main()
