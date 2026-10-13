#!/usr/bin/env python3
"""Upload a future root-accepted DDP8 full checkpoint; publish closure last."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from return_checkpoint_common import RETURN_EXTRA
from transport_common import PAYLOAD_REQUIRED, atomic_json, hashes, require, safe_message, sha256
from upload_private_checkpoint import client, remote_matches

ACTION = "upload_full_weight_return_checkpoint_private"


def run(spec_path, authorization_path, journal_path, *, api=None, download=None, token=None):
    spec = json.loads(Path(spec_path).read_text())
    require(spec.get("schema") == "sepalith.pre04.full_weight_return_transport_spec.v1" and spec.get("status") == "prepared_upload_not_authorized" and set(spec.get("files", {})) == PAYLOAD_REQUIRED | RETURN_EXTRA, "return spec differs")
    observed_auth = json.loads(Path(authorization_path).read_text())
    expected_auth = {"schema": "sepalith.pre04.full_weight_return_transport_authorization.v1", "status": "admitted", "authorized": True, "action": ACTION, "repo": spec["repo"], "prefix": spec["prefix"], "campaign_manifest_sha256": spec["campaign_manifest_sha256"], "checkpoint_path": spec["checkpoint_path"], "step": spec["step"], "maximum_bytes": spec["total_bytes"]}
    require(observed_auth == expected_auth, "root return authorization differs")
    if api is None:
        api, download, token = client()
    root = Path(spec["checkpoint_path"])
    journal_path = Path(journal_path)
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"schema": "sepalith.pre04.return_upload_journal.v1", "spec_sha256": sha256(spec_path), "completed": {}}
    require(journal.get("schema") == "sepalith.pre04.return_upload_journal.v1" and journal.get("spec_sha256") == sha256(spec_path), "return journal differs")
    for name, expected0 in sorted(spec["files"].items()):
        path = root / name
        before = path.stat()
        observed = hashes(path)
        expected = {**expected0, "git_blob_sha1": observed["git_blob_sha1"]}
        require(observed["bytes"] == expected["bytes"] and observed["sha256"] == expected["sha256"], "return local hash differs:" + name)
        remote = spec["prefix"] + "/files/" + name
        saved = journal["completed"].get(name)
        if saved:
            require(saved.get("remote_path") == remote and saved.get("expected") == expected, "return journal row differs:" + name)
            remote_matches(api, spec, remote, saved["revision"], expected)
        else:
            commit = api.upload_file(repo_id=spec["repo"], repo_type="model", path_or_fileobj=path, path_in_repo=remote, commit_message="Full-weight portable return: " + name)
            remote_matches(api, spec, remote, commit.oid, expected)
            journal["completed"][name] = {"remote_path": remote, "revision": commit.oid, "expected": expected}
            atomic_json(journal_path, journal)
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "return checkpoint changed:" + name)
    require(set(journal["completed"]) == set(spec["files"]), "return journal closure incomplete")
    closure = {"schema": "sepalith.pre04.full_weight_return_closure.v1", "status": "complete", "repo": spec["repo"], "prefix": spec["prefix"], "campaign_manifest_sha256": spec["campaign_manifest_sha256"], "step": spec["step"], "files": spec["files"], "total_bytes": spec["total_bytes"], "source_world_size": 8, "portable_world_size": 1, "credential_persisted": False}
    closure_path = journal_path.with_name("return-closure.json")
    atomic_json(closure_path, closure)
    closure_sha = sha256(closure_path)
    remote = spec["prefix"] + "/closure.json"
    commit = api.upload_file(repo_id=spec["repo"], repo_type="model", path_or_fileobj=closure_path, path_in_repo=remote, commit_message="Publish full-weight return closure")
    back = Path(download(spec["repo"], filename=remote, revision=commit.oid, repo_type="model", token=token))
    require(sha256(back) == closure_sha, "return closure readback differs")
    receipt = {"schema": "sepalith.pre04.return_upload_receipt.v1", "status": "complete", "repo": spec["repo"], "prefix": spec["prefix"], "revision": commit.oid, "closure_sha256": closure_sha, "step": spec["step"], "files": len(spec["files"]), "bytes": spec["total_bytes"], "portable_world_size": 1, "credential_persisted": False}
    atomic_json(journal_path.with_name("return-upload-receipt.json"), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--spec", type=Path, required=True); p.add_argument("--authorization", type=Path, required=True); p.add_argument("--journal", type=Path, required=True); a = p.parse_args()
    try:
        print(json.dumps(run(a.spec, a.authorization, a.journal), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error_message": safe_message(exc), "credential_persisted": False}), flush=True); raise SystemExit(1)


if __name__ == "__main__":
    main()
