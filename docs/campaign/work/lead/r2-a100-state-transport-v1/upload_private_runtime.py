#!/usr/bin/env python3
"""Resumable private-HF upload of immutable cache/source/control files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transport_common import atomic_json, hashes, require, safe_message, sha256

ACTION = "upload_full_weight_runtime_bundle_private"


def lfs_sha(value):
    value = getattr(value, "lfs", None)
    return value.get("sha256") if isinstance(value, dict) else getattr(value, "sha256", None)


def remote_matches(api, spec, path, revision, expected):
    got = api.get_paths_info(spec["repo"], path, repo_type="model", revision=revision, expand=True)
    require(len(got) == 1 and getattr(got[0], "path", None) == path, "remote object absent")
    row = got[0]
    require(row.size == expected["bytes"], "remote object size differs")
    observed_lfs = lfs_sha(row)
    require(observed_lfs == expected["sha256"] if observed_lfs else getattr(row, "blob_id", None) == expected["git_blob_sha1"], "remote object hash differs")


def client():
    from huggingface_hub import HfApi, get_token, hf_hub_download
    token = get_token()
    require(bool(token), "cached private token unavailable")
    api = HfApi(token=token)
    require(api.repo_info("scholzmx/sepalith-lora", repo_type="model").private is True, "repository is not private")
    return api, hf_hub_download, token


def authorize(path, spec):
    observed = json.loads(Path(path).read_text())
    expected = {
        "schema": "sepalith.pre04.full_weight_runtime_transport_authorization.v1",
        "status": "admitted", "authorized": True, "action": ACTION,
        "repo": spec["repo"], "prefix": spec["prefix"],
        "cache_manifest_sha256": spec["cache_manifest_sha256"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
        "parent_manifest_sha256": spec["parent_manifest_sha256"],
        "maximum_bytes": spec["total_bytes"],
    }
    require(observed == expected, "root runtime authorization differs")


def run(spec_path, authorization_path, journal_path, *, api=None, download=None, token=None, stop_after=None):
    spec = json.loads(Path(spec_path).read_text())
    require(spec.get("schema") == "sepalith.pre04.full_weight_runtime_bundle_spec.v1" and spec.get("status") == "prepared_upload_not_authorized", "runtime spec differs")
    authorize(authorization_path, spec)
    if api is None:
        api, download, token = client()
    journal_path = Path(journal_path)
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"schema": "sepalith.pre04.runtime_upload_journal.v1", "spec_sha256": sha256(spec_path), "completed": {}}
    require(journal.get("schema") == "sepalith.pre04.runtime_upload_journal.v1" and journal.get("spec_sha256") == sha256(spec_path), "runtime journal differs")
    for index, (name, expected0) in enumerate(sorted(spec["files"].items())):
        local = Path(expected0["source_path"])
        before = local.stat()
        observed = hashes(local)
        expected = {**expected0, "git_blob_sha1": observed["git_blob_sha1"]}
        require(observed["bytes"] == expected["bytes"] and observed["sha256"] == expected["sha256"], "runtime local hash differs:" + name)
        remote = spec["prefix"] + "/files/" + name
        saved = journal["completed"].get(name)
        if saved:
            require(saved.get("remote_path") == remote and saved.get("expected") == expected, "runtime journal row differs:" + name)
            remote_matches(api, spec, remote, saved["revision"], expected)
        else:
            commit = api.upload_file(repo_id=spec["repo"], repo_type="model", path_or_fileobj=local, path_in_repo=remote, commit_message="Full-weight runtime bundle: " + name)
            remote_matches(api, spec, remote, commit.oid, expected)
            journal["completed"][name] = {"remote_path": remote, "revision": commit.oid, "expected": expected}
            atomic_json(journal_path, journal)
        after = local.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "runtime source changed:" + name)
        if stop_after is not None and index + 1 >= stop_after:
            raise RuntimeError("injected interruption")
    require(set(journal["completed"]) == set(spec["files"]), "runtime journal closure incomplete")
    closure = {
        "schema": "sepalith.pre04.full_weight_runtime_closure.v1", "status": "complete",
        "repo": spec["repo"], "prefix": spec["prefix"], "files": spec["files"],
        "total_bytes": spec["total_bytes"], "cache_manifest_sha256": spec["cache_manifest_sha256"],
        "source_manifest_sha256": spec["source_manifest_sha256"], "recipe_sha256": spec["recipe_sha256"],
        "parent_manifest_sha256": spec["parent_manifest_sha256"],
        "resume_world_size_required": 1, "credential_persisted": False,
    }
    closure_path = journal_path.with_name("runtime-closure.json")
    atomic_json(closure_path, closure)
    closure_sha = sha256(closure_path)
    remote = spec["prefix"] + "/closure.json"
    commit = api.upload_file(repo_id=spec["repo"], repo_type="model", path_or_fileobj=closure_path, path_in_repo=remote, commit_message="Publish full-weight runtime closure")
    back = Path(download(spec["repo"], filename=remote, revision=commit.oid, repo_type="model", token=token))
    require(sha256(back) == closure_sha, "runtime closure readback differs")
    receipt = {"schema": "sepalith.pre04.runtime_upload_receipt.v1", "status": "complete", "repo": spec["repo"], "prefix": spec["prefix"], "revision": commit.oid, "closure_sha256": closure_sha, "files": len(spec["files"]), "bytes": spec["total_bytes"], "credential_persisted": False}
    atomic_json(journal_path.with_name("runtime-upload-receipt.json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.spec, args.authorization, args.journal), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error_message": safe_message(exc), "credential_persisted": False}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
