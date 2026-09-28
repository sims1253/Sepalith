"""Private sentinel and durable-only artifact upload interface.

The network client is imported only when this module is executed for an
admitted cloud run.  CPU tests inject a fake client and exercise hash and
failure behavior without credentials or network access.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any, Callable, Mapping

TRANSIENT_NAMES = {
    "ephemeral-target-cache",
    "tmp",
    "hf-cache",
    "xdg-cache",
    "torch-cache",
    "triton-cache",
}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashes(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    size = path.stat().st_size
    sha256 = digest(path)
    git = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            git.update(block)
    return {"bytes": size, "sha256": sha256, "git_blob_sha1": git.hexdigest()}


def build_local_manifest(root: str | Path) -> dict[str, dict[str, Any]]:
    """Hash only the caller-provided durable staging root."""
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), "durable upload root is not a directory")
    local: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlink forbidden")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in TRANSIENT_NAMES for part in Path(relative).parts):
            raise ValueError("transient cache entered durable upload root: " + relative)
        require(relative not in local, "duplicate durable upload path")
        local[relative] = hashes(path)
    require(local, "durable upload root is empty")
    return local


def validate_remote(local: Mapping[str, Mapping[str, Any]], remote: Mapping[str, Mapping[str, Any]]) -> None:
    require(set(local) == set(remote), "remote durable artifact set differs")
    for name, expected in local.items():
        actual = remote[name]
        require(int(actual.get("bytes", -1)) == int(expected["bytes"]), "remote artifact size differs: " + name)
        remote_sha = actual.get("sha256")
        if remote_sha:
            require(remote_sha == expected["sha256"], "remote artifact hash differs: " + name)
        else:
            require(actual.get("git_blob_sha1") == expected.get("git_blob_sha1"), "remote artifact hash differs: " + name)


def _repo_private(api: Any, repo: str) -> None:
    info = api.repo_info(repo_id=repo, repo_type="model")
    require(bool(getattr(info, "private", False)), "artifact repository must be private")


def upload_sentinel(
    run: str | Path,
    binding: Mapping[str, Any],
    api: Any,
    download: Callable[..., str],
) -> dict[str, Any]:
    """Prove one fresh private prefix before profile work begins."""
    run = Path(run).resolve()
    repo = str(binding["artifact_repo"])
    prefix = str(binding["artifact_prefix"])
    _repo_private(api, repo)
    require(not api.file_exists(repo_id=repo, filename=prefix + "/sentinel.json", repo_type="model"), "artifact prefix already used")
    path = run / "sentinel.json"
    path.write_text(json.dumps({"run_id": binding["run_id"], "nonce": secrets.token_hex(32)}, sort_keys=True) + "\n", encoding="utf-8")
    commit = api.upload_file(
        path_or_fileobj=path,
        path_in_repo=prefix + "/sentinel.json",
        repo_id=repo,
        repo_type="model",
        commit_message="R2 private draft-profile sentinel",
    )
    downloaded = download(
        repo,
        filename=prefix + "/sentinel.json",
        revision=commit.oid,
        local_dir=run / "sentinel-readback",
    )
    require(digest(downloaded) == digest(path), "private sentinel readback differs")
    result = {
        "schema": 1,
        "status": "private_sentinel_verified",
        "commit": commit.oid,
        "sha256": digest(path),
        "repository_private": True,
        "token_persisted": False,
    }
    (run / "sentinel-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def upload_durable(
    run: str | Path,
    binding: Mapping[str, Any],
    api: Any,
    download: Callable[..., str],
    *,
    mode: str,
) -> dict[str, Any]:
    """Upload only a staged durable root, including success or failure output."""
    require(mode in {"success", "failure"}, "durable upload mode is invalid")
    run = Path(run).resolve()
    durable = run / "durable-upload"
    _repo_private(api, str(binding["artifact_repo"]))
    local = build_local_manifest(durable)
    manifest_path = durable / "upload-manifest.json"
    manifest_path.write_text(json.dumps({"schema": 1, "mode": mode, "files": local}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    local["upload-manifest.json"] = hashes(manifest_path)
    repo = str(binding["artifact_repo"])
    prefix = str(binding["artifact_prefix"]) + "/" + mode
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=durable,
        path_in_repo=prefix,
        commit_message="R2 durable draft-profile " + mode,
    )
    remote: dict[str, dict[str, Any]] = {}
    for row in api.list_repo_tree(repo_id=repo, repo_type="model", path_in_repo=prefix, revision=commit.oid, recursive=True, expand=True):
        name = getattr(row, "path", "").removeprefix(prefix + "/")
        if not name or not hasattr(row, "size"):
            continue
        lfs = getattr(row, "lfs", None)
        remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        remote[name] = {"bytes": int(row.size), "sha256": remote_sha, "git_blob_sha1": getattr(row, "blob_id", None)}
    validate_remote(local, remote)
    receipt = {
        "schema": 1,
        "status": "durable_upload_verified",
        "mode": mode,
        "repository": repo,
        "prefix": prefix,
        "commit": commit.oid,
        "files": local,
        "remote_files_verified": len(local),
        "token_persisted": False,
        "independent_root_readback_required": True,
    }
    receipt_path = run / "persistence-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final = api.upload_file(
        path_or_fileobj=receipt_path,
        path_in_repo=prefix + "/persistence-receipt.json",
        repo_id=repo,
        repo_type="model",
        commit_message="R2 durable profile receipt",
    )
    downloaded = download(
        repo,
        filename=prefix + "/persistence-receipt.json",
        revision=final.oid,
        local_dir=run / "receipt-readback",
    )
    require(digest(downloaded) == digest(receipt_path), "durable persistence receipt readback differs")
    receipt["receipt_commit"] = final.oid
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def upload_from_environment(run: str | Path, *, mode: str) -> dict[str, Any]:
    """Use the provider client only for an admitted upload subprocess."""
    token = os.environ.get("HF_TOKEN")
    require(bool(token), "private upload token absent")
    from huggingface_hub import HfApi, hf_hub_download

    run = Path(run).resolve()
    binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
    api = HfApi(token=token)
    def download_private(repo_id: str, *, filename: str, revision: str, local_dir: Path) -> str:
        return hf_hub_download(
            repo_id,
            filename=filename,
            revision=revision,
            token=token,
            local_dir=local_dir,
        )

    return upload_sentinel(run, binding, api, download_private) if mode == "sentinel" else upload_durable(run, binding, api, download_private, mode=mode)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sentinel", "success", "failure"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = upload_from_environment(args.run, mode=args.mode)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never print environment values or credentials.  The entrypoint owns
        # the terminal failure receipt and decides whether this is fatal.
        print(json.dumps({"status": "artifact_operation_failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
