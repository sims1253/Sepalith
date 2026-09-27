"""Bounded target/public artifact staging for an admitted cloud run.

The payload carries only small metadata, source, and TRAIN files.  This module
downloads the two pinned model sets directly into the fresh run directory,
relocates the metadata-only target manifest without changing its model
identity fields, and records hashes without recording the private token.  The
provider client is imported only by the admitted download subprocess.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping


PRIVATE_TARGET_REPO = "scholzmx/sepalith-lora"
PRIVATE_TARGET_REVISION = "257487b64044600fee8c27cbcb1ceb19e424a9ac"
PRIVATE_TARGET_PREFIX = "r2-draft-target/b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4"
PUBLIC_DRAFT_REPO = "openbmb/MiniCPM5-2B-DSpark"
PUBLIC_DRAFT_REVISION = "114a20fdbf53220712c7fbdd7dccddbf1dedebb4"
TARGET_FILES = {
    "model.safetensors": "b862986475d8b7f9dd74639e53c2b79b7d85abe30e79af5763efdcc9a1ed6fc4",
    "config.json": "f1b9bfce12195f72a1a64847dfb6c97adba5200a16f3dd6cf1b4075755851991",
    "generation_config.json": "7fd42fdf451ae26258ea1d30a6efa4f1871642110b208e8a4a631c77ad9dc269",
    "tokenizer.json": "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config.json": "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}
PUBLIC_FILES = {
    "model.safetensors": "ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97",
    "config.json": "bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b",
}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_run_path(run: Path, value: str) -> Path:
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts and relative.parts not in ((), (".",)), "download destination must be run-relative")
    result = run / relative
    require(result.resolve().is_relative_to(run.resolve()), "download destination escaped run")
    require(not any(part.is_symlink() for part in (result, *result.parents) if part != run.parent), "download destination contains a symlink")
    return result


def validate_download_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    downloads = binding.get("downloads")
    require(isinstance(downloads, Mapping), "explicit model download binding is missing")
    target = downloads.get("target")
    public = downloads.get("public")
    require(isinstance(target, Mapping) and isinstance(public, Mapping), "target/public download bindings are incomplete")
    require(target.get("repo_id") == PRIVATE_TARGET_REPO and target.get("revision") == PRIVATE_TARGET_REVISION, "private target repository or immutable revision differs")
    require(target.get("prefix") == PRIVATE_TARGET_PREFIX, "private target prefix differs")
    require(target.get("destination") == "inputs/target-model", "private target destination differs")
    require(public.get("repo_id") == PUBLIC_DRAFT_REPO and public.get("revision") == PUBLIC_DRAFT_REVISION, "public draft repository or revision differs")
    require(public.get("destination") == "inputs/public-draft", "public draft destination differs")
    require(set(target.get("files", {})) == set(TARGET_FILES), "private target file set differs")
    require(set(public.get("files", {})) == set(PUBLIC_FILES), "public draft file set differs")
    require(all(target["files"][name] == digest for name, digest in TARGET_FILES.items()), "private target file pins differ")
    require(all(public["files"][name] == digest for name, digest in PUBLIC_FILES.items()), "public draft file pins differ")
    manifest_destination = downloads.get("manifest_destination", "inputs/target-manifest.json")
    require(manifest_destination == "inputs/target-manifest.json", "relocated target manifest destination differs")
    return {
        "target": dict(target),
        "public": dict(public),
        "manifest_destination": manifest_destination,
    }


def _materialize(downloaded: str | Path, destination: Path, expected_sha256: str, run: Path) -> dict[str, Any]:
    source = Path(downloaded).resolve(strict=True)
    require(source.is_file() and not Path(downloaded).is_symlink(), "download result is missing or symlinked")
    require(source.is_relative_to(run.resolve()), "download result escaped run-owned staging")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(source == destination.resolve() and not destination.is_symlink(), "download destination already exists; implicit resume is forbidden")
    else:
        source.replace(destination)
    require(destination.is_file() and not destination.is_symlink(), "downloaded artifact was not materialized")
    digest = sha256_file(destination)
    require(digest == expected_sha256, "downloaded artifact SHA-256 differs: " + destination.name)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": digest}


def stage_downloads(
    binding: Mapping[str, Any],
    *,
    payload_root: str | Path,
    run: str | Path,
    download: Callable[..., str],
    token: str | None = None,
) -> dict[str, Any]:
    """Download and hash exact artifacts using an injected client in tests."""
    payload_root = Path(payload_root).expanduser().resolve(strict=True)
    run = Path(run).expanduser().resolve(strict=True)
    spec = validate_download_binding(binding)
    staging = binding["staging"]
    source_manifest = payload_root / Path(str(staging["target_manifest"]))
    require(source_manifest.is_file() and not source_manifest.is_symlink(), "source target manifest is missing")
    source_manifest_sha256 = sha256_file(source_manifest)
    require(source_manifest_sha256 == str(binding["pins"]["target_manifest_sha256"]), "source target manifest SHA-256 differs")
    try:
        record = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("source target manifest is invalid JSON") from exc
    require(isinstance(record, Mapping), "source target manifest must be an object")
    require(record.get("weights_file") == "model.safetensors" and record.get("config_file", "config.json") == "config.json", "source target manifest filenames are not canonical")
    require(record.get("weights_sha256") == TARGET_FILES["model.safetensors"] and record.get("config_sha256") == TARGET_FILES["config.json"], "source target manifest tensor pins differ")
    require(record.get("deepspec_revision") == binding["pins"]["deepspec_revision"], "source target manifest revision differs")
    target_dir = _relative_run_path(run, str(spec["target"]["destination"]))
    public_dir = _relative_run_path(run, str(spec["public"]["destination"]))
    manifest_path = _relative_run_path(run, str(spec["manifest_destination"]))
    require(not target_dir.exists() and not public_dir.exists() and not manifest_path.exists(), "download staging paths must be fresh")
    target_rows = []
    for name, expected in TARGET_FILES.items():
        remote_name = str(spec["target"]["prefix"]).rstrip("/") + "/" + name
        downloaded = download(
            str(spec["target"]["repo_id"]),
            filename=remote_name,
            revision=str(spec["target"]["revision"]),
            local_dir=target_dir.parent,
            cache_dir=run / "download-cache",
            token=token,
        )
        target_rows.append({"repo_id": PRIVATE_TARGET_REPO, "revision": PRIVATE_TARGET_REVISION, "filename": remote_name, **_materialize(downloaded, target_dir / name, expected, run)})
    public_rows = []
    for name, expected in PUBLIC_FILES.items():
        downloaded = download(
            str(spec["public"]["repo_id"]),
            filename=name,
            revision=str(spec["public"]["revision"]),
            local_dir=public_dir,
            cache_dir=run / "download-cache",
            token=False,
        )
        public_rows.append({"repo_id": PUBLIC_DRAFT_REPO, "revision": PUBLIC_DRAFT_REVISION, "filename": name, **_materialize(downloaded, public_dir / name, expected, run)})
    effective = dict(record)
    # The shipped manifest may contain a workstation absolute path.  Preserve
    # its hash and all model identity fields while relocating only the path to
    # the fresh run's sibling target-model directory.
    effective["model_dir"] = target_dir.name
    effective["source_manifest_sha256"] = source_manifest_sha256
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema": 1,
        "status": "model_downloads_staged",
        "private_target": {"repo_id": PRIVATE_TARGET_REPO, "revision": PRIVATE_TARGET_REVISION, "prefix": PRIVATE_TARGET_PREFIX, "files": target_rows, "token_used": bool(token)},
        "public_draft": {"repo_id": PUBLIC_DRAFT_REPO, "revision": PUBLIC_DRAFT_REVISION, "files": public_rows, "token_used": False},
        "source_manifest": {"path": str(source_manifest), "sha256": source_manifest_sha256},
        "effective_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "model_dir": str(target_dir)},
        "cache_root": str(run / "download-cache"),
        "token_persisted": False,
    }
    (run / "staging-download-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def stage_from_environment(binding_path: str | Path, *, payload_root: str | Path, run: str | Path) -> dict[str, Any]:
    """Run the networked downloader in its own admitted subprocess."""
    token = os.environ.get("HF_TOKEN")
    require(token, "private target download token absent")
    from huggingface_hub import hf_hub_download

    def download(repo_id: str, *, filename: str, revision: str, local_dir: Path, cache_dir: Path, token: str | bool | None) -> str:
        return hf_hub_download(
            repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir),
            token=token,
        )

    try:
        binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
        return stage_downloads(binding, payload_root=payload_root, run=run, download=download, token=token)
    finally:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("binding", type=Path)
    parser.add_argument("payload_root", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(stage_from_environment(args.binding, payload_root=args.payload_root, run=args.run), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "model_download_failed", "error_type": type(exc).__name__, "reason": str(exc)[:240]}))
        raise SystemExit(1)
