#!/usr/bin/env python3
"""Download and byte-verify the pinned private CPT checkpoint tar.

The checkpoint's pickle files are intentionally opaque.  This program never
imports torch or loads model/state tensors; it reads the safetensors JSON
header only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import posixpath
import shutil
import struct
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

REPO = "scholzmx/sepalith-lora"
REVISION = 'ee35e423e9c943ad3acb4bfd7223c0789fb03157'
REMOTE_PATH = "r2-cpt/4f899bbc6e9d46c0a88985d64e6d40e2/checkpoint-tars/checkpoint-1902.tar"
TAR_BYTES = 613888000
TAR_SHA256 = 'bd04572e6d419941aebf7edc8de8b3ab136730c1542757a998edb49ef99d9b8c'
STEP = 1902
CONSUMED_DRAWS = 30_432
PREFIX = f"checkpoint-{STEP}/"
CHUNK = 4 * 1024 * 1024


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and posixpath.normpath(name) == name
        and name.startswith(PREFIX)
        and len(path.parts) == 2
    )


def validate_member_table(tar_path: Path, expected: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    wanted = {PREFIX + name: row for name, row in expected.items()}
    seen: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            require(member.isfile(), f"non-regular tar member: {member.name!r}")
            require(safe_member_name(member.name), f"unsafe tar member: {member.name!r}")
            require(member.name in wanted, f"unexpected tar member: {member.name!r}")
            require(member.name not in seen, f"duplicate tar member: {member.name!r}")
            expected_row = wanted[member.name]
            require(member.size == expected_row["bytes"], f"tar member size differs: {member.name}")
            row = {"name": member.name, "bytes": member.size, "mode": member.mode}
            seen[member.name] = row
            records.append(row)
    require(set(seen) == set(wanted), "tar member set differs from pinned receipt")
    return sorted(records, key=lambda row: str(row["name"]))


def extract_and_verify(tar_path: Path, destination: Path, expected: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=False)
    verified: dict[str, dict[str, object]] = {}
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            target = destination / name
            source = archive.extractfile(member)
            require(source is not None, f"tar member unreadable: {member.name}")
            digest = hashlib.sha256()
            count = 0
            with target.open("xb") as output:
                for block in iter(lambda: source.read(CHUNK), b""):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
                output.flush()
                os.fsync(output.fileno())
            actual = {"bytes": count, "sha256": digest.hexdigest()}
            require(actual == expected[name], f"extracted member bytes/hash differs: {name}")
            os.chmod(target, member.mode & 0o777)
            verified[name] = actual
    directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    require(set(verified) == set(expected), "extracted file set differs")
    return dict(sorted(verified.items()))


def safetensors_header(path: Path) -> dict[str, object]:
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "short safetensors length prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        require(2 <= header_bytes <= min(file_bytes - 8, 64 * 1024 * 1024), "invalid safetensors header length")
        raw = stream.read(header_bytes)
        require(len(raw) == header_bytes, "short safetensors header")
    header = json.loads(raw)
    require(isinstance(header, dict), "safetensors header is not an object")
    tensors = []
    intervals = []
    dtype_counts: Counter[str] = Counter()
    element_count = 0
    for name, row in header.items():
        if name == "__metadata__":
            continue
        require(isinstance(name, str) and name, "invalid tensor name")
        require(isinstance(row, dict), f"invalid tensor row: {name}")
        dtype, shape, offsets = row.get("dtype"), row.get("shape"), row.get("data_offsets")
        require(isinstance(dtype, str) and dtype, f"invalid tensor dtype: {name}")
        require(isinstance(shape, list) and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape), f"invalid tensor shape: {name}")
        require(isinstance(offsets, list) and len(offsets) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in offsets), f"invalid tensor offsets: {name}")
        start, end = offsets
        require(0 <= start <= end <= file_bytes - 8 - header_bytes, f"out-of-range tensor offsets: {name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        element_count += elements
        dtype_counts[dtype] += 1
        intervals.append((start, end, name))
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "data_offsets": offsets, "elements": elements})
    intervals.sort()
    cursor = 0
    for start, end, name in intervals:
        require(start == cursor, f"non-contiguous or overlapping tensor payload before {name}")
        cursor = end
    require(cursor == file_bytes - 8 - header_bytes, "tensor offsets do not span safetensors payload")
    return {
        "schema": "sepalith.safetensors-header-inventory.v1",
        "path": path.name,
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "payload_bytes": cursor,
        "tensor_count": len(tensors),
        "element_count": element_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "metadata": header.get("__metadata__"),
        "tensors": sorted(tensors, key=lambda row: str(row["name"])),
        "inspection": "first 8 bytes and JSON header only; tensor payload not loaded",
    }


def verify_identities(extracted: Path, root317: Path) -> dict[str, object]:
    manifest = json.loads((extracted / "campaign-manifest.json").read_bytes())
    state = json.loads((extracted / "campaign-state.json").read_bytes())
    trainer = json.loads((extracted / "trainer_state.json").read_bytes())
    prior_state = json.loads((root317 / "campaign-state.json").read_bytes())
    require(manifest.get("full") is True and manifest.get("step") == STEP, "manifest full/step differs")
    require(state.get("full") is True and state.get("step") == STEP, "campaign state full/step differs")
    require(trainer.get("global_step") == STEP, "trainer global_step differs")
    require(trainer.get("max_steps") == 1902, "trainer max_steps differs")
    sampler = state.get("sampler")
    require(isinstance(sampler, dict), "sampler absent")
    require(sampler.get("consumed_draws") == CONSUMED_DRAWS == STEP * 16, "consumed draw cursor differs")
    require(sampler.get("schedule_sha256") == "914cf353199f45697c028da3a6f06b732fd57bd265869890b75bf43151851b58", "schedule identity differs")
    require(state.get("identity") == manifest.get("identity"), "manifest/state identity differs")
    require(state.get("identity") == prior_state.get("identity"), "step-317 predecessor identity differs")
    identity = state["identity"]
    return {
        "step": STEP,
        "full": True,
        "trainer_global_step": trainer["global_step"],
        "trainer_max_steps": trainer["max_steps"],
        "consumed_draws": sampler["consumed_draws"],
        "schedule_sha256": sampler["schedule_sha256"],
        "split_id": sampler["split_id"],
        "source_identity": identity["source"],
        "parent": identity["parent"],
        "renderer": identity["renderer"],
        "tokenizer": identity["tokenizer"],
        "policy": identity["policy"],
        "data_core": {key: identity["data"][key] for key in ("train_rows_sha256", "unique_train_rows", "scheduled_draws", "explicit_replay_rows", "draw_schedule_sha256", "split_id")},
        "entire_identity_equal_to_step317": True,
        "state_files": ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin", "campaign-state.json"],
        "opaque_pickle_files_not_loaded": ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"],
    }


def download(target: Path) -> dict[str, object]:
    from huggingface_hub import get_token, hf_hub_url
    import requests

    token = get_token()
    require(bool(token), "cached Hugging Face authentication unavailable")
    url = hf_hub_url(REPO, REMOTE_PATH, revision=REVISION)
    start = time.monotonic()
    digest = hashlib.sha256()
    count = 0
    with requests.get(url, headers={"Authorization": "Bearer " + token}, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with target.open("xb") as output:
            for block in response.iter_content(chunk_size=CHUNK):
                if not block:
                    continue
                output.write(block)
                digest.update(block)
                count += len(block)
            output.flush()
            os.fsync(output.fileno())
    elapsed = time.monotonic() - start
    require(count == TAR_BYTES, "downloaded tar byte count differs")
    require(digest.hexdigest() == TAR_SHA256, "downloaded tar SHA-256 differs")
    return {"bytes": count, "sha256": digest.hexdigest(), "elapsed_seconds": elapsed, "revision": REVISION, "remote_path": REMOTE_PATH}


def run(metadata_receipt: Path, root317: Path, destination: Path) -> dict[str, object]:
    require(not destination.exists(), "destination already exists")
    metadata = json.loads(metadata_receipt.read_bytes())
    remote = metadata["remote_receipt"]
    require(metadata.get("revision") == REVISION, "Hub revision differs")
    require(remote.get("remote_path") == REMOTE_PATH, "remote path differs")
    require(remote.get("tar_bytes") == TAR_BYTES and remote.get("tar_sha256") == TAR_SHA256, "tar pin differs")
    require(remote.get("step") == STEP and remote.get("consumed_draws") == CONSUMED_DRAWS, "remote cursor differs")
    expected = remote.get("files")
    require(isinstance(expected, dict) and len(expected) == 13, "expected inventory differs")

    partial = destination.with_name("." + destination.name + f".partial-{os.getpid()}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        archive_dir = partial / "archive"
        archive_dir.mkdir()
        tar_path = archive_dir / f"checkpoint-{STEP}.tar"
        download_evidence = download(tar_path)
        table = validate_member_table(tar_path, expected)
        verified = extract_and_verify(tar_path, partial / "extracted", expected)
        manifest = json.loads((partial / "extracted" / "campaign-manifest.json").read_bytes())
        require(set(manifest.get("files", {})) | {"campaign-manifest.json"} == set(expected), "original manifest file set differs")
        for name, row in manifest["files"].items():
            require(row == expected[name], f"original manifest inventory differs: {name}")
        identities = verify_identities(partial / "extracted", root317)
        tensor_inventory = safetensors_header(partial / "extracted" / "adapter_model.safetensors")
        write_json(partial / "tensor-header-inventory.json", tensor_inventory)
        evidence = {
            "schema": "sepalith.cloud-cpt.candidate-readback.v1",
            "status": "verified",
            "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repository": REPO,
            "private_repository": True,
            "hub_revision": REVISION,
            "run_id": remote["run_id"],
            "download": download_evidence,
            "archive_member_count": len(table),
            "archive_members": table,
            "verified_files": verified,
            "identities": identities,
            "tensor_header_summary": {key: tensor_inventory[key] for key in ("file_bytes", "header_bytes", "payload_bytes", "tensor_count", "element_count", "dtype_counts", "inspection")},
            "security": {"path_traversal_rejected": True, "links_rejected": True, "non_regular_members_rejected": True, "unexpected_and_duplicate_members_rejected": True, "credential_persisted": False},
            "preservation": {"campaign_manifest_rewritten": False, "tokenizer_files_rewritten": False, "checkpoint_pickle_loaded": False},
            "selection": "root review required; checkpoint-1902 comparison pending",
        }
        write_json(partial / "readback-evidence.json", evidence)
        os.replace(partial, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return evidence
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-receipt", type=Path, required=True)
    parser.add_argument("--root317", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    evidence = run(args.metadata_receipt, args.root317, args.destination)
    print(json.dumps({"status": evidence["status"], "step": STEP, "tar_sha256": evidence["download"]["sha256"], "files": len(evidence["verified_files"]), "tensor_count": evidence["tensor_header_summary"]["tensor_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
