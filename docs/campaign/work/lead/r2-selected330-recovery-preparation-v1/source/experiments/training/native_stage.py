#!/usr/bin/env python3
"""Copy admitted immutable inputs to a content-addressed native bundle.

Every payload byte is read once from the source while it is copied and hashed.
Publication is an atomic rename on the native filesystem.  A later launch must
use ``verify-run`` so native bytes are independently rehashed and held under
shared advisory locks for the child lifetime.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BLOCK = 8 * 1024 * 1024
MAX_NATIVE_BYTES = 70 * 1024**3
MIN_FREE_AFTER = 70 * 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(stat):
    return {key: int(getattr(stat, key)) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        flush_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def flush_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_relative(value):
    path = Path(value)
    require(not path.is_absolute() and value not in ("", "."), "payload path is not relative")
    require(".." not in path.parts, "payload path escapes object")
    return path


def open_regular(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    stat = os.fstat(fd)
    require(os.path.isfile(f"/proc/self/fd/{fd}"), f"source is not a regular file: {path}")
    return fd, stat


def copy_one(source, destination, expected_sha256, expected_bytes=None):
    source = Path(source); destination = Path(destination)
    require(len(expected_sha256) == 64, "expected SHA-256 differs")
    source_fd, before = open_regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    h = hashlib.sha256(); written = 0
    try:
        while True:
            block = os.read(source_fd, BLOCK)
            if not block: break
            h.update(block)
            view = memoryview(block)
            while view:
                count = os.write(out_fd, view)
                require(count > 0, "short destination write")
                view = view[count:]; written += count
        os.fsync(out_fd)
        after = os.fstat(source_fd)
        require(fingerprint(before) == fingerprint(after), f"source changed while copied: {source}")
        require(written == before.st_size and (expected_bytes is None or written == expected_bytes), f"source size differs: {source}")
        require(h.hexdigest() == expected_sha256, f"source hash differs: {source}")
    finally:
        os.close(out_fd); os.close(source_fd)
    os.chmod(destination, 0o400)
    return {"bytes": written, "sha256": expected_sha256, "source_fingerprint": fingerprint(before), "staged_fingerprint": fingerprint(destination.stat())}


def object_files(record):
    kind = record.get("kind")
    source = Path(record["source"])
    if kind == "file":
        return [(Path(record.get("destination_name", source.name)), source, record["sha256"], record.get("bytes"))], record["sha256"]
    require(kind == "manifest_tree", "unsupported stage object kind")
    manifest_record = record["manifest"]
    manifest_path = Path(manifest_record["path"])
    manifest_bytes = manifest_path.read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == manifest_record["sha256"], "tree manifest differs")
    manifest = json.loads(manifest_bytes)
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "tree manifest file inventory is empty")
    manifest_name = safe_relative(record.get("manifest_name", manifest_path.name))
    if manifest_path.parent.resolve() == source.resolve():
        actual = set()
        for path in source.rglob("*"):
            require(not path.is_symlink(), f"tree source contains symlink: {path}")
            if path.is_file(): actual.add(str(path.relative_to(source)))
        require(actual == set(files) | {str(manifest_name)}, "tree source file set differs from manifest closure")
    result = [(manifest_name, manifest_path, manifest_record["sha256"], manifest_path.stat().st_size)]
    for name, expected in sorted(files.items()):
        relative = safe_relative(name)
        require(type(expected.get("bytes")) is int and expected["bytes"] > 0, "manifest byte count differs")
        result.append((relative, source / relative, expected["sha256"], expected["bytes"]))
    return result, manifest_record["sha256"]


def normalized_spec(spec):
    require(spec.get("schema") == "sepalith.sft11.native-stage-spec.v1", "stage spec schema differs")
    native_root = Path(spec["native_root"])
    require(native_root.is_absolute() and str(native_root).startswith("/home/"), "native root must be an absolute native Linux path")
    max_bytes = spec.get("max_native_bytes")
    min_free = spec.get("min_free_after_bytes")
    require(type(max_bytes) is int and 0 < max_bytes <= MAX_NATIVE_BYTES, "native usage cap exceeds 70 GiB")
    require(type(min_free) is int and min_free >= MIN_FREE_AFTER, "native free-space floor is below 70 GiB")
    objects = spec.get("objects")
    require(isinstance(objects, list) and objects, "stage objects are empty")
    names = [item.get("name") for item in objects]
    require(all(isinstance(name, str) and name for name in names) and len(set(names)) == len(names), "stage object names differ")
    return native_root, max_bytes, min_free, objects


def build(spec_path, receipt_path):
    spec_path = Path(spec_path); spec = json.loads(spec_path.read_text(encoding="utf-8"))
    native_root, max_bytes, min_free, objects = normalized_spec(spec)
    expanded = []
    for record in objects:
        files, manifest_identity = object_files(record)
        expanded.append((record, files, manifest_identity))
    total = sum(expected_bytes for _, files, _ in expanded for _, _, _, expected_bytes in files if expected_bytes is not None)
    require(total <= max_bytes, "declared stage bytes exceed native budget")
    native_root.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(native_root).free - total >= min_free, "native free-space floor would be violated")
    identity = {"schema": spec["schema"], "objects": [{"name": r["name"], "kind": r["kind"], "identity_sha256": ident} for r, _, ident in expanded]}
    bundle_id = hashlib.sha256(canonical(identity)).hexdigest()
    destination = native_root / bundle_id
    require(not destination.exists(), "content-addressed destination already exists; use verify-run on its receipt")
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=native_root))
    records = []
    try:
        for record, files, identity_sha in expanded:
            object_root = temporary / "objects" / record["name"]
            copied = {}
            for relative, source, expected_sha, expected_bytes in files:
                copied[str(relative)] = copy_one(source, object_root / relative, expected_sha, expected_bytes)
            records.append({"name": record["name"], "kind": record["kind"], "canonical_source": str(Path(record["source"]).resolve()), "staged_path": str((destination / "objects" / record["name"]).resolve()), "identity_sha256": identity_sha, "files": copied})
        closure = {"schema": "sepalith.sft11.native-stage-closure.v1", "bundle_id": bundle_id, "spec_sha256": digest(spec_path), "total_bytes": total, "objects": records}
        write_json(temporary / "closure.json", closure)
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_dir(): os.chmod(path, 0o500)
        flush_dir(temporary)
        os.rename(temporary, destination); flush_dir(native_root)
        receipt = {"schema": "sepalith.sft11.native-stage-receipt.v1", "status": "staged_immutable_verified", "bundle_id": bundle_id, "bundle_root": str(destination.resolve()), "closure_sha256": digest(destination / "closure.json"), "spec_path": str(spec_path.resolve()), "spec_sha256": digest(spec_path), "total_bytes": total, "native_free_after_bytes": shutil.disk_usage(native_root).free, "objects": records}
        write_json(receipt_path, receipt)
        return receipt
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def verify_and_lock(receipt_path, expected_receipt_sha256):
    receipt_path = Path(receipt_path)
    require(digest(receipt_path) == expected_receipt_sha256, "stage receipt differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == "sepalith.sft11.native-stage-receipt.v1" and receipt.get("status") == "staged_immutable_verified", "stage receipt status differs")
    root = Path(receipt["bundle_root"])
    require(root.is_dir() and not root.is_symlink(), "native bundle missing")
    require(digest(root / "closure.json") == receipt["closure_sha256"], "native closure differs")
    held = []
    try:
        for obj in receipt["objects"]:
            object_root = root / "objects" / obj["name"]
            require(object_root.resolve() == Path(obj["staged_path"]), "staged object path differs")
            for relative, expected in obj["files"].items():
                path = object_root / safe_relative(relative)
                fd, before = open_regular(path)
                fcntl.flock(fd, fcntl.LOCK_SH)
                h = hashlib.sha256(); size = 0
                while True:
                    block = os.read(fd, BLOCK)
                    if not block: break
                    h.update(block); size += len(block)
                after = os.fstat(fd)
                require(fingerprint(before) == fingerprint(after), f"native input changed during verification: {path}")
                require(size == expected["bytes"] and h.hexdigest() == expected["sha256"], f"native input bytes differ: {path}")
                held.append(fd)
        return receipt, held
    except Exception:
        for fd in held: os.close(fd)
        raise


def verify_run(receipt_path, receipt_sha256, command):
    require(command, "verified command is empty")
    receipt, held = verify_and_lock(receipt_path, receipt_sha256)
    try:
        env = dict(os.environ); env["SEPALITH_NATIVE_STAGE_BUNDLE_ID"] = receipt["bundle_id"]
        records=[];index=0
        for obj in receipt['objects']:
            root=Path(obj['staged_path'])
            for relative,expected in obj['files'].items():
                records.append({'path':str((root/safe_relative(relative)).resolve()),'fd':held[index],'bytes':expected['bytes'],'sha256':expected['sha256'],'fingerprint':expected['staged_fingerprint']});index+=1
        env['SEPALITH_NATIVE_STAGE_ATTESTATION']=json.dumps({'receipt_sha256':receipt_sha256,'bundle_id':receipt['bundle_id'],'files':records},sort_keys=True,separators=(',',':'))
        result = subprocess.run(command, env=env, pass_fds=tuple(held), check=False)
        return result.returncode
    finally:
        for fd in held: os.close(fd)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build"); build_parser.add_argument("--spec", required=True); build_parser.add_argument("--receipt", required=True)
    run_parser = sub.add_parser("verify-run"); run_parser.add_argument("--receipt", required=True); run_parser.add_argument("--receipt-sha256", required=True); run_parser.add_argument("child", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "build": print(json.dumps(build(args.spec, args.receipt), sort_keys=True)); return 0
    child = args.child[1:] if args.child and args.child[0] == "--" else args.child
    return verify_run(args.receipt, args.receipt_sha256, child)


if __name__ == "__main__": raise SystemExit(main())
