"""Stage profile receipts, checkpoints, and approved logs for private upload.

The target cache and all framework/temp caches remain outside this staging
root.  A successful profile uses its hash-bound persistence manifest.  A
failed profile is still allowed to persist useful partial checkpoints and
terminal receipts, but only from explicitly approved roots.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

TRANSIENT_ROOTS = {
    "ephemeral-target-cache",
    "tmp",
    "hf-cache",
    "xdg-cache",
    "torch-cache",
    "triton-cache",
}
APPROVED_ROOTS = {"durable", "checkpoints", "stages", "logs", "tensorboard"}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _relative(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    require(not rel.is_absolute() and ".." not in rel.parts and str(rel) not in {"", "."}, "invalid staged artifact path")
    require(rel.parts[0] in APPROVED_ROOTS, "unapproved durable artifact root: " + rel.parts[0])
    require(not any(part in TRANSIENT_ROOTS for part in rel.parts), "transient artifact path cannot be staged")
    return rel


def _source_path(run: Path, rel: PurePosixPath) -> Path:
    source = run.joinpath(*rel.parts)
    require(source.resolve().is_relative_to(run.resolve()), "artifact path escaped profile run")
    chain = [source]
    parent = source.parent
    while parent != run and run in parent.parents:
        chain.append(parent)
        parent = parent.parent
    chain.append(run)
    require(not any(item.is_symlink() for item in chain), "artifact symlink is forbidden")
    require(source.is_file(), "profile artifact is missing: " + str(rel))
    return source


def _manifest_entries(run: Path) -> list[tuple[Path, PurePosixPath]]:
    manifest_path = run / "durable/persistence-manifest.json"
    terminal_path = run / "profile-terminal.json"
    require(manifest_path.is_file() and terminal_path.is_file(), "successful profile persistence manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    require(terminal.get("status") == "profile_passed", "profile terminal is not successful")
    selected: list[tuple[Path, PurePosixPath]] = []
    seen: set[str] = set()
    for row in manifest.get("files", []):
        require(isinstance(row, dict), "profile persistence entry is not an object")
        rel = _relative(str(row.get("path", "")))
        require(str(rel) not in seen, "duplicate profile persistence path")
        seen.add(str(rel))
        source = _source_path(run, rel)
        require(source.stat().st_size == int(row.get("bytes", -1)), "profile artifact size differs")
        require(digest(source) == str(row.get("sha256")), "profile artifact hash differs")
        selected.append((source, rel))
    require(selected, "successful profile has no durable artifacts")
    return selected


def _partial_entries(run: Path) -> list[tuple[Path, PurePosixPath]]:
    selected: list[tuple[Path, PurePosixPath]] = []
    for root_name in sorted(APPROVED_ROOTS):
        root = run / root_name
        if not root.is_dir():
            continue
        for source in sorted(root.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            rel = PurePosixPath(source.relative_to(run).as_posix())
            _relative(str(rel))
            _source_path(run, rel)
            selected.append((source, rel))
    return selected


def stage(run: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy verified success or useful partial-failure artifacts to a fresh root."""
    run_path = Path(run).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().resolve()
    require(not destination_path.exists(), "use a fresh durable staging directory")
    terminal_path = run_path / "profile-terminal.json"
    terminal: dict[str, Any] = {}
    if terminal_path.is_file():
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    success = terminal.get("status") == "profile_passed"
    entries = _manifest_entries(run_path) if success else _partial_entries(run_path)
    # Always include top-level profile and entry terminal receipts explicitly.
    extras: list[tuple[Path, PurePosixPath]] = []
    for name in (
        "profile-terminal.json",
        "entry-terminal.json",
        "entry-failure.json",
        "durable/persistence-manifest.json",
        "durable/sentinel-receipt.json",
    ):
        source = run_path / name
        if source.is_file():
            extras.append((source, PurePosixPath(name)))
    selected = entries + extras
    seen: set[str] = set()
    unique: list[tuple[Path, PurePosixPath]] = []
    for source, rel in selected:
        key = str(rel)
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, rel))
    require(unique, "no profile artifacts are available for durable upload")
    destination_path.mkdir(parents=True)
    for source, rel in unique:
        target = destination_path.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(digest(source) == digest(target), "durable staging readback differs: " + str(rel))
    result = {
        "schema": 1,
        "status": "durable_staged",
        "profile_status": terminal.get("status", "missing"),
        "partial_failure_allowed": not success,
        "source_run": str(run_path),
        "destination": str(destination_path),
        "files": [
            {"path": str(rel), "bytes": source.stat().st_size, "sha256": digest(source)}
            for source, rel in unique
        ],
        "cache_excluded": sorted(TRANSIENT_ROOTS),
        "checkpoints_included": any(str(rel).startswith("checkpoints/") for _, rel in unique),
    }
    (destination_path / "staging-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["staging_receipt"] = "staging-receipt.json"
    return result
