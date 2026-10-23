"""C15 checkpoint and continue records, without torch (the night driver and the evaluator use them too).

A checkpoint directory ``<archive>/full/checkpoint-<update>`` holds the model
(``model.safetensors``, config, generation config), the tokenizer files copied
from the starting model, ``optimizer.pt``, ``rng_state.pt`` and
``trainer-state.json`` (update count, pool-sampler cursor). Its
``campaign-manifest.json`` is written last and lists every file's size and
hash plus the recipe hash, so a partial or foreign checkpoint never verifies.

``continue-step-<update>.json`` in the evidence directory is the driver's R6
continue record for an evaluation checkpoint; the trainer refuses to train
past an evaluation step without it.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import cast

Json = dict[str, object]

CHECKPOINT_SCHEMA = "sepalith.c15.grpo-checkpoint.v1"
STATE_SCHEMA = "sepalith.c15.grpo-trainer-state.v1"
CONTINUE_SCHEMA = "sepalith.c15.grpo-continue.v1"


class StateError(RuntimeError):
    """A checkpoint or continue record does not verify."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise StateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path: Path) -> Json:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_checkpoint(directory: Path, recipe_sha: str) -> Json:
    """The manifest, every file's hash and the recipe binding; returns the trainer state."""
    manifest_path = directory / "campaign-manifest.json"
    require(manifest_path.is_file(), f"checkpoint manifest missing: {directory}")
    manifest = cast(Json, json.loads(manifest_path.read_text()))
    require(manifest.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema differs")
    require(manifest.get("recipe_sha256") == recipe_sha, "checkpoint belongs to another recipe")
    require(manifest.get("step") == int(directory.name.removeprefix("checkpoint-")), "checkpoint step differs from its name")
    files = cast(Json, manifest["files"])
    present = {path.name for path in directory.iterdir() if path.name != "campaign-manifest.json"}
    require(present == set(files), f"checkpoint files differ from the manifest: {sorted(present ^ set(files))}")
    for name, entry in files.items():
        require(file_entry(directory / name) == entry, f"checkpoint file differs: {name}")
    state = cast(Json, json.loads((directory / "trainer-state.json").read_text()))
    require(state.get("schema") == STATE_SCHEMA and state.get("update") == manifest["step"], "trainer state differs")
    return state


def continue_record_path(recipe: Mapping[str, object], step: int) -> Path:
    return Path(str(cast(Json, recipe["dev_gate"])["evidence_directory"])) / f"continue-step-{step}.json"


def continue_record(recipe_sha: str, step: int, checkpoint: Path, decision: Path) -> Json:
    return {"schema": CONTINUE_SCHEMA, "step": step, "recipe_sha256": recipe_sha,
            "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": sha256(checkpoint / "campaign-manifest.json"),
            "decision": {"path": str(decision), "sha256": sha256(decision)}}


def check_continue(recipe: Mapping[str, object], recipe_sha: str, checkpoint: Path, step: int) -> None:
    """Training past an R6 evaluation step needs the driver's continue record for that checkpoint."""
    path = continue_record_path(recipe, step)
    require(path.is_file(), f"no R6 continue record for evaluation step {step}: {path}")
    record = cast(Json, json.loads(path.read_text()))
    require(record.get("schema") == CONTINUE_SCHEMA and record.get("step") == step and record.get("recipe_sha256") == recipe_sha
            and record.get("checkpoint_manifest_sha256") == sha256(checkpoint / "campaign-manifest.json"),
            f"R6 continue record for step {step} does not bind this checkpoint")
