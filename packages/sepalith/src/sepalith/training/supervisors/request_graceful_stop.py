#!/usr/bin/env python3
"""Create one recipe-bound save-and-stop request for the SFT or CPT trainer."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

# Bound-recipe schema -> trainer module whose load_bound/sha256 validate it.
TRAINERS = {
    "sepalith.sft11.full-weight-edit-sft-eval-gate-bound.v1": "sepalith.training.sft.full_weight_edit_sft",
    "sepalith.sft11.full-weight-cpt-stage-transition-bound.v1": "sepalith.training.cpt.full_weight_cpt_trainer",
}


def trainer_for(recipe_path: str | os.PathLike[str]) -> ModuleType:
    schema = json.loads(Path(recipe_path).read_text()).get("schema")
    if schema not in TRAINERS:
        raise ValueError(f"no trainer accepts bound recipe schema: {schema}")
    return importlib.import_module(TRAINERS[schema])


def request(recipe_path: str | os.PathLike[str]) -> dict[str, str]:
    recipe_path = Path(recipe_path).resolve()
    trainer = trainer_for(recipe_path)
    recipe = trainer.load_bound(recipe_path)
    destination = Path(recipe["outputs"]["graceful_stop"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"action": "save_and_stop", "bound_recipe_sha256": trainer.sha256(recipe_path)}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"status": "save_and_stop_requested", "path": str(destination), "recipe_sha256": payload["bound_recipe_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(request(args.recipe), sort_keys=True))


if __name__ == "__main__":
    main()
