#!/usr/bin/env python3
"""Create one recipe-bound save-and-stop request for the CPT trainer."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from full_weight_cpt_trainer import load_bound, sha256


def request(recipe_path):
    recipe_path = Path(recipe_path).resolve()
    recipe = load_bound(recipe_path)
    destination = Path(recipe["outputs"]["graceful_stop"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"action": "save_and_stop", "bound_recipe_sha256": sha256(recipe_path)}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        try: destination.unlink()
        except FileNotFoundError: pass
        raise
    return {"status": "save_and_stop_requested", "path": str(destination), "recipe_sha256": payload["bound_recipe_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(request(args.recipe), sort_keys=True))


if __name__ == "__main__":
    main()
