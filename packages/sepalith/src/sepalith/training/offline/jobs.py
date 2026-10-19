#!/usr/bin/env python3
"""Night-queue job files for the C13 offline steps (written, never enqueued here).

For a bound RFT recipe: one resumable job ``C13-<step>-<parent tag>`` running
``offline.night run``. For a bound DPO recipe: ``C13-<step>-<parent tag>-reference``
(``offline.night reference``) and the training job, which requires it.
Enqueue with ``python3 -m sepalith.ops.night_queue enqueue --file <job>.json``.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import cast

from sepalith.training.sft.recipe_v2 import Json, PathMap, write_new

PYTHON = ".venv-sft/bin/python"
PRIORITY = {"rft": 50, "dpo": 55}


def command(root: Path, *args: str) -> list[str]:
    return ["/usr/bin/env", f"PYTHONPATH={root}/packages/sepalith/src", "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=0",
            "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", str(root / PYTHON), "-B", "-m", "sepalith.training.offline.night", *args]


def minutes(recipe: Json, kind: str) -> dict[str, int]:
    """Planning estimates from arm A's measured rate (about 1,030 trained tokens/s, gates 10 to 13 minutes)."""
    cohort = cast(Json, recipe["cohort"])
    if kind == "rft":
        manifest = json.loads(Path(cast(str, cast(Json, cohort["manifest"])["path"])).read_text())
        tokens = manifest["schedule"]["token_denominators"]["scheduled"]["input_tokens"]
        return {"train": round(tokens / 1030 / 60) + 26}
    tokens = cast(Json, cohort["tokens"])
    trained = 2 * cast(int, tokens["prompt"]) + cast(int, tokens["chosen"]) + cast(int, tokens["rejected"])
    return {"reference": round(trained / 3000 / 60), "train": round(trained / 1030 / 60) + 26}


def jobs(recipe_path: Path, deploy_recipe: str, root: Path) -> dict[str, Json]:
    recipe = cast(Json, json.loads(recipe_path.read_text()))
    kind = cast(str, recipe["kind"])
    name = cast(str, cast(Json, recipe["arm"])["name"])
    estimate = minutes(recipe, kind)
    gates = cast(list[int], cast(Json, recipe["dev_gate"])["mandatory_steps"])
    out: dict[str, Json] = {}
    requires: list[str] = []
    if kind == "dpo":
        reference = f"{name}-reference"
        out[reference] = {
            "id": reference, "cwd": str(root), "env": [], "priority": PRIORITY[kind], "resumable": True,
            "estimated_minutes": estimate["reference"], "requires": [],
            "command": command(root, "reference", "--recipe", deploy_recipe),
            "note": (f"C13 DPO reference log-probabilities of all pairs under {cast(Json, recipe['parent'])['tag']} (bf16, stored and "
                     "hashed); resumable: keeps its verified prefix at the deadline"),
        }
        requires = [reference]
    out[name] = {
        "id": name, "cwd": str(root), "env": [], "priority": PRIORITY[kind], "resumable": True,
        "estimated_minutes": estimate["train"], "requires": requires,
        "command": command(root, "run", "--recipe", deploy_recipe),
        "note": (f"C13 {kind.upper()} from {cast(Json, recipe['parent'])['tag']}: gates {gates} under R1-offline (stop-for-user exits 3; "
                 f"write override-step-<step>.json and retry), then R5 against the starting model; read r5-decision.json"),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root", type=Path, default=Path("/home/m0hawk/sandbox/Sepalith"))
    parser.add_argument("--local-root", type=Path, default=Path(__file__).resolve().parents[6])
    parser.add_argument("--recipe", type=Path, required=True, help="the bound recipe in the local tree")
    parser.add_argument("--output", type=Path, required=True, help="directory for the job files")
    args = parser.parse_args(argv)
    paths = PathMap(args.deploy_root, args.local_root.resolve())
    written = []
    for identifier, job in jobs(args.recipe, paths.to_deploy(args.recipe), args.deploy_root).items():
        write_new(args.output / f"{identifier}.json", job)
        written.append(identifier)
    print(json.dumps(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
