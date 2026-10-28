#!/usr/bin/env python3
"""Night-queue job files for the C18 smoke and pilot (written, never enqueued here), the final hold-out
evaluation job of the selected checkpoint, and the pre-registered verdict.

``C18-grpo-exec-smoke`` (not resumable, about 40 minutes) proves the real CUDA path: train to update 4,
save, DEV75 + 96 validation tasks + CPT 2K (and the 803 CPT reference), resume to update 6.
``C18-grpo-exec-pilot-1`` (resumable) is the pilot from arm A 803 and requires the smoke.
Enqueue with ``python3 -m sepalith.ops.night_queue enqueue --file <job>.json``.

    python -m sepalith.training.grpo_exec.jobs write --output docs/training/plan-20260927/c18/jobs
    python -m sepalith.training.grpo_exec.jobs verdict --summary <archive>/pilot-summary.json --metrics <final work dir>/metrics.json
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import shutil
from typing import cast

from sepalith.training.grpo_exec import recipe as recipe_module, rules

Json = dict[str, object]
PYTHON = ".venv-sft/bin/python"
BOUND = "docs/training/plan-20260927/c18/bound"
SMOKE_RECIPE = f"{BOUND}/grpo-exec-smoke-arm-a-803.json"
PILOT_RECIPE = f"{BOUND}/grpo-exec-pilot-1-arm-a-803.json"
START_LABEL, CANDIDATE_LABEL = "arm-a-803", "c18-selected"
ENV = ("PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES=0", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "OMP_NUM_THREADS=8",
       "MKL_NUM_THREADS=8", "TOKENIZERS_PARALLELISM=false", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


WORKTREE_SMOKE_RECIPE = f"{BOUND}/grpo-exec-smoke-worktree-arm-a-803.json"


def command(root: Path, module: str, *args: str, python: Path | None = None) -> list[str]:
    return ["/usr/bin/env", f"PYTHONPATH={root}/packages/sepalith/src", *ENV, str(python or root / PYTHON), "-B", "-m", module, *args]


def worktree_smoke_job(root: Path, python: Path) -> Json:
    """The smoke from an unmerged worktree: cwd, PYTHONPATH and recipe in ``root``, the interpreter from the main checkout."""
    job = jobs(root)["C18-grpo-exec-smoke"]
    job.update(id="C18-grpo-exec-smoke-worktree",
               command=command(root, "sepalith.training.grpo_exec.night", "smoke", "--recipe", str(root / WORKTREE_SMOKE_RECIPE),
                               python=python),
               note=(f"Pre-merge run of the C18 smoke from the worktree {root} (PRs #91/#92 unmerged): same data, model and "
                     "settings as C18-grpo-exec-smoke (train to update 4 and save; DEV75, 96 validation tasks and CPT 2K on "
                     "checkpoint 4 plus the 803 CPT reference; resume to update 6 and save), its own outputs. Exit 0 pass, 1 fail. "
                     "Receipt: /mnt/e/sepalith/campaign-20260915/checkpoints/C18-grpo-exec-smoke-worktree/smoke-receipt.json "
                     "(seconds per update, projected updates per night, peak memory and headroom)."))
    return job


def jobs(root: Path = recipe_module.DEPLOY_ROOT) -> dict[str, Json]:
    night = "sepalith.training.grpo_exec.night"
    return {
        "C18-grpo-exec-smoke": {
            "id": "C18-grpo-exec-smoke", "cwd": str(root), "env": [], "priority": 40, "resumable": False,
            "estimated_minutes": 45, "requires": [],
            "command": command(root, night, "smoke", "--recipe", str(root / SMOKE_RECIPE)),
            "note": ("C18 smoke on the real CUDA path (cuda:0): GRPO on the exec environment from arm A 803 (exec reward + "
                     "1-in-5 legacy v2.1 replay, SFT cross-entropy replay 0.1 x 8 rows), train to update 4 and save; DEV75, "
                     "the first 96 validation tasks (exec reward) and the CPT 2K probe on checkpoint 4 plus the 803 CPT reference; "
                     "resume to update 6 and save. Exit 0 pass, 1 fail. Receipt: /mnt/e/.../checkpoints/C18-grpo-exec-smoke/"
                     "smoke-receipt.json (seconds per update, projected updates per night, peak memory and headroom)."),
        },
        "C18-grpo-exec-pilot-1": {
            "id": "C18-grpo-exec-pilot-1", "cwd": str(root), "env": [], "priority": 45, "resumable": True,
            "estimated_minutes": 960, "requires": ["C18-grpo-exec-smoke"],
            "command": command(root, night, "run", "--recipe", str(root / PILOT_RECIPE)),
            "note": ("C18 GRPO pilot on the exec environment from arm A 803 (card C18, pre-registered): first the 803 validation "
                     "and CPT references, then DEV75 + validation exec + CPT 2K every 50 updates and DEV250 every 150; R6x decides "
                     "(stop-for-user exits 3: write override-step-<n>.json in the evidence directory and retry; rollback ends the "
                     "pilot); complete at update 1,000 or a plateau; then selects the best accepted checkpoint by validation "
                     "all-pass and writes final-holdout-job.json for the orchestrator. Resumable at buffer boundaries (exit 75)."),
        },
    }


def final_holdout_job(recipe: Mapping[str, object], selected: str, work_dir: Path, root: Path) -> Json:
    """The frozen hold-out evaluation of the selected checkpoint vs 803; seeds 803's pinned greedy completions."""
    final = cast(Json, recipe["final_evaluation"])
    tasks = cast(Json, final["holdout_tasks"])
    start = cast(Json, final["start_completions"])
    work_dir.mkdir(parents=True, exist_ok=True)
    seeded = work_dir / f"completions-{START_LABEL}.jsonl"
    if not seeded.exists():
        source = Path(str(start["path"]))
        if recipe_module.sha256(source) != start["sha256"]:
            raise ValueError("803 hold-out completions differ from their pin")
        shutil.copyfile(source, seeded)
    if recipe_module.sha256(seeded) != start["sha256"]:
        raise ValueError(f"{seeded} differs from 803's pinned completions")
    models = work_dir / "inputs" / "models.json"
    models.parent.mkdir(parents=True, exist_ok=True)
    models.write_text(json.dumps({"models": [
        {"label": START_LABEL, "name": "arm A checkpoint 803 (start)", "path": str(cast(Json, recipe["parent"])["path"])},
        {"label": CANDIDATE_LABEL, "name": "C18 selected checkpoint", "path": selected}]}, indent=1) + "\n")
    return {
        "id": "C18-final-holdout", "cwd": str(root), "env": [], "priority": 30, "resumable": True, "estimated_minutes": 120,
        "requires": [],
        "command": command(root, "sepalith.exec_eval.jobs", "run", "--kind", "holdout", "--job-id", "C18-final-holdout",
                           "--work-dir", str(work_dir), "--tasks", str(tasks["path"]), "--tasks-sha256", str(tasks["sha256"]),
                           "--models", str(models)),
        "note": (f"C18 pre-registered final evaluation: greedy completions of {selected} on the frozen exec hold-out "
                 "(3,803 tasks, 630 functions), 803's pinned completions reused, exec reward at c631141; paired, "
                 "function-clustered differences vs 803 in metrics.json (vs_arm-a-803). Verdict: "
                 "python -m sepalith.training.grpo_exec.jobs verdict --summary <pilot-summary.json> --metrics <work dir>/metrics.json"),
    }


def verdict(summary_path: Path, metrics_path: Path) -> Json:
    summary = json.loads(summary_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    selected = cast(Json, summary["selected"])
    start = cast(Json, cast(Json, summary["start"])["metrics"])
    holdout = cast(Json, cast(Json, metrics[CANDIDATE_LABEL])[f"vs_{START_LABEL}"])
    out = rules.final_verdict(holdout, dev250_candidate=int(cast(int, cast(Json, selected["dev250"])["score250"])),
                              dev250_start=int(cast(int, start["score250"])), cpt_rise=float(cast(float, selected["cpt_rise_vs_start"])))
    overall = {label: cast(Json, metrics[label])["overall"] for label in (START_LABEL, CANDIDATE_LABEL)}
    out["protocol_validity"] = {label: cast(Json, value)["protocol_valid"] for label, value in overall.items()}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--deploy-root", type=Path, default=recipe_module.DEPLOY_ROOT)
    tree = sub.add_parser("write-worktree-smoke")
    tree.add_argument("--root", type=Path, required=True)
    tree.add_argument("--python", type=Path, default=recipe_module.DEPLOY_ROOT / PYTHON)
    tree.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verdict")
    check.add_argument("--summary", type=Path, required=True)
    check.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "write":
        written = []
        for identifier, job in jobs(args.deploy_root).items():
            recipe_module.write_new(args.output / f"{identifier}.json", job)
            written.append(identifier)
        print(json.dumps(written))
    elif args.command == "write-worktree-smoke":
        job = worktree_smoke_job(args.root.resolve(), args.python)
        recipe_module.write_new(args.output / f"{job['id']}.json", job)
        print(json.dumps([job["id"]]))
    else:
        print(json.dumps(verdict(args.summary, args.metrics), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
