#!/usr/bin/env python3
"""Dress rehearsal of A2 to A5 on dev groups: authorization, build, plan and freeze, launcher scoring.

Every step uses the code and settings of the real final run. The differences are listed here:

* The build authorization names `split: dev_group` with `rehearsal: true`. `build_final()` then takes
  the holdback split's 105 dev build groups by the same rule as the final split.
* The freeze statement is a rehearsal text written by this script, not the user's.
* `--build-groups` names a rehearsal group count (`rehearsal_build_groups`, refused for the final split).
  `--dev-blind` replaces the dev inventory with an empty one: every dev case intersects it, so the real
  memberships stop a dev rehearsal at the 20% overlap ceiling.
* The panel is labelled `rehearsal_dev_group`. For scoring, the launcher's panel-split constant and the
  guard's semantic-receipt split constant are set to that label in memory. Nothing else in the launcher
  is changed. It runs `protocol.run()`: freeze and plan pins, the guard, the frozen runtime with the
  pinned interpreter, weights, tokenizer and llama-server, the seal, audit, construct and scoring.
* Each model's generation is timed. The timing goes to a separate receipt.
* Scoring may use a pre-registered subset (`subset`): the first SUBSET_CASES row IDs of the full
  rehearsal panel in sha256_json([3407, "dress-rehearsal-subset", row_id]) order, written as a panel of
  its own (cases, payloads and a semantic receipt bound to the subset's bytes) over the same staged
  sources. The full panel still goes through the build's launcher dry run (load, semantic receipt,
  roots, O_NOFOLLOW reads and construct for every case). Reason: CPU bf16 on this Zen 3 machine
  (no AVX-512 or bf16 units) took about 250 s per case under load, so the full roster would not
  finish in a day. (The CPU run of 2026-11-23 used it; the GPU rehearsal scores the full panel.)
* GPU rehearsal (2026-11-24, user: "Switch to GPU"): `assemble --panel-root` plans a fresh root over the
  built panel of an earlier rehearsal root, pins the night-job file (`--night-job`) and writes the rehearsal
  freeze digest to `freeze-approved.sha256`. The night job (`final-protocol-v2/night_job.py rehearsal`) then
  runs `score` under the CUDA host guard, as the final job runs the launcher.

Commands (run from the repository root):

    python dress_rehearsal.py authorize --root DIR          # repository .venv (needs nothing special)
    python final_panel.py --split dev_group ... (printed by `authorize`)   # .venv with tree_sitter_r
    python dress_rehearsal.py subset --root DIR             # optional, pre-registered subset
    python dress_rehearsal.py assemble --root DIR [--subset]
    .venv-sft/bin/python dress_rehearsal.py score --root DIR            # the pinned interpreter
    python dress_rehearsal.py assemble --root NEW --panel-root DIR --night-job JOB.json   # GPU rehearsal

DIR must be a fresh directory under the mode-700 sealed-final state directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent / "final-protocol-v2"), str(HERE.parent / "final-admission-guards"),
                str(ROOT / "packages/sepalith/src")]

from admission_guard import _write_immutable_json, canonical_json, sha256_bytes, sha256_json  # noqa: E402

Json = dict[str, Any]
STATE = Path.home() / ".local/state/sepalith/sealed-final-v2"
REHEARSAL_SPLIT = "rehearsal_dev_group"
REHEARSAL_JOB = ROOT / "docs/training/plan-20260927/c17/jobs/C17-sealed-final-v2-gpu-rehearsal.json"
SUBSET_CASES = 30  # pre-registered 2026-11-23 before any rehearsal scoring


def write_once(path: Path, value: Json | str) -> str:
    data = value.encode() if isinstance(value, str) else canonical_json(value).encode() + b"\n"
    _write_immutable_json(path, data)
    os.chmod(path, 0o400)
    return sha256_bytes(data)


def file_sha(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def regular_files(*paths: Path) -> None:
    """Refuse symlinks, FIFOs and other non-regular files before anything reads them."""
    for path in paths:
        if not path.is_absolute() or not stat.S_ISREG(path.lstat().st_mode):
            raise SystemExit("rehearsal panel inputs must be absolute regular files")


def dev_blind_memberships(path: Path) -> Json:
    """The real eight inventories with the dev stage emptied: every dev rehearsal case is in the dev inventory."""
    memberships = json.loads(path.read_bytes())
    memberships["dev"] = {"row_ids": [], "package_ids": ["rehearsal:no-dev-inventory"], "group_ids": [],
                          "split_identities": [], "function_ids": []}
    return memberships


def authorize(root: Path, build_groups: int | None, dev_blind: bool) -> Json:
    import final_panel as F
    import protocol as v2

    root.mkdir(mode=0o700)
    models = STATE / "pins/models.json"
    memberships = STATE / "inventories/memberships.json"
    if dev_blind:
        blind = root / "memberships-dev-blind.json"
        write_once(blind, dev_blind_memberships(memberships))
        memberships = blind
    statement = root / "freeze-statement.txt"
    statement_sha = write_once(statement, (
        "Dress rehearsal freeze statement (not the user's). Weights: the plan roster in pins/models.json, "
        f"sha256_json {sha256_json(json.loads(models.read_bytes()))}. Harness: {v2.harness_hash()}.\n"))
    receipt = {
        "schema": F.SCHEMA + ".build-authorization", "status": "user-approved", "split": "dev_group",
        "rehearsal": True, "constructor_sha256": F.sha256_file(Path(F.__file__)),
        "recipe_sha256": F.sha256_file(F.RECIPE_PATH),
        "weights_sha256": sha256_json(json.loads(models.read_bytes())), "harness_sha256": v2.harness_hash(),
        "memberships_sha256": file_sha(memberships), "freeze_statement_sha256": statement_sha,
        "weights_and_harness_frozen": True, "out_dir": str(root / "panel"),
        "attempt_marker": str(root / "attempt.json"),
        **({"rehearsal_build_groups": build_groups} if build_groups else {}),
    }
    digest = write_once(root / "build-authorization.json", receipt)
    command = [
        "python", str(HERE / "final_panel.py"), "--split", "dev_group", "--out", str(root / "panel"),
        "--tokenizer", str(STATE / "tokenizer"), "--memberships", str(memberships), "--models", str(models),
        "--freeze-statement", str(statement), "--authorization", str(root / "build-authorization.json"),
        "--authorization-sha256", digest]
    return {"authorization_sha256": digest, "build_command": command}


def rehearsal_panel(root: Path) -> Path:
    """The rehearsal panel, after its aggregate build receipt shows a built dev rehearsal; nothing else is opened first."""
    panel = root / "panel"
    if panel.resolve() != panel or not panel.is_dir():
        raise SystemExit("rehearsal panel must be a real directory inside the rehearsal root")
    regular_files(panel / "build-receipt.json", root / "build-authorization.json", panel / "cases.json",
                  panel / "semantic-receipt.json", panel / "payloads.json")
    receipt = json.loads((panel / "build-receipt.json").read_bytes())
    authorization = file_sha(root / "build-authorization.json")
    if (receipt.get("split") != "dev_group" or receipt.get("status") != "built"
            or receipt.get("authorization_sha256") != authorization):
        raise SystemExit("not a built, authorized dev rehearsal panel")
    outputs = receipt["outputs"]
    if (file_sha(panel / "cases.json") != outputs["cases_sha256"]
            or file_sha(panel / "semantic-receipt.json") != outputs["semantic_receipt_sha256"]
            or file_sha(panel / "payloads.json") != outputs["payloads_sha256"]):
        raise SystemExit("rehearsal panel files differ from its build receipt")
    return panel


def subset(root: Path) -> Json:
    """Write the pre-registered scoring subset as a panel of its own, bound like the full panel."""
    panel, out = rehearsal_panel(root), root / "subset-panel"
    cases = json.loads((panel / "cases.json").read_bytes())
    if cases["split"] != REHEARSAL_SPLIT:
        raise SystemExit("subset is for the dev rehearsal panel only")
    order = sorted(cases["cases"], key=lambda c: sha256_json([3407, "dress-rehearsal-subset", c["row_id"]]))
    chosen = sorted(order[:SUBSET_CASES], key=lambda c: c["row_id"])
    ids = [c["row_id"] for c in chosen]
    payloads = json.loads((panel / "payloads.json").read_bytes())
    semantic = json.loads((panel / "semantic-receipt.json").read_bytes())
    fragment = json.loads((panel / "plan-fragment.json").read_bytes())
    out.mkdir(mode=0o700)
    cases_doc = {"split": cases["split"], "cases": chosen}
    cases_sha = write_once(out / "cases.json", cases_doc)
    write_once(out / "payloads.json", {i: payloads[i] for i in ids})
    semantic_sha = write_once(out / "semantic-receipt.json", {
        **semantic, "source_lock_sha256": cases_sha, "verified_case_ids": ids,
        "case_bindings": {i: semantic["case_bindings"][i] for i in ids}})
    write_once(out / "plan-fragment.json", {
        **fragment, "cases": {"path": str(out / "cases.json"), "sha256": cases_sha}, "case_count": len(ids),
        "semantic_receipt": {"path": str(out / "semantic-receipt.json"), "sha256": semantic_sha},
        "payloads_path": str(out / "payloads.json")})
    return {"cases": len(ids), "families": dict(sorted(Counter(c["family"] for c in chosen).items()))}


def assemble(root: Path, use_subset: bool, panel_root: Path | None = None, night_job: Path | None = None) -> Json:
    import protocol as v2

    rehearsals = STATE / "rehearsals"
    if panel_root is not None or night_job is not None:
        # GPU rehearsal: a fresh root over the built panel of an earlier one, both under rehearsals/, pinning
        # only the reviewed rehearsal job. The final run root is never written here.
        command = json.loads(REHEARSAL_JOB.read_bytes())["command"]
        if (panel_root is None or night_job is None or use_subset or panel_root == root
                or root.parent != rehearsals or panel_root.parent != rehearsals
                or night_job.resolve(strict=True) != REHEARSAL_JOB
                or command[command.index("--root") + 1] != str(root)):
            raise SystemExit("--panel-root and --night-job plan a fresh rehearsals/ root with the rehearsal job")
        root.mkdir(mode=0o700)
    source = panel_root or root
    panel = rehearsal_panel(source)
    chosen = (source / "subset-panel") if use_subset else panel
    if chosen.resolve() != chosen:
        raise SystemExit("subset panel must be a real directory inside the rehearsal root")
    regular_files(*(chosen / name for name in ("plan-fragment.json", "cases.json", "semantic-receipt.json",
                                               "payloads.json")))
    fragment = json.loads((chosen / "plan-fragment.json").read_bytes())
    # The plan takes exactly the chosen panel's canonical files at their current digests. rehearsal_panel()
    # checked the full panel's files against its build receipt.
    expected = {"cases": {"path": str(chosen / "cases.json"), "sha256": file_sha(chosen / "cases.json")},
                "semantic_receipt": {"path": str(chosen / "semantic-receipt.json"),
                                     "sha256": file_sha(chosen / "semantic-receipt.json")},
                "payloads_path": str(chosen / "payloads.json"), "source_roots": [str(panel / "final-sources")]}
    if any(fragment.get(key) != value for key, value in expected.items()):
        raise SystemExit("plan fragment differs from the verified rehearsal panel files")
    # The audit's memberships are the ones the build authorization bound.
    authorization = json.loads((source / "build-authorization.json").read_bytes())
    memberships = Path(fragment["memberships"]["path"])
    if fragment["memberships"]["sha256"] != authorization["memberships_sha256"]:
        raise SystemExit("plan fragment memberships differ from the build authorization")
    regular_files(memberships)
    if file_sha(memberships) != authorization["memberships_sha256"]:
        raise SystemExit("plan fragment memberships differ from the build authorization")
    cases_count = len(json.loads((chosen / "cases.json").read_bytes())["cases"])
    if fragment["case_count"] != cases_count:
        raise SystemExit("plan fragment case count differs from its cases")
    fragment = {**expected, "memberships": fragment["memberships"], "case_count": cases_count}
    if use_subset:
        # Every subset row must be a row of the verified full panel, unchanged, under the rehearsal label.
        full = {c["row_id"]: c for c in json.loads((panel / "cases.json").read_bytes())["cases"]}
        cases = json.loads((chosen / "cases.json").read_bytes())
        if cases["split"] != REHEARSAL_SPLIT or any(full.get(c["row_id"]) != c for c in cases["cases"]):
            raise SystemExit("subset cases are not cases of the verified rehearsal panel")
    pins = {name: json.loads((STATE / "pins" / f"{name}.json").read_bytes())
            for name in ("models", "release_artifacts", "tokenizer", "runtime")}
    state = root / "run-state"
    state.mkdir(mode=0o700)
    runtime = {k: v for k, v in pins["runtime"].items()
               if k in ("python_sha256", "distributions", "torch_cuda", "device", "llama_server")}
    plan = {"schema": v2.SCHEMA + ".plan", "models": pins["models"], "release_artifacts": pins["release_artifacts"],
            "tokenizer": pins["tokenizer"], "runtime": runtime, **fragment, "state_dir": str(state)}
    if night_job is not None:
        plan["night_job"] = {"path": str(REHEARSAL_JOB), "sha256": file_sha(REHEARSAL_JOB)}
    plan_sha = write_once(root / "plan.json", plan)
    now = datetime.now(timezone.utc).isoformat()
    freeze = {"schema": v2.SCHEMA + ".freeze", "status": "frozen", "weights_frozen": True, "harness_frozen": True,
              "final_access_unlocked": True, "weights_frozen_at": now, "harness_frozen_at": now,
              "plan_sha256": plan_sha, "weights_sha256": sha256_json(pins["models"]),
              "harness_sha256": v2.harness_hash()}
    freeze_sha = write_once(root / "freeze.json", freeze)
    if night_job is not None:
        # The night job reads the digest apart from the receipt, as it reads the user's for the final run.
        write_once(root / "freeze-approved.sha256", freeze_sha + "\n")
    return {"plan_sha256": plan_sha, "freeze_sha256": freeze_sha, "case_count": plan["case_count"]}


def score(root: Path, freeze_sha256: str) -> Json:
    import admission_guard
    import protocol as v2
    import runtime as frozen

    # The only in-memory change: the rehearsal panel's split label.
    v2.PANEL_SPLIT = admission_guard.SEMANTIC_RECEIPT_SPLIT = REHEARSAL_SPLIT
    timings: Json = {}

    class TimedRuntime(frozen.FrozenRuntime):
        @contextmanager
        def generator(self, label: str) -> Iterator[Any]:
            calls = {"cases": 0, "generated_tokens": 0, "cap_hits": 0}
            start = time.monotonic()
            with super().generator(label) as generate:
                loaded = time.monotonic()

                def timed(prompt: list[int]) -> list[int]:
                    ids = generate(prompt)
                    calls["cases"] += 1
                    calls["generated_tokens"] += len(ids)
                    calls["cap_hits"] += int(ids[-1] not in v2.STOP_IDS)
                    if calls["cases"] % 25 == 0:
                        print(f"{label}: {calls['cases']} cases, {time.monotonic() - loaded:.0f} s", flush=True)
                    return ids

                yield timed
            timings[label] = {**calls, "load_seconds": round(loaded - start, 1),
                              "generation_seconds": round(time.monotonic() - loaded, 1)}

    frozen.FrozenRuntime = TimedRuntime
    args = argparse.Namespace(plan=root / "plan.json", freeze_receipt=root / "freeze.json",
                              freeze_sha256=freeze_sha256, resume=None, resume_sha256=None)
    start = time.monotonic()
    report = v2.run(args)
    timing = {"schema": "sepalith.final-panel.v1.dress-rehearsal-timing", "models": timings,
              "total_seconds": round(time.monotonic() - start, 1), "threads": frozen.THREADS,
              "load_average_at_end": os.getloadavg(), "cpu_count": os.cpu_count()}
    write_once(root / "timing.json", timing)
    return {"status": report["status"], "reason": report.get("reason"), "timing": timing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("authorize", "subset", "assemble", "score"))
    parser.add_argument("--subset", action="store_true", help="assemble: plan the pre-registered subset panel")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--panel-root", type=Path, help="assemble: plan a fresh root over this root's built panel")
    parser.add_argument("--night-job", type=Path, help="assemble: pin this night-job file in the plan")
    parser.add_argument("--build-groups", type=int, help="authorize: rehearsal group count (default: the recipe's)")
    parser.add_argument("--dev-blind", action="store_true",
                        help="authorize: empty the dev inventory so the launcher's audit can pass on dev cases")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_relative_to(STATE):
        parser.error("--root must lie under the sealed-final state directory")
    if args.step == "authorize":
        out = authorize(root, args.build_groups, args.dev_blind)
    elif args.step == "subset":
        out = subset(root)
    elif args.step == "assemble":
        panel_root = args.panel_root.resolve() if args.panel_root else None
        out = assemble(root, args.subset, panel_root, args.night_job)
    else:
        if not args.freeze_sha256:
            parser.error("score needs --freeze-sha256")
        out = score(root, args.freeze_sha256)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 2 if args.step == "score" and out["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
