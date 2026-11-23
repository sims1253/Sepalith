#!/usr/bin/env python3
"""Dress rehearsal of A2 to A5 on dev groups: authorization, build, plan and freeze, launcher scoring.

Every step uses the code and settings of the real final run. The differences are listed here:

* The build authorization names `split: dev_group` with `rehearsal: true`. `build_final()` then takes
  the holdback split's 105 dev build groups by the same rule as the final split.
* The freeze statement is a rehearsal text written by this script, not the user's.
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
  finish in a day.

Commands (run from the repository root):

    python dress_rehearsal.py authorize --root DIR          # repository .venv (needs nothing special)
    python final_panel.py --split dev_group ... (printed by `authorize`)   # .venv with tree_sitter_r
    python dress_rehearsal.py subset --root DIR             # optional, pre-registered subset
    python dress_rehearsal.py assemble --root DIR [--subset]
    .venv-sft/bin/python dress_rehearsal.py score --root DIR            # the pinned interpreter

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
SUBSET_CASES = 30  # pre-registered 2026-11-23 before any rehearsal scoring


def write_once(path: Path, value: Json | str) -> str:
    data = value.encode() if isinstance(value, str) else canonical_json(value).encode() + b"\n"
    _write_immutable_json(path, data)
    os.chmod(path, 0o400)
    return sha256_bytes(data)


def file_sha(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def authorize(root: Path) -> Json:
    import final_panel as F
    import protocol as v2

    root.mkdir(mode=0o700)
    models = STATE / "pins/models.json"
    memberships = STATE / "inventories/memberships.json"
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
    }
    digest = write_once(root / "build-authorization.json", receipt)
    command = [
        "python", str(HERE / "final_panel.py"), "--split", "dev_group", "--out", str(root / "panel"),
        "--tokenizer", str(STATE / "tokenizer"), "--memberships", str(memberships), "--models", str(models),
        "--freeze-statement", str(statement), "--authorization", str(root / "build-authorization.json"),
        "--authorization-sha256", digest]
    return {"authorization_sha256": digest, "build_command": command}


def subset(root: Path) -> Json:
    """Write the pre-registered scoring subset as a panel of its own, bound like the full panel."""
    panel, out = root / "panel", root / "subset-panel"
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


def assemble(root: Path, use_subset: bool) -> Json:
    import protocol as v2

    fragment = json.loads((root / ("subset-panel" if use_subset else "panel") / "plan-fragment.json").read_bytes())
    pins = {name: json.loads((STATE / "pins" / f"{name}.json").read_bytes())
            for name in ("models", "release_artifacts", "tokenizer", "runtime")}
    state = root / "run-state"
    state.mkdir(mode=0o700)
    runtime = {k: v for k, v in pins["runtime"].items() if k in ("python_sha256", "distributions", "llama_server")}
    plan = {"schema": v2.SCHEMA + ".plan", "models": pins["models"], "release_artifacts": pins["release_artifacts"],
            "tokenizer": pins["tokenizer"], "runtime": runtime, **fragment, "state_dir": str(state)}
    plan_sha = write_once(root / "plan.json", plan)
    now = datetime.now(timezone.utc).isoformat()
    freeze = {"schema": v2.SCHEMA + ".freeze", "status": "frozen", "weights_frozen": True, "harness_frozen": True,
              "final_access_unlocked": True, "weights_frozen_at": now, "harness_frozen_at": now,
              "plan_sha256": plan_sha, "weights_sha256": sha256_json(pins["models"]),
              "harness_sha256": v2.harness_hash()}
    freeze_sha = write_once(root / "freeze.json", freeze)
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
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_relative_to(STATE):
        parser.error("--root must lie under the sealed-final state directory")
    if args.step == "authorize":
        out = authorize(root)
    elif args.step == "subset":
        out = subset(root)
    elif args.step == "assemble":
        out = assemble(root, args.subset)
    else:
        if not args.freeze_sha256:
            parser.error("score needs --freeze-sha256")
        out = score(root, args.freeze_sha256)
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
