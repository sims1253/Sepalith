#!/usr/bin/env python3
"""Card C08 morning step, automated: apply rule R2 and enqueue C09 arm A.

Reads the gate-268 decision of each learning-rate pilot arm, checks that every
decision is bound to its arm (bound recipe sha256, step, outcome, checkpoint
manifest, result files), and applies R2 (docs/training/plan-20260927/RULES.md)
exactly as written:

1. Discard arms that fail the R1 first-gate floor.
2. Pick the arm with the highest ``score250``.
3. If the top two are within 5, pick the lower learning rate.
4. If they are still tied, pick the lower CPT 2K validation loss. The pilot
   decisions do not record it; this script never computes it, so reaching
   this step is stop-for-user.
5. If no arm survives step 1, stop-for-user.

It writes a selection receipt with every input and the outcome. On a clear
winner it enqueues the continuation job ``C09-arm-a-<lr>`` through the
``sepalith.ops.night_queue`` API (atomic, under ``queue.lock``): the winner's
C08 job with ``--stop-after-gate 268`` removed, so ``night_arm arm`` resumes
from the gate-268 checkpoint through gates 536, 803 and 1,071 under R1.

Standard library only. Run with ``/usr/bin/python3 -I -B``. No CUDA.

Exit codes: 0 winner enqueued (or already queued with the same definition);
3 stop-for-user (the message names the question); anything else is an error.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import cast

CODE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPO = Path("/home/m0hawk/sandbox/Sepalith")
DEFAULT_STATE = Path.home() / ".local/state/sepalith"
PLAN = Path("docs/training/plan-20260927")

ARMS = ("lr3e-6", "lr1e-5", "lr3e-5")
GATE = 268
TIE_BAND = 5
FIRST_GATE_MIN_VALID = 60
FIRST_GATE_MAX_CAP = 2
COUNTS = ("valid", "cap", "edit", "nofp", "score250")

BOUND_SCHEMA = "sepalith.sft-v2.full-weight-edit-sft-bound.v1"
DECISION_SCHEMA = "sepalith.sft11.full-weight-edit-dev-decision.v1"
RECEIPT_SCHEMA = "sepalith.c08.r2-selection.v1"
SELECT_JOB_ID = "C08-r2-select"
CONTINUATION_PREFIX = "C09-arm-a-"
CONTINUATION_MINUTES = 240
CONTINUATION_PRIORITY = 24
# Recipe sections that must be identical across pilot arms ("same parent with identical data order").
SHARED_RECIPE_KEYS = ("parent", "seed", "data_queue", "pre_registered", "cohort", "development")
# Decision fields that must be identical across pilot arms (same panels, same decoding).
SHARED_DECISION_KEYS = ("panel_sha256", "dev250", "generation_runtime")

CONTINUE = "continue"
STOP_FOR_USER = "stop_for_user"
EXIT_WINNER = 0
EXIT_STOP_FOR_USER = 3

Json = dict[str, object]


class Refused(Exception):
    """An input is missing, inconsistent or not bound to its arm: stop-for-user."""


# Pure rule logic -------------------------------------------------------------------------------

def first_gate_floor(metrics: Mapping[str, object], parent_edit: object) -> list[str]:
    """Reasons the R1 first-gate floor fails; empty if it passes.

    R1: continue if valid >= 60, cap <= 2, the loss is finite and edit beats the parent's edit.
    """
    reasons = []
    for name in COUNTS:
        if type(metrics.get(name)) is not int:
            reasons.append(f"{name} is missing or not a count ({metrics.get(name)!r})")
    loss = metrics.get("loss")
    if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
        reasons.append(f"loss is missing or not finite ({loss!r})")
    if type(parent_edit) is not int:
        reasons.append(f"parent edit is missing or not a count ({parent_edit!r})")
    if reasons:
        return reasons
    valid, cap, edit = (int(str(metrics[name])) for name in ("valid", "cap", "edit"))
    if valid < FIRST_GATE_MIN_VALID:
        reasons.append(f"valid {valid} < {FIRST_GATE_MIN_VALID}")
    if cap > FIRST_GATE_MAX_CAP:
        reasons.append(f"cap {cap} > {FIRST_GATE_MAX_CAP}")
    if edit <= int(str(parent_edit)):
        reasons.append(f"edit {edit} does not beat parent edit {parent_edit}")
    return reasons


def apply_r2(arms: Sequence[Mapping[str, object]]) -> Json:
    """Apply R2 to arms ``{"arm", "learning_rate", "score250", "floor_failures", "cpt_2k_validation_loss"}``.

    Returns ``{"outcome", "winner", "steps", "reason"}``. ``outcome`` is
    ``continue`` with a winner or ``stop_for_user`` without one.
    """
    steps: list[Json] = []

    def stop(reason: str) -> Json:
        return {"outcome": STOP_FOR_USER, "winner": None, "steps": steps, "reason": reason}

    def pick(arm: Mapping[str, object], reason: str) -> Json:
        return {"outcome": CONTINUE, "winner": str(arm["arm"]), "steps": steps, "reason": reason}

    survivors = [arm for arm in arms if not arm["floor_failures"]]
    steps.append({"step": 1, "rule": "discard arms that fail the R1 first-gate floor",
                  "discarded": {str(arm["arm"]): arm["floor_failures"] for arm in arms if arm["floor_failures"]},
                  "survivors": [str(arm["arm"]) for arm in survivors]})
    if not survivors:
        steps.append({"step": 5, "rule": "no arm survives step 1: stop-for-user"})
        return stop("no pilot arm passed the R1 first-gate floor")
    ranked = sorted(survivors, key=lambda arm: -_score(arm))
    steps.append({"step": 2, "rule": "highest score250",
                  "ranking": [{"arm": str(arm["arm"]), "score250": _score(arm)} for arm in ranked]})
    if len(ranked) == 1:
        return pick(ranked[0], f"{ranked[0]['arm']} is the only arm that passed the R1 first-gate floor")
    top, second = _score(ranked[0]), _score(ranked[1])
    gap = top - second
    if gap > TIE_BAND:
        steps.append({"step": 3, "rule": f"top two within {TIE_BAND}: pick the lower learning rate",
                      "gap": gap, "applies": False})
        return pick(ranked[0], f"{ranked[0]['arm']} has the highest score250 ({top}), {gap} above the next arm "
                               f"(outside the tie band of {TIE_BAND})")
    # The top two, counting every arm tied with the second-ranked score.
    band = [arm for arm in ranked if _score(arm) >= second]
    lowest_lr = min(_lr(arm) for arm in band)
    lowest = [arm for arm in band if _lr(arm) == lowest_lr]
    steps.append({"step": 3, "rule": f"top two within {TIE_BAND}: pick the lower learning rate", "gap": gap,
                  "applies": True, "band": [str(arm["arm"]) for arm in band],
                  "lowest_learning_rate": [str(arm["arm"]) for arm in lowest]})
    if len(lowest) == 1:
        return pick(lowest[0], f"top two within {TIE_BAND} (gap {gap}); {lowest[0]['arm']} has the lower "
                               f"learning rate ({lowest_lr:g})")
    losses = {str(arm["arm"]): arm.get("cpt_2k_validation_loss") for arm in lowest}
    step4: Json = {"step": 4, "rule": "still tied: pick the lower CPT 2K validation loss", "losses": losses}
    steps.append(step4)
    usable = {name: value for name, value in losses.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
    if len(usable) != len(losses):
        return stop(f"R2 step 4 needs the CPT 2K validation loss of {sorted(losses)}, which the pilot "
                    "decisions do not record; this selector does not compute it")
    best = min(usable.values())
    winners = [name for name, value in usable.items() if value == best]
    if len(winners) != 1:
        return stop(f"R2 step 4 is still tied on CPT 2K validation loss ({best}) between {winners}")
    step4["winner"] = winners[0]
    return pick(next(arm for arm in lowest if arm["arm"] == winners[0]),
                f"still tied after step 3; {winners[0]} has the lower CPT 2K validation loss ({best})")


def _score(arm: Mapping[str, object]) -> int:
    value = arm["score250"]
    if type(value) is not int:
        raise Refused(f"{arm['arm']}: score250 is not a count ({value!r})")
    return value


def _lr(arm: Mapping[str, object]) -> float:
    value = arm["learning_rate"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0:
        raise Refused(f"{arm['arm']}: learning rate is not positive ({value!r})")
    return float(value)


# Inputs and binding ----------------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> Json:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise Refused(f"{label} is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"{label} is unreadable: {path}: {error}") from None
    return obj(value, f"{label} ({path})")


def obj(value: object, label: str) -> Json:
    if not isinstance(value, dict):
        raise Refused(f"{label} must be an object")
    return cast(Json, value)


def _get(record: Mapping[str, object], key: str, label: str) -> Json:
    return obj(record.get(key), f"{label}: {key}")


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise Refused(f"{label} must be a list")
    return cast(list[object], value)


def _text(record: Mapping[str, object], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise Refused(f"{label}: {key} must be a non-empty string")
    return value


def lr_value(arm: str) -> float:
    """``lr1e-5`` -> 1e-05."""
    try:
        return float(arm.removeprefix("lr"))
    except ValueError:
        raise Refused(f"cannot read a learning rate from arm name {arm!r}") from None


def decision_files(evidence: Path) -> list[Path]:
    """The gate-268 decisions in attempt order; ``night_arm`` acts on the last one."""
    found = []
    for path in (evidence / "decisions").glob(f"decision-step-{GATE}-*.json"):
        suffix = path.stem.removeprefix(f"decision-step-{GATE}-")
        if suffix.isdigit():
            found.append((int(suffix), path))
    return [path for _, path in sorted(found)]


def stop_after_gate_index(command: Sequence[object]) -> int:
    hits = [index for index, part in enumerate(command) if part == "--stop-after-gate"]
    if len(hits) != 1 or hits[0] + 1 >= len(command) or command[hits[0] + 1] != str(GATE):
        raise Refused(f"the C08 job must carry exactly one '--stop-after-gate {GATE}'")
    return hits[0]


def recipe_argument(command: Sequence[object]) -> Path:
    hits = [index for index, part in enumerate(command) if part == "--recipe"]
    if len(hits) != 1 or hits[0] + 1 >= len(command):
        raise Refused("the C08 job must carry exactly one '--recipe PATH'")
    return Path(str(command[hits[0] + 1]))


def load_arm(arm: str, repo: Path, queue_root: Path, gates_root: Path | None) -> Json:
    """Every R2 input of one pilot arm, after checking it is bound to that arm."""
    label = arm
    recipe_path = repo / PLAN / f"c04/bound/sft-v2-{arm}.json"
    recipe = read_object(recipe_path, f"{label} bound recipe")
    recipe_sha = sha256(recipe_path)
    if recipe.get("schema") != BOUND_SCHEMA:
        raise Refused(f"{label}: {recipe_path} is not a v2 bound recipe")
    recipe_arm = _get(recipe, "arm", label)
    if recipe_arm.get("name") != arm:
        raise Refused(f"{label}: bound recipe names arm {recipe_arm.get('name')!r}")
    learning_rate = recipe_arm.get("learning_rate")
    if not isinstance(learning_rate, float) or learning_rate != lr_value(arm):
        raise Refused(f"{label}: bound recipe learning rate {learning_rate!r} does not match the arm name")
    gate = _get(recipe, "dev_gate", label)
    steps = gate.get("mandatory_steps")
    if not isinstance(steps, list) or not steps or steps[0] != GATE:
        raise Refused(f"{label}: the first mandatory gate is not {GATE} ({steps!r})")
    evidence = Path(_text(gate, "evidence_directory", label))
    if gates_root is not None:
        evidence = gates_root / evidence.name
    archive = Path(_text(_get(recipe, "outputs", label), "archive", label))

    # The C08 job that ran this arm: the repository file, and the copy the runner finished.
    job_path = repo / PLAN / f"c04/jobs/C08-arm-{arm}.json"
    job = read_object(job_path, f"{label} C08 job file")
    if job.get("id") != f"C08-arm-{arm}":
        raise Refused(f"{label}: {job_path} has id {job.get('id')!r}")
    command = job.get("command")
    if not isinstance(command, list):
        raise Refused(f"{label}: {job_path} has no command list")
    stop_after_gate_index(command)
    job_recipe = recipe_argument(command)
    if not job_recipe.is_file() or sha256(job_recipe) != recipe_sha:
        raise Refused(f"{label}: the C08 job's recipe {job_recipe} is not the bound recipe {recipe_path}")
    queued = [state for state in ("done", "pending", "running", "deferred", "failed")
              if (queue_root / state / f"C08-arm-{arm}.json").is_file()]
    if queued != ["done"]:
        raise Refused(f"{label}: C08-arm-{arm} must be in done/ of {queue_root}, found {queued or 'nothing'}")
    ran = read_object(queue_root / "done" / f"C08-arm-{arm}.json", f"{label} finished queue job")
    if ran.get("command") != command or ran.get("cwd") != job.get("cwd"):
        raise Refused(f"{label}: the finished C08-arm-{arm} job differs from {job_path}")

    # The gate-268 decision night_arm acts on: its last attempt.
    decisions = decision_files(evidence)
    if not decisions:
        raise Refused(f"{label}: no gate-{GATE} decision in {evidence / 'decisions'}")
    decision_path = decisions[-1]
    decision = read_object(decision_path, f"{label} decision")
    problems = []
    expect = {"schema": DECISION_SCHEMA, "step": GATE, "bound_recipe_sha256": recipe_sha,
              "evaluation_complete": True, "decided_by": "rule_R1", "override": None}
    problems += [f"{key} is {decision.get(key)!r}, expected {value!r}" for key, value in expect.items()
                 if decision.get(key) != value]
    outcome = decision.get("outcome")
    if outcome not in (CONTINUE, STOP_FOR_USER) or decision.get("rule_outcome") != outcome:
        problems.append(f"outcome {outcome!r} / rule_outcome {decision.get('rule_outcome')!r} is not a first-gate R1 outcome")
    rule = _get(decision, "rule", label)
    if rule.get("rule") != "R1" or rule.get("kind") != "first_gate" or rule.get("outcome") != outcome:
        problems.append("the rule record is not the R1 first gate that produced this outcome")
    if rule.get("problems"):
        problems.append(f"R1 recorded problems (evaluator crash or binding): {rule.get('problems')}")
    metrics = _get(decision, "metrics", label)
    inputs = _get(rule, "inputs", label)
    if inputs.get("current") != metrics:
        problems.append("metrics differ from the R1 inputs")
    parent = _get(inputs, "parent", label)
    floor = first_gate_floor(metrics, parent.get("edit"))
    if (outcome == CONTINUE) != (not floor):
        problems.append(f"recorded outcome {outcome!r} disagrees with the R1 floor recomputed here ({floor or 'passes'})")
    results = _get(decision, "results", label)
    for panel in ("dev75", "dev250"):
        entry = _get(results, panel, f"{label} results")
        path = Path(_text(entry, "path", f"{label} {panel} result"))
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            problems.append(f"{panel} result {path} is missing or its sha256 differs")
    checkpoint = archive / "full" / f"checkpoint-{GATE}"
    manifest = checkpoint / "campaign-manifest.json"
    manifest_sha = sha256(manifest) if manifest.is_file() else None
    if manifest_sha is None or manifest_sha != decision.get("checkpoint_manifest_sha256"):
        problems.append(f"checkpoint manifest {manifest} is missing or not the one the decision evaluated")
    later = sorted(int(path.name.removeprefix("checkpoint-")) for path in (archive / "full").glob("checkpoint-*")
                   if path.name.removeprefix("checkpoint-").isdigit() and (path / "campaign-manifest.json").is_file())
    if later and later[-1] != GATE:
        problems.append(f"the latest durable checkpoint is {later[-1]}, not {GATE}; a resume would not start at the gate")
    gate_evidence = evidence / f"dev-gate-step-{GATE}.json"
    if outcome == CONTINUE and not gate_evidence.is_file():
        problems.append(f"gate evidence {gate_evidence} is missing; night_arm needs it to continue past {GATE}")
    if problems:
        raise Refused(f"{label}: decision {decision_path} is not usable: " + "; ".join(problems))
    return {
        "arm": arm, "learning_rate": learning_rate, "score250": metrics.get("score250"), "metrics": metrics,
        "parent_metrics": parent, "floor_failures": floor, "r1_outcome": outcome,
        "cpt_2k_validation_loss": None,
        "recipe": {"path": str(recipe_path), "sha256": recipe_sha},
        "decision": {"path": str(decision_path), "sha256": sha256(decision_path), "attempts": len(decisions),
                     "decided_by": decision.get("decided_by"), "at": decision.get("at")},
        "checkpoint": {"path": str(checkpoint), "campaign_manifest_sha256": manifest_sha},
        "gate_evidence": str(gate_evidence) if gate_evidence.is_file() else None,
        "c08_job": {"path": str(job_path), "sha256": sha256(job_path)},
        "_shared_recipe": {key: recipe.get(key) for key in SHARED_RECIPE_KEYS},
        "_shared_decision": {key: decision.get(key) for key in SHARED_DECISION_KEYS},
        "_job": job,
    }


def check_shared(arms: Sequence[Mapping[str, object]]) -> None:
    """The pilot compares learning rates only: same parent, data order, panels and decoding."""
    first = arms[0]
    for arm in arms[1:]:
        for key in ("_shared_recipe", "_shared_decision", "parent_metrics"):
            if arm[key] != first[key]:
                raise Refused(f"{arm['arm']} and {first['arm']} differ in {key.strip('_')}; R2 compares like with like")


# Continuation job ------------------------------------------------------------------------------

def continuation_job(winner: Mapping[str, object], receipt: Path, environ: Mapping[str, str]) -> Json:
    """The winner's C08 job without ``--stop-after-gate``: resume from gate 268 to the end (C09)."""
    job = obj(winner["_job"], "C08 job")
    command = [str(part) for part in _list(job["command"], "C08 job command")]
    index = stop_after_gate_index(command)
    del command[index:index + 2]
    arm = str(winner["arm"])
    checkpoint = obj(winner["checkpoint"], "checkpoint")
    requires = [str(item) for item in _list(job.get("requires", []), "requires")] + [f"C08-arm-{arm}"]
    # Under the night runner, hold the continuation until this selection job is in done/.
    if environ.get("SEPALITH_NIGHT_JOB_ID"):
        requires.append(environ["SEPALITH_NIGHT_JOB_ID"])
    return {
        "id": CONTINUATION_PREFIX + arm,
        "cwd": job["cwd"],
        "env": _list(job.get("env", []), "env"),
        "priority": CONTINUATION_PRIORITY,
        "resumable": True,
        "estimated_minutes": CONTINUATION_MINUTES,
        "requires": list(dict.fromkeys(requires)),
        "command": command,
        "note": (f"C09 arm A: {arm} selected by R2 (receipt {receipt}); resumes from gate-{GATE} checkpoint "
                 f"{checkpoint['path']} through gates 536, 803 and 1,071 under R1"),
    }


def _definition(job: Mapping[str, object]) -> Json:
    keys = ("id", "cwd", "env", "resumable", "requires", "command")
    return {key: job.get(key) for key in keys}


def enqueue(queue_root: Path, job: Json) -> Json:
    """Enqueue once. An identical existing job is accepted; any other C09 arm-A job is a conflict."""
    sys.path.insert(0, str(CODE_ROOT / "packages/sepalith/src"))
    from sepalith.ops.night_queue import STATES, Queue, QueueError

    queue = Queue(queue_root)
    existing = sorted((state, path) for state in STATES for path in (queue_root / state).glob(CONTINUATION_PREFIX + "*.json"))
    for state, path in existing:
        if path.stem != job["id"]:
            raise Refused(f"{path.stem} is already in {state}/; R2 selected {job['id']} now")
        current = read_object(path, f"queued {path.stem}")
        if _definition(current) != _definition(job):
            raise Refused(f"{path.stem} is already in {state}/ with a different definition")
        return {"job_id": job["id"], "state": state, "path": str(path), "already_queued": True}
    try:
        queue.enqueue(job)
    except QueueError as error:
        raise Refused(f"night_queue refused {job['id']}: {error}") from None
    return {"job_id": job["id"], "state": "pending", "path": str(queue.path("pending", str(job["id"]))),
            "already_queued": False}


# Output ----------------------------------------------------------------------------------------

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_receipt(path: Path, receipt: Json) -> None:
    """Keep an earlier receipt next to the new one rather than overwrite it."""
    if path.exists():
        os.replace(path, path.with_name(f"{path.stem}.superseded-{time.strftime('%Y%m%dT%H%M%S')}{path.suffix}"))
    write_json(path, receipt)


def public(arm: Mapping[str, object]) -> Json:
    return {key: value for key, value in arm.items() if not key.startswith("_")}


def select(repo: Path, queue_root: Path, receipt_path: Path, gates_root: Path | None, *, dry_run: bool,
           environ: Mapping[str, str]) -> tuple[int, Json]:
    receipt: Json = {
        "schema": RECEIPT_SCHEMA, "card": "C08", "rule": "R2",
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selector": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "repository": str(repo), "queue_root": str(queue_root), "dry_run": dry_run,
        "night_job_id": environ.get("SEPALITH_NIGHT_JOB_ID"),
        "inputs": {}, "problems": [], "selection": None, "winner": None, "continuation": None,
        "outcome": STOP_FOR_USER, "question_for_user": None,
    }
    arms: list[Json] = []
    problems: list[str] = []
    for arm in ARMS:
        try:
            arms.append(load_arm(arm, repo, queue_root, gates_root))
        except Refused as error:
            problems.append(str(error))
    receipt["inputs"] = {str(arm["arm"]): public(arm) for arm in arms}
    if not problems:
        try:
            check_shared(arms)
        except Refused as error:
            problems.append(str(error))
    if problems:
        receipt["problems"] = problems
        receipt["question_for_user"] = ("R2 was not applied because an input is missing or not bound to its arm: "
                                        + " | ".join(problems) + ". Resolve it, then retry C08-r2-select.")
        return EXIT_STOP_FOR_USER, receipt
    selection = apply_r2(arms)
    receipt["selection"] = selection
    if selection["outcome"] != CONTINUE:
        scores = ", ".join(f"{arm['arm']}: score250 {arm['score250']}, floor "
                           f"{'pass' if not arm['floor_failures'] else 'fail'}" for arm in arms)
        receipt["question_for_user"] = f"R2 says stop-for-user: {selection['reason']} ({scores})."
        return EXIT_STOP_FOR_USER, receipt
    winner = next(arm for arm in arms if arm["arm"] == selection["winner"])
    receipt["winner"] = {"arm": winner["arm"], "learning_rate": winner["learning_rate"],
                         "resume_from": winner["checkpoint"], "gate_evidence": winner["gate_evidence"],
                         "recipe": winner["recipe"]}
    job = continuation_job(winner, receipt_path, environ)
    receipt["continuation"] = {"job": job}
    if dry_run:
        receipt["outcome"] = CONTINUE
        return EXIT_WINNER, receipt
    try:
        queued = enqueue(queue_root, job)
    except Refused as error:
        receipt["problems"] = [str(error)]
        receipt["question_for_user"] = f"R2 selected {winner['arm']}, but the continuation was not enqueued: {error}"
        return EXIT_STOP_FOR_USER, receipt
    receipt["continuation"] = {"job": job, **queued}
    receipt["outcome"] = CONTINUE
    return EXIT_WINNER, receipt


def job_result(code: int, receipt: Mapping[str, object], receipt_path: Path | None) -> Json:
    summary_inputs = {}
    for name, value in obj(receipt.get("inputs"), "inputs").items():
        arm = obj(value, name)
        summary_inputs[name] = {"score250": arm.get("score250"), "learning_rate": arm.get("learning_rate"),
                                "r1_floor": "pass" if not arm.get("floor_failures") else arm.get("floor_failures")}
    selection = receipt.get("selection")
    winner = obj(receipt["winner"], "winner") if receipt.get("winner") is not None else None
    if code == EXIT_WINNER and winner is not None:
        continuation = obj(receipt.get("continuation"), "continuation")
        state = "already queued" if continuation.get("already_queued") else "enqueued"
        summary = f"R2 selected {winner['arm']}; {obj(continuation['job'], 'job')['id']} {state}"
    else:
        summary = "R2 stop-for-user"
    return {"summary": summary, "needs_user": receipt.get("question_for_user"),
            "rule_decisions": [{"rule": "R2", "inputs": summary_inputs, "outcome": receipt.get("outcome"),
                                "winner": winner["arm"] if winner is not None else None,
                                "reason": obj(selection, "selection").get("reason") if selection is not None else None,
                                "receipt": str(receipt_path) if receipt_path else None}]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO,
                        help="checkout with the bound recipes and C08 job files (default: %(default)s)")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_STATE / "night-queue")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_STATE / "campaign-20260915/C08-r2-selection.json")
    parser.add_argument("--gates-root", type=Path,
                        help="read each arm's evidence directory under this root instead of the recipe's path (tests)")
    parser.add_argument("--dry-run", action="store_true", help="print the selection; write and enqueue nothing")
    args = parser.parse_args(argv)
    environ = dict(os.environ)
    code, receipt = select(args.repo_root.resolve(), args.queue_root, args.receipt, args.gates_root,
                           dry_run=args.dry_run, environ=environ)
    if not args.dry_run:
        write_receipt(args.receipt, receipt)
    result_path = environ.get("SEPALITH_JOB_RESULT")
    if result_path and not args.dry_run:
        write_json(Path(result_path), job_result(code, receipt, args.receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if code == EXIT_WINNER:
        continuation, winner = obj(receipt["continuation"], "continuation"), obj(receipt["winner"], "winner")
        print(f"select_r2: R2 selected {winner['arm']}; "
              f"{'dry run, not enqueued' if args.dry_run else continuation.get('path')}", file=sys.stderr)
    else:
        print(f"select_r2: stop-for-user: {receipt['question_for_user']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
