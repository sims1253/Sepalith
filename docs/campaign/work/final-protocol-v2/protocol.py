#!/usr/bin/env python3
"""Construct and score the user-frozen sealed-final v2 panel."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Protocol, cast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GUARD_DIR = HERE.parent / "final-admission-guards"
PACKAGE_DIR = ROOT / "packages/sepalith/src"
sys.path[:0] = [str(GUARD_DIR), str(PACKAGE_DIR)]

from admission_guard import (  # noqa: E402
    AdmissionGuard,
    AdmissionSourceError,
    MEMBERSHIP_KEYS,
    SourceCaseSpec,
    _identity_tokens,
    _membership_tokens,
    _required_hash,
    _write_immutable_json,
    canonical_json,
    sha256_bytes,
    sha256_json,
)
from sepalith.campaign_protocol import (  # noqa: E402
    PromptContext,
    build_training_row,
)
from sepalith.quant.gguf_dev import DevTokenizer  # noqa: E402
from sepalith.training.eval.campaign_eval import validate_development_capacity  # noqa: E402

SCHEMA = "sepalith.sealed-final.v2"
EVALUATOR_COMMIT = "523efc07bcb86ebba7fe1030c0a2dc6356f9d111"
STAGES = ("train", "dev", "cpt", "c14", "synth", "exec_train", "exec_validation", "exec_holdout")
CAP = 1024
STOP_IDS = (1, 130073)
CAPACITY = 8192
MODELS = ("arm-a-803", "c19-100-bf16", "c19-100-q6_k")
OPTIONAL_MODEL = "c20-adopted"
COMPARISONS = (
    ("c19-100-bf16", "arm-a-803"),
    ("c19-100-q6_k", "c19-100-bf16"),
    ("c20-adopted", "c19-100-bf16"),
)
BOOTSTRAP_REPS = 10_000
SEED = 3407
Json = dict[str, Any]
Generate = Callable[[list[int]], list[int]]


class Runtime(Protocol):
    tokenizer: DevTokenizer

    def generator(self, label: str) -> AbstractContextManager[Generate]: ...


class RuntimeFactory(Protocol):
    def __call__(self, plan: Json, *, replay: bool) -> Runtime: ...


class ProtocolError(ValueError):
    """An input or run-state invariant failed."""


def pinned_json(path: Path, expected: str) -> Json:
    digest = _required_hash(expected, "pinned_json")
    data = path.read_bytes()
    if sha256_bytes(data) != digest:
        raise ProtocolError("input_hash_mismatch")
    value = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ProtocolError("json_object_required")
    return value


def read_pin(pin: Json) -> Json:
    return pinned_json(Path(pin["path"]), pin["sha256"])


def write_once(path: Path, value: Json) -> None:
    _write_immutable_json(path, canonical_json(value).encode("utf-8") + b"\n")


def harness_hash() -> str:
    """Bind the v2 entry points and guard to the unchanged pinned evaluator tree."""
    package = PACKAGE_DIR.relative_to(ROOT)
    diff = subprocess.run(
        ["git", "diff", "--quiet", EVALUATOR_COMMIT, "--", str(package)], cwd=ROOT, check=False,
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", str(package)], cwd=ROOT,
    )
    if diff.returncode or untracked:
        raise ProtocolError("evaluator_tree_differs_from_pin")
    files = [HERE / "protocol.py", HERE / "runtime.py", GUARD_DIR / "admission_guard.py"]
    return sha256_json({
        "evaluator_commit": EVALUATOR_COMMIT,
        "files": {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes()) for path in files},
    })


@dataclass(frozen=True)
class Case:
    spec: SourceCaseSpec
    function_ids: frozenset[str]


@dataclass(frozen=True)
class PreparedCase:
    spec: SourceCaseSpec
    context: PromptContext
    expected: list[str]
    prompt: list[int]
    overlapping: bool


def identities(values: object) -> frozenset[str]:
    if not isinstance(values, list) or any(
        not isinstance(item, str) or not item or item.strip() != item or "\x00" in item for item in values
    ):
        raise ProtocolError("invalid_identity_list")
    return frozenset(cast("list[str]", values))


def load_cases(document: Json, expected_case_count: int) -> list[Case]:
    rows = document["cases"]
    if (document.get("split") != "final_locked_v1" or not isinstance(rows, list)
            or len(rows) != expected_case_count or not rows):
        raise ProtocolError("unexpected_final_split_or_case_count")
    cases = [Case(SourceCaseSpec.from_mapping(row), identities(row["function_ids"])) for row in rows]
    for field in ("row_id", "split_identity"):
        if len({getattr(case.spec, field) for case in cases}) != len(cases):
            raise ProtocolError("duplicate_case_identity")
    return sorted(cases, key=lambda case: case.spec.row_id)


def package_identity(package: str) -> str:
    return package.removeprefix("exec:")


def audit(cases: list[Case], memberships: Json) -> tuple[set[str], Json]:
    """Flag a whole group when any member intersects any frozen inventory."""
    if set(memberships) != set(STAGES):
        raise ProtocolError("membership_stages_missing_or_unknown")
    stages = {}
    for stage, membership in memberships.items():
        if set(membership) != {*MEMBERSHIP_KEYS, "function_ids"}:
            raise ProtocolError("membership_keys_missing_or_unknown")
        members = {key: identities(membership[key]) for key in MEMBERSHIP_KEYS}
        functions = identities(membership["function_ids"])
        if not members["package_ids"] or (stage.startswith("exec_") and not functions):
            raise ProtocolError("incomplete_package_or_exec_function_membership")
        stages[stage] = (
            _membership_tokens(members, stage),
            {package_identity(name) for name in members["package_ids"]},
            functions,
        )

    all_groups = {case.spec.group_id for case in cases}
    overlap: set[str] = set()
    stage_counts = {}
    for stage, (tokens, packages, functions) in stages.items():
        groups = {
            case.spec.group_id for case in cases
            if (_identity_tokens(case.spec) & tokens or package_identity(case.spec.package_id) in packages
                or case.function_ids & functions)
        }
        stage_counts[stage] = len(groups)
        overlap.update(groups)
    return overlap, {
        "cases": len(cases),
        "groups": len(all_groups),
        "overlap_groups": len(overlap),
        "overlap_cases": sum(case.spec.group_id in overlap for case in cases),
        "overlap_groups_by_stage": stage_counts,
    }


def construct(
    cases: list[Case],
    payload_path: Path,
    guard: AdmissionGuard,
    overlap: set[str],
    tokenizer: DevTokenizer,
) -> tuple[list[PreparedCase], Counter[str]]:
    roots = guard._validate_roots()
    plans = [guard._validate_source_path(case.spec, roots) for case in cases]
    # Per-case pins detect drift without letting a whole-payload pin hide its count.
    if not payload_path.is_absolute():
        raise ProtocolError("payload_path_must_be_absolute")
    payloads = json.loads(payload_path.read_bytes(), object_pairs_hook=unique_object)
    if not isinstance(payloads, dict) or set(payloads) != {case.spec.row_id for case in cases}:
        raise ProtocolError("payload_ids_differ_from_case_manifest")
    mismatches: Counter[str] = Counter()
    intact = []
    for plan in plans:
        spec = plan.spec
        payload = payloads[spec.row_id]
        if set(payload) != {"context", "target"}:
            raise ProtocolError("unexpected_case_payload_fields")
        fields = set()
        try:
            # The source itself is independently pinned and checked by the guard.
            guard._read_source(plan, None)
        except AdmissionSourceError as error:
            if error.code != "source_sha256_mismatch":
                raise
            fields.add("source")
        for field in ("context", "target"):
            if sha256_json(payload[field]) != getattr(spec, field + "_sha256"):
                fields.add(field)
        if fields:
            mismatches.update(fields)
            mismatches["cases"] += 1
        else:
            intact.append((spec, payload))
    if mismatches["cases"] * 100 > 5 * len(cases):
        return [], mismatches

    prepared = []
    for spec, payload in intact:
        context = PromptContext.from_mapping(payload["context"])
        target = payload["target"]
        if set(target) != {"operation", "region_new"} or not isinstance(target["region_new"], list):
            raise ProtocolError("invalid_target_shape")
        row = build_training_row(
            context, operation=target["operation"], region_new=target["region_new"],
            tokenizer=tokenizer, row_id=spec.row_id, family=spec.family,
            package_id=spec.package_id, split="final_locked_v1",
        )
        validate_development_capacity(row, CAPACITY, CAP, spec.row_id)
        prepared.append(PreparedCase(
            spec, context, target["region_new"], row["input_ids"][:row["target_start"]],
            spec.group_id in overlap,
        ))
    return prepared, mismatches


def percentile_interval(samples: list[float]) -> list[float]:
    ordered = sorted(samples)
    return [ordered[len(ordered) // 40 - 1], ordered[39 * len(ordered) // 40 - 1]]


def summarize(
    cases: list[PreparedCase], model_counts: dict[str, dict[str, Counter[str]]], subgroup: bool,
) -> Json:
    selected = [case for case in cases if case.overlapping == subgroup]
    packages = Counter(package_identity(case.spec.package_id) for case in selected)
    n = len(selected)
    if not n:
        return {"cases": 0, "packages": 0, "models": {}, "comparisons": {}}

    # All models use the same resampled package counts, so differences are paired.
    keys = sorted(packages)
    rng = random.Random(SEED)
    samples: dict[str, list[float]] = {label: [] for label in model_counts}
    clusters = [
        (packages[key], [counts[key]["correct"] for counts in model_counts.values()])
        for key in keys
    ]
    for _ in range(BOOTSTRAP_REPS):
        sample = [clusters[rng.randrange(len(clusters))] for _ in keys]
        denominator = sum(size for size, _ in sample)
        for column, rates in enumerate(samples.values()):
            rates.append(sum(correct[column] for _, correct in sample) / denominator)

    models: dict[str, Json] = {}
    for label, by_package in model_counts.items():
        counts: Counter[str] = Counter()
        for key in keys:
            counts.update(by_package[key])
        models[label] = {
            "counts": {metric: counts[metric] for metric in (
                "correct", "edits", "edit_exact", "noop", "noop_correct",
                "noop_false_suggestions", "protocol_valid", "cap_hit", "response_complete",
            )},
            "rate": counts["correct"] / n,
            "ci95": percentile_interval(samples[label]),
        }

    comparisons = {}
    for candidate, baseline in COMPARISONS:
        if candidate in models and baseline in models:
            deltas = [a - b for a, b in zip(samples[candidate], samples[baseline], strict=True)]
            comparisons[f"{candidate}_minus_{baseline}"] = {
                "delta": models[candidate]["rate"] - models[baseline]["rate"],
                "ci95": percentile_interval(deltas),
            }
    return {"cases": n, "packages": len(keys), "models": models, "comparisons": comparisons}


def score(
    cases: list[PreparedCase], labels: list[str], tokenizer: DevTokenizer,
    generators: Callable[[str], AbstractContextManager[Generate]], completion_pins: Json, plan_sha256: str,
) -> Json:
    from runtime import score_case

    categories: dict[bool, dict[str, dict[str, Counter[str]]]] = {False: {}, True: {}}
    expected_ids = {case.spec.row_id for case in cases}

    def score_model(label: str, completed: Iterable[tuple[PreparedCase, list[int]]]) -> None:
        for category in categories.values():
            category[label] = {}
        for case, ids in completed:
            counts = categories[case.overlapping][label].setdefault(
                package_identity(case.spec.package_id), Counter(),
            )
            counts.update(score_case(case, ids, tokenizer))

    for label in labels:
        if completion_pins:
            artifact = read_pin(completion_pins[label])
            if (artifact.get("schema") != SCHEMA + ".completions" or artifact.get("model") != label
                    or artifact.get("plan_sha256") != plan_sha256
                    or set(artifact["completions"]) != expected_ids):
                raise ProtocolError("resume_completion_binding_mismatch")
            score_model(label, ((case, artifact["completions"][case.spec.row_id]) for case in cases))
        else:
            with generators(label) as generate:
                score_model(label, ((case, generate(case.prompt)) for case in cases))
    return {
        "clean": summarize(cases, categories[False], False),
        "overlapping": summarize(cases, categories[True], True),
    }


def run(args: argparse.Namespace) -> Json:
    from runtime import FrozenRuntime

    freeze = pinned_json(args.freeze_receipt, args.freeze_sha256)
    if freeze.get("schema") != SCHEMA + ".freeze":
        raise ProtocolError("freeze_schema_mismatch")
    plan_sha256 = freeze["plan_sha256"]
    plan = pinned_json(args.plan, plan_sha256)
    if plan.get("schema") != SCHEMA + ".plan":
        raise ProtocolError("plan_schema_mismatch")
    labels = list(MODELS)
    if OPTIONAL_MODEL in plan["models"]:
        if not isinstance(plan.get("c20_adoption"), dict):
            raise ProtocolError("c20_adoption_receipt_required")
        adoption = read_pin(plan["c20_adoption"])
        if adoption.get("status") != "adopted" or adoption.get("model") != plan["models"][OPTIONAL_MODEL]:
            raise ProtocolError("c20_adoption_not_bound")
        labels.append(OPTIONAL_MODEL)
    if set(plan["models"]) != set(labels):
        raise ProtocolError("unexpected_model_roster")
    guard = AdmissionGuard(
        allowed_source_roots=[Path(path) for path in plan["source_roots"]],
        train_identities=(), dev_identities=(),  # v2 uses the group policy in audit().
        freeze_receipt=freeze, semantic_receipt={},
        expected_weights_sha256=sha256_json(plan["models"]),
        expected_harness_sha256=harness_hash(),
        expected_source_lock_sha256=plan["cases"]["sha256"],
        observed_at=datetime.now(timezone.utc),
    )
    guard._validate_freeze_receipt()

    state = Path(plan["state_dir"])
    if not state.is_absolute() or state.resolve(strict=True) != state:
        raise ProtocolError("state_dir_must_be_absolute_existing_real_directory")
    descriptor = os.open(state, os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return evaluate_locked(args, plan, plan_sha256, guard, labels, state, FrozenRuntime)
    finally:
        os.close(descriptor)


def evaluate_locked(
    args: argparse.Namespace, plan: Json, plan_sha256: str, guard: AdmissionGuard,
    labels: list[str], state: Path, runtime_factory: RuntimeFactory,
) -> Json:
    start = state / "started.json"
    result = state / "result.json"
    resumed = state / "resumed.json"
    binding = {"schema": SCHEMA, "plan_sha256": plan_sha256, "freeze_sha256": args.freeze_sha256}
    if os.path.lexists(result) or os.path.lexists(resumed):
        raise ProtocolError("run_already_closed")

    completion_pins = {}
    if args.resume:
        resume = pinned_json(args.resume, args.resume_sha256)
        if (resume.get("schema") != SCHEMA + ".resume" or resume.get("authorization") != "user-approved"
                or resume.get("plan_sha256") != plan_sha256 or not start.is_file()
                or pinned_json(start, resume["started_sha256"]) != binding):
            raise ProtocolError("resume_receipt_mismatch")
        completion_pins = resume["completions"]
        if set(completion_pins) != set(labels):
            raise ProtocolError("resume_requires_pinned_completions_for_every_model")
    elif any(state.iterdir()):
        raise ProtocolError("run_requires_empty_state_directory")

    # Pin and validate the model/runtime artifacts before the run takes the seal.
    runtime = runtime_factory(plan, replay=bool(completion_pins))
    memberships = read_pin(plan["memberships"])
    if args.resume:
        write_once(resumed, {**binding, "resume_sha256": args.resume_sha256})
    else:
        write_once(start, binding)

    cases = load_cases(read_pin(plan["cases"]), plan["case_count"])
    guard.semantic_receipt = read_pin(plan["semantic_receipt"])
    guard._validate_semantic_receipt([case.spec for case in cases])
    overlap, audit_counts = audit(cases, memberships)
    report = {**binding, "audit": audit_counts}
    if len(overlap) * 100 > 20 * audit_counts["groups"]:
        report.update(status="aborted", reason="overlap_ceiling_exceeded")
    else:
        prepared, mismatches = construct(cases, Path(plan["payloads_path"]), guard, overlap, runtime.tokenizer)
        audit_counts["hash_mismatches"] = {name: mismatches[name] for name in ("cases", "source", "context", "target")}
        if mismatches["cases"] * 100 > 5 * len(cases):
            report.update(status="aborted", reason="mismatch_ceiling_exceeded")
        elif not any(not case.overlapping for case in prepared):
            report.update(status="aborted", reason="no_clean_cases")
        else:
            report.update(status="complete", **score(
                prepared, labels, runtime.tokenizer, runtime.generator, completion_pins, plan_sha256,
            ))
    write_once(result, report)
    return report


def unique_object(pairs: list[tuple[str, object]]) -> Json:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ProtocolError("duplicate_json_key")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    if bool(args.resume) != bool(args.resume_sha256):
        parser.error("--resume and --resume-sha256 must be supplied together")
    try:
        report = run(args)
    except Exception:  # noqa: BLE001 - a runtime exception may include sealed text
        print("Sealed final v2 stopped. Inspect run-state markers and consult the freeze owner.", file=sys.stderr)
        return 2
    print(report["status"], report.get("reason", "result.json"))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.modules["protocol"] = sys.modules[__name__]
    raise SystemExit(main())
