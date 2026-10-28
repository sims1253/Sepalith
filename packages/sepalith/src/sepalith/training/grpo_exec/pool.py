#!/usr/bin/env python3
"""Card C18 pool: execution-graded training tasks, a validation split and the SFT replay rows (CPU only).

Three stages, each writing new files only:

1. ``build`` (expensive, about an hour at 8 R workers): every gated training function of slices
   S000..S017 goes through the exec-eval task builder (``exec_eval.jobs.build_tasks`` ->
   ``exec_env.tasks.build_for_function``: synth finish_block cuts at every depth with the TRAIN
   prompt variants, TRAIN-geometry rename_propagation, PRM03 rendering and the per-cut observability,
   local-mutant and inferability gates). No per-function cap here (all usable cuts are kept), so the
   split can be re-cut without rebuilding. A function is used only if its ``gate.json`` passed at the
   pinned gate (commit c631141, fingerprint 7f8e073fbedb4b9d, sources unmodified), is newer than every
   input it judged, reports no stale or killed equivalent declarations, has no ``SKIP`` file, and is
   neither a frozen hold-out function nor from a hold-out package. Cuts whose hidden remainder touches
   a branch the function's ``equivalent.txt`` declares untested (``SCOPED``/``BUGFIX`` conditions, as in
   the hold-out freeze) are dropped by the builder.
2. ``split`` (seconds): whole packages go to the validation split, in sha256 order of the package
   name, until it holds at least ``--val-share`` of the functions with tasks. The training split keeps
   at most ``--train-per-function`` tasks per function and validation at most ``--val-per-function``,
   both round-robin over (family, cut length) strata (``exec_eval.jobs.select_tasks``). The frozen
   hold-out is never read for tasks; its manifest is read only to exclude its functions and packages.
3. ``replay`` (a minute): a seeded sample of the v4 TRAIN rows arm A was trained on (prompt <= 3,072,
   target <= 1,024 tokens), as prompt and target token IDs, for the SFT cross-entropy replay term.

The ``split`` stage writes the pool manifest (counts by family, cut length and split; file hashes)
that the bound recipe pins.

    nice -n 10 .venv-sft/bin/python -m sepalith.training.grpo_exec.pool build --out-dir <DATA>/all
    nice -n 10 .venv-sft/bin/python -m sepalith.training.grpo_exec.pool split --all <DATA>/all/tasks.jsonl \\
        --out-dir <DATA> --manifest docs/training/plan-20260927/c18/C18-exec-pool.json
    nice -n 10 .venv-sft/bin/python -m sepalith.training.grpo_exec.pool replay --out <DATA>/sft-replay.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import re
from typing import cast

Json = dict[str, object]

EXEC_DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1")
OUT_DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/C18-grpo-exec-v1")
FREEZE_MANIFEST = EXEC_DATA / "holdout-freeze-v1.json"
FREEZE_SHA256 = "9e74c1644c96576693b48a5fb37cfdd5af93c7501c505943a5fc31852e0eca19"
SLICES_FILE = EXEC_DATA / "slices.json"
GATE_COMMIT = "c631141b0b58cbd9135a424c2cd59f1305fe634e"
GATE_FINGERPRINT = "7f8e073fbedb4b9d"
TRAIN_SLICES = "S000..S017"
GATE_INPUTS = ("source.R", "meta.json", "test.R", "equivalent.txt")
MANIFEST_SCHEMA = "sepalith.c18.exec-pool.v1"
REPLAY_SCHEMA = "sepalith.c18.sft-replay-row.v1"
SPLIT_SEED = "c18-val-split-v1"
SEALED_MARKERS = ("DAT-07-final", "sealed-final", "DAT-08")
PROMPT_MAX, COMPLETION_MAX = 3072, 1024


class PoolError(ValueError):
    """An input differs from its pin or a split invariant fails."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise PoolError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[Json]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> str:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return sha256(path)


def write_json_new(path: Path, value: object) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def no_sealed(*paths: Path) -> None:
    for path in paths:
        require(not any(marker in str(path) for marker in SEALED_MARKERS), f"refusing a sealed final-set path: {path}")


# Hold-out exclusion --------------------------------------------------------------------------------

def holdout_identity(freeze: Path = FREEZE_MANIFEST, slices: Path = SLICES_FILE) -> tuple[set[str], set[str]]:
    """(function ids, package names) of the frozen hold-out and every hold-out slice; never task content."""
    require(sha256(freeze) == FREEZE_SHA256, "hold-out freeze manifest differs from its pin")
    manifest = json.loads(freeze.read_text())
    functions = {str(r["function"]) for r in manifest["functions"]}
    packages = {str(r["package"]) for r in manifest["functions"] if r.get("package")}
    layout = json.loads(slices.read_text())
    for entry in layout["slices"]:
        if entry.get("role") == "holdout":
            packages.update(str(p) for p in entry.get("packages", []))
            functions.update(str(f) for f in entry.get("function_ids", []))
    return functions, packages


# Function selection ---------------------------------------------------------------------------------

def untested_mutants(directory: Path) -> set[str]:
    """Mutant ids whose ``equivalent.txt`` declaration is conditioned SCOPED/BUGFIX (an untested broken branch)."""
    from sepalith.exec_env.gate import parse_declaration

    path = directory / "equivalent.txt"
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or parts[0].startswith("#"):
            continue
        if parse_declaration(parts[1])["cond"] and re.search(r"\b(SCOPED|BUGFIX)\[", parts[1]):
            out.add(parts[0])
    return out


def function_status(directory: Path, held_functions: set[str], held_packages: set[str]) -> tuple[str, Json]:
    """('used', meta) or (reason, meta) for one function directory."""
    meta = json.loads((directory / "meta.json").read_text()) if (directory / "meta.json").is_file() else {}
    if (directory / "SKIP").exists():
        return "skipped", meta
    if directory.name in held_functions:
        return "holdout_function", meta
    if str(meta.get("package")) in held_packages:
        return "holdout_package", meta
    gate_path = directory / "gate.json"
    if not gate_path.is_file() or not (directory / "test.R").is_file():
        return "not_gated", meta
    gate = json.loads(gate_path.read_text())
    version = gate.get("gate_version") or {}
    if version.get("commit") != GATE_COMMIT or version.get("fingerprint") != GATE_FINGERPRINT or version.get("exec_env_modified"):
        return "gated_at_other_version", meta
    if gate.get("verdict") != "pass":
        return "gate_not_pass", meta
    mutants = gate.get("mutants") or {}
    if isinstance(mutants, dict) and (mutants.get("stale_equivalent_ids") or mutants.get("declared_equivalent_but_killed")):
        return "stale_equivalents", meta
    gated_at = gate_path.stat().st_mtime
    inputs = [directory / name for name in GATE_INPUTS if (directory / name).exists()]
    inputs += sorted((directory / "probes").glob("*.R")) if (directory / "probes").is_dir() else []
    if any(path.stat().st_mtime > gated_at + 1 for path in inputs):
        return "changed_after_gating", meta
    return "used", meta


def expand_slices(spec: str) -> list[str]:
    from sepalith.exec_eval.jobs import expand_slices as expand

    return expand(spec)


def training_directories(root: Path, slices: Sequence[str]) -> tuple[list[tuple[str, Path, set[str]]], Json]:
    held_functions, held_packages = holdout_identity()
    out: list[tuple[str, Path, set[str]]] = []
    counts: Counter = Counter()
    untested = 0
    for slice_id in slices:
        require(re.fullmatch(r"S\d{3}", slice_id), f"{slice_id} is not a training slice")
        for directory in sorted(p for p in (root / slice_id).iterdir() if (p / "source.R").is_file()):
            status, _meta = function_status(directory, held_functions, held_packages)
            counts[status] += 1
            if status == "used":
                ids = untested_mutants(directory)
                untested += len(ids)
                out.append((slice_id, directory, ids))
    return out, {"kind": "c18-train", "root": str(root), "slices": list(slices), "function_counts": dict(sorted(counts.items())),
                 "untested_branch_declarations": untested, "holdout_functions_excluded_by_id": len(held_functions),
                 "holdout_packages_excluded": len(held_packages),
                 "freeze_manifest": {"path": str(FREEZE_MANIFEST), "sha256": FREEZE_SHA256, "read_for": "exclusion only"}}


def build(out_dir: Path, *, root: Path = EXEC_DATA / "authors", slices: str = TRAIN_SLICES, workers: int = 8,
          families: str = "finish_block,rename_propagation") -> Json:
    from sepalith.exec_eval.jobs import build_tasks

    no_sealed(out_dir, root)
    require(not (out_dir / "tasks.jsonl").exists(), f"{out_dir}/tasks.jsonl exists")
    directories, source = training_directories(root, expand_slices(slices))
    receipt = build_tasks(directories, out_dir, 10**6, min(workers, 8), set(families.split(",")), source)
    return {k: v for k, v in receipt.items() if k != "builder_reasons"}


# Split ------------------------------------------------------------------------------------------------

def bucket_of(task: Mapping[str, object]) -> str:
    return str(cast(Json, task.get("curriculum") or {}).get("bucket") or "short")


def stratum(task: Mapping[str, object]) -> str:
    return f"{task['family']}:{bucket_of(task)}"


def validation_packages(function_packages: Mapping[str, str], share: float, seed: str = SPLIT_SEED) -> set[str]:
    """Whole packages in sha256(seed:package) order until they hold at least ``share`` of the functions."""
    require(0 < share < 0.5, "the validation share must lie in (0, 0.5)")
    by_package: dict[str, int] = Counter(function_packages.values())
    target = share * len(function_packages)
    chosen: set[str] = set()
    held = 0
    for package in sorted(by_package, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()):
        if held >= target:
            break
        chosen.add(package)
        held += by_package[package]
    return chosen


def split_tasks(tasks: Sequence[Json], *, val_share: float, train_per_function: int, val_per_function: int,
                held_functions: set[str], held_packages: set[str]) -> tuple[list[Json], list[Json], Json]:
    from sepalith.exec_eval.jobs import select_tasks

    require(all(str(t["function_id"]) not in held_functions for t in tasks), "a hold-out function reached the task file")
    require(all(str(t["package"]) not in held_packages for t in tasks), "a hold-out package reached the task file")
    by_function: dict[str, list[Json]] = defaultdict(list)
    packages: dict[str, str] = {}
    for task in tasks:
        by_function[str(task["function_id"])].append(task)
        packages[str(task["function_id"])] = str(task["package"])
    val_packages = validation_packages(packages, val_share)
    train: list[Json] = []
    val: list[Json] = []
    for function_id in sorted(by_function):
        items = by_function[function_id]
        if packages[function_id] in val_packages:
            val += select_tasks(items, val_per_function, f"c18-val:{function_id}")
        else:
            train += select_tasks(items, train_per_function, f"c18-train:{function_id}")
    train.sort(key=lambda t: (str(t["function_id"]), str(t["id"])))
    val.sort(key=lambda t: (str(t["function_id"]), str(t["id"])))
    train_functions = {str(t["function_id"]) for t in train}
    val_functions = {str(t["function_id"]) for t in val}
    require(not train_functions & val_functions, "a function is in both splits")
    require(not {str(t["package"]) for t in train} & {str(t["package"]) for t in val}, "a package is in both splits")
    require(len({str(t["id"]) for t in train + val}) == len(train) + len(val), "task ids repeat")
    return train, val, {"validation_packages": sorted(val_packages), "functions_with_tasks": len(by_function),
                        "tasks_available": len(tasks)}


def stats(tasks: Sequence[Json]) -> Json:
    functions_by: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        functions_by[stratum(task)].add(str(task["function_id"]))
        functions_by[f"family:{task['family']}"].add(str(task["function_id"]))
    prompt = sorted(int(cast(int, cast(Json, t["curriculum"])["prompt_tokens"])) for t in tasks)
    gold = sorted(int(cast(int, cast(Json, t["curriculum"])["completion_tokens"])) for t in tasks)
    return {"tasks": len(tasks), "functions": len({str(t["function_id"]) for t in tasks}),
            "packages": len({str(t["package"]) for t in tasks}),
            "tasks_by_stratum": dict(sorted(Counter(stratum(t) for t in tasks).items())),
            "tasks_by_family": dict(sorted(Counter(str(t["family"]) for t in tasks).items())),
            "functions_by_stratum": {k: len(v) for k, v in sorted(functions_by.items()) if not k.startswith("family:")},
            "functions_by_family": {k.removeprefix("family:"): len(v) for k, v in sorted(functions_by.items()) if k.startswith("family:")},
            "prompt_tokens": {"mean": round(sum(prompt) / len(prompt), 1), "median": prompt[len(prompt) // 2], "max": prompt[-1]}
            if prompt else None,
            "gold_completion_tokens": {"mean": round(sum(gold) / len(gold), 1), "median": gold[len(gold) // 2], "max": gold[-1]}
            if gold else None}


def census_overlap(train: Sequence[Json], val: Sequence[Json], census_records: Path | None) -> Json | None:
    """How the arm A 536 census (k = 8, T = 1.0) covers each split, as curriculum context only."""
    if census_records is None or not census_records.is_file():
        return None
    rewards: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(census_records):
        rewards[str(row["task_id"])].append(float(cast(float, row["reward"])))

    def block(tasks: Sequence[Json]) -> Json:
        seen = [rewards[str(t["id"])] for t in tasks if str(t["id"]) in rewards]
        zero = sum(1 for r in seen if max(r) - min(r) <= 1e-9)
        return {"tasks_in_census": len(seen), "zero_reward_variance_at_k8": zero,
                "zero_variance_share": round(zero / len(seen), 4) if seen else None}
    return {"records": str(census_records), "sha256": sha256(census_records), "train": block(train), "validation": block(val),
            "use": "context only: the census covers a small share of the pool, so zero-variance tasks are left to dynamic sampling"}


def split(all_tasks: Path, out_dir: Path, manifest_path: Path, *, val_share: float = 0.10, train_per_function: int = 8,
          val_per_function: int = 3, census_records: Path | None = None) -> Json:
    no_sealed(all_tasks, out_dir, manifest_path)
    receipt_path = all_tasks.with_name("tasks-receipt.json")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("tasks_sha256") == sha256(all_tasks), "the built task file differs from its receipt")
    require(dict(receipt.get("gate_version") or {}).get("fingerprint") == GATE_FINGERPRINT, "tasks were built at another gate")
    held_functions, held_packages = holdout_identity()
    tasks = read_jsonl(all_tasks)
    train, val, info = split_tasks(tasks, val_share=val_share, train_per_function=train_per_function,
                                   val_per_function=val_per_function, held_functions=held_functions, held_packages=held_packages)
    train_sha = write_jsonl_new(out_dir / "train-tasks.jsonl", train)
    val_sha = write_jsonl_new(out_dir / "val-tasks.jsonl", val)
    manifest: Json = {
        "schema": MANIFEST_SCHEMA, "card": "C18",
        "built": {"path": str(all_tasks), "sha256": receipt["tasks_sha256"], "receipt": str(receipt_path),
                  "receipt_sha256": sha256(receipt_path), "n_tasks": receipt["n_tasks"], "n_functions": receipt["n_functions"],
                  "source": receipt["source"], "builder_reasons": receipt.get("builder_reasons"),
                  "gate_version": receipt["gate_version"], "builder": receipt["builder"]},
        "rule": {"validation": f"whole packages in sha256('{SPLIT_SEED}:' + package) order until >= {val_share:.0%} of the "
                               "functions with tasks",
                 "train_per_function": train_per_function, "val_per_function": val_per_function,
                 "per_function_selection": "exec_eval.jobs.select_tasks: round-robin over (family, cut length) strata, seeded "
                                           "by function id",
                 "holdout": "frozen hold-out functions and every hold-out-slice package excluded (manifest read for ids only)",
                 "untested_branches": "cuts touching a SCOPED/BUGFIX-declared branch dropped at build time"},
        "train": {"path": str(out_dir / "train-tasks.jsonl"), "sha256": train_sha, **stats(train)},
        "validation": {"path": str(out_dir / "val-tasks.jsonl"), "sha256": val_sha, **stats(val),
                       "package_names": info["validation_packages"]},
        "functions_with_tasks": info["functions_with_tasks"], "tasks_available": info["tasks_available"],
        "census_context": census_overlap(train, val, census_records),
        "final_set_access": False,
    }
    write_json_new(manifest_path, manifest)
    return {"manifest": str(manifest_path), "sha256": sha256(manifest_path),
            "train": {k: manifest["train"][k] for k in ("tasks", "functions", "packages", "tasks_by_stratum")},  # type: ignore[index]
            "validation": {k: manifest["validation"][k] for k in ("tasks", "functions", "packages", "tasks_by_stratum")}}  # type: ignore[index]


# SFT replay rows --------------------------------------------------------------------------------------

def replay_rows(rows_path: Path, expected_sha: str, count: int, seed: int = 3407) -> tuple[list[Json], Json]:
    """A seeded sample of TRAIN rows (prompt and target token IDs) within the GRPO length caps."""
    require(sha256(rows_path) == expected_sha, f"TRAIN rows {rows_path} differ from their pin")
    eligible: list[Json] = []
    seen = 0
    with rows_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            seen += 1
            require(row.get("split") == "train", f"row {row.get('id')} is not TRAIN")
            ids, start = row["input_ids"], int(row["target_start"])
            prompt, target = ids[:start], ids[start:]
            if len(prompt) <= PROMPT_MAX and 0 < len(target) <= COMPLETION_MAX and target[-1] == 1 and prompt[0] == 0:
                eligible.append({"schema": REPLAY_SCHEMA, "id": str(row["id"]), "family": str(row["family"]),
                                 "prompt_ids": prompt, "target_ids": target})
    require(len(eligible) >= count, f"only {len(eligible)} eligible TRAIN rows for {count}")
    rng = random.Random(seed)
    chosen = sorted(rng.sample(eligible, count), key=lambda r: str(r["id"]))
    return chosen, {"train_rows": seen, "eligible": len(eligible), "sampled": count, "seed": seed,
                    "families": dict(sorted(Counter(str(r["family"]) for r in chosen).items())),
                    "target_tokens_mean": round(sum(len(cast(list, r["target_ids"])) for r in chosen) / count, 1),
                    "prompt_tokens_mean": round(sum(len(cast(list, r["prompt_ids"])) for r in chosen) / count, 1)}


def replay(out: Path, count: int) -> Json:
    from sepalith.training.rl.length_policy import TRAIN_ROWS_V4_BRACEINDENT_SHA256
    from sepalith.training.rl.reward_v21 import DATA_ROOT, TRAIN_INPUTS

    rows_path = DATA_ROOT / TRAIN_INPUTS["rows"][0]
    no_sealed(rows_path, out)
    rows, info = replay_rows(rows_path, TRAIN_ROWS_V4_BRACEINDENT_SHA256, count)
    digest = write_jsonl_new(out, rows)
    receipt = {"path": str(out), "sha256": digest, "rows": len(rows), "source": str(rows_path),
               "source_sha256": TRAIN_ROWS_V4_BRACEINDENT_SHA256, **info}
    write_json_new(out.with_suffix(".receipt.json"), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", type=Path, default=OUT_DATA / "all")
    b.add_argument("--root", type=Path, default=EXEC_DATA / "authors")
    b.add_argument("--slices", default=TRAIN_SLICES)
    b.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("split")
    s.add_argument("--all", type=Path, default=OUT_DATA / "all" / "tasks.jsonl")
    s.add_argument("--out-dir", type=Path, default=OUT_DATA)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--val-share", type=float, default=0.10)
    s.add_argument("--train-per-function", type=int, default=8)
    s.add_argument("--val-per-function", type=int, default=3)
    s.add_argument("--census-records", type=Path)
    r = sub.add_parser("replay")
    r.add_argument("--out", type=Path, default=OUT_DATA / "sft-replay.jsonl")
    r.add_argument("--rows", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.out_dir, root=args.root, slices=args.slices, workers=args.workers)
    elif args.command == "split":
        result = split(args.all, args.out_dir, args.manifest, val_share=args.val_share,
                       train_per_function=args.train_per_function, val_per_function=args.val_per_function,
                       census_records=args.census_records)
    else:
        result = replay(args.out, args.rows)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
