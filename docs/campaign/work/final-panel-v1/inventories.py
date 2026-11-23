#!/usr/bin/env python3
"""Build the sealed-final v2 membership inventories and model/runtime pins. Reads no final content.

Writes only into a mode-700 state directory (default
``~/.local/state/sepalith/sealed-final-v2``), never into git:

    python inventories.py inventories [--state DIR]   # memberships.json + coverage receipt
    python inventories.py pins        [--state DIR]   # model, tokenizer and runtime pins

Rules (sealed-final-v2.md, final-panel-v1.md):

* ``train`` covers every DAT-02 v2 group whose split is neither ``dev_group``
  nor ``final_candidate_group``, plus every row that trained a scored model.
  All identity forms go to ``split_identities``; ``pkg:`` forms also give
  ``package_ids``.
* DEV75 is the ``bf278755…`` file recorded in the DAT-07 registry preflight.
  Only identity keys are kept, and the file is not rehashed here.
* ``exec_holdout`` comes from the hold-out-role entries of
  ``EXEC-ENV-v1/slices.json`` only (function IDs and package names). No
  hold-out task, pool row or frozen manifest is opened.
* Each stage records its sources, rule and expected-versus-observed coverage
  in ``coverage.json``, separate from ``memberships.json`` so the launcher's
  strict five-key schema holds.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

Json = dict[str, Any]
REPO = Path(__file__).resolve().parents[4]
DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work")
CHECKPOINTS = Path("/mnt/e/sepalith/campaign-20260915/checkpoints")
QUANT = Path("/mnt/e/sepalith/campaign-20260915/quant-work/c19-100")
STATE = Path.home() / ".local/state/sepalith/sealed-final-v2"

SPLIT = DATA / "DAT-02-global-split-v2.json"
SPLIT_SHA256 = "c4e1219a494372a32f5d657a8454801128a2aede96440359679905f7051a8f09"
NOT_TRAIN = ("dev_group", "final_candidate_group")
TRAIN_ROWS = DATA / "DAT10-finish-source-repair-v4-braceindent/train-token-rows.jsonl"
SFT_REPLAY = DATA / "C18-grpo-exec-v1/sft-replay.jsonl"
DEV75 = DATA / "DAT-07-final-dev-candidates.jsonl"
DEV75_SHA256 = "bf278755c795b51ba58bc14c674edf76dcb7799475e5982adc7bd50f9c0d05df"  # registry preflight
DEV250 = DATA / "C05-dev250-v1r3/dev250-v1.jsonl"
CPT_ROWS = DATA / "CPT-prefix-extension-v1/combined-cpt-train-ctx16384.jsonl"
CPT_REPLAY = DATA / "C18-grpo-exec-v1/cpt-replay.jsonl"
C14_POOL = REPO / "docs/training/plan-20260927/c14/pool/C14-grpo-pool-arm-a-536.jsonl"
SYNTH_CASES = tuple(DATA / f"SYNTH-{v}/cases.jsonl" for v in ("v1", "v2", "v3", "v3.1"))
EXEC_SLICES = DATA / "EXEC-ENV-v1/slices.json"
EXEC_TRAIN_TASKS = (DATA / "C18-grpo-exec-v1/train-tasks.jsonl", DATA / "C19-hard-mix-v1/train.jsonl")
EXEC_VAL_TASKS = DATA / "C18-grpo-exec-v1/val-tasks.jsonl"
STAGES = ("train", "dev", "cpt", "c14", "synth", "exec_train", "exec_validation", "exec_holdout")
KEYS = ("row_ids", "package_ids", "group_ids", "split_identities", "function_ids")

MODELS = {
    "arm-a-803": CHECKPOINTS / "SFT-v2-r4-lr1e-5/full/checkpoint-803",
    "c19-100-bf16": CHECKPOINTS / "C19-grpo-exec-pilot-1/full/checkpoint-100",
}
Q6_K = (QUANT / "c19-100-imat-Q6_K.gguf", "6593785330b99076b80ae3ac679bda76c5fcff1ebd1873f46f944c1ec43a3261")
MIXB_E4 = (QUANT / "c19-100-mixb-e4-Q4_K_M.gguf", "e24fbfbc5ea47c88647445ec682196fc67af9134d4fa36e826beeda045f1c71b")
CAMPAIGN_TOKENIZER = Path("/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
LLAMA_SERVER = Path.home() / ".local/opt/llama.cpp-b10453/build-cpu/bin/llama-server"
LLAMA_COMMIT = "3cb7ffb1a"
DISTRIBUTIONS = ("torch", "transformers", "tokenizers", "safetensors", "accelerate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> Iterator[Json]:
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fields(path: Path, names: Iterable[str]) -> Iterator[dict[str, str | None]]:
    """Top-level string fields of JSONL rows.

    A regex pass is used only when each field occurs exactly once in the line, so a nested key
    (sort_keys rows put context and exec ahead of the identity keys) can never be picked up;
    otherwise the line is parsed in full.
    """
    names = tuple(names)
    patterns = {name: re.compile(rb'"%s":\s*"([^"\\]*)"' % name.encode()) for name in names}
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            found = {name: rx.findall(line) for name, rx in patterns.items()}
            if all(len(m) == 1 and line.count(b'"%s":' % n.encode()) == 1 for n, m in found.items()):
                yield {name: m[0].decode() for name, m in found.items()}
            else:
                row = json.loads(line)
                yield {name: row.get(name) if isinstance(row.get(name), str) else None for name in names}


def complete(rows: Iterable[dict[str, str | None]], path: Path) -> list[dict[str, str | None]]:
    """Reject a source with any row that lacks a required identity field."""
    rows = list(rows)
    if not rows or any(value is None for row in rows for value in row.values()):
        raise ValueError(f"missing identity fields in {path}")
    return rows


def membership(rows: Iterable[object] = (), packages: Iterable[object] = (), groups: Iterable[object] = (),
               splits: Iterable[object] = (), functions: Iterable[object] = ()) -> Json:
    value = {}
    for key, items in zip(KEYS, (rows, packages, groups, splits, functions), strict=True):
        value[key] = sorted({i for i in items if isinstance(i, str) and i and i.strip() == i and "\x00" not in i})
    return value


def split_groups(split: Json, keep: Callable[[str], bool]) -> tuple[list[str], set[str], set[str]]:
    chosen = [group for group in split["groups"] if keep(group["split"])]
    packages = {form[4:] for group in chosen for form in group["identity_forms"] if form.startswith("pkg:")}
    forms = {form for group in chosen for form in group["identity_forms"]}
    return [group["group_id"] for group in chosen], packages, forms



def build_inventories() -> tuple[Json, Json]:
    if sha256_file(SPLIT) != SPLIT_SHA256:
        raise ValueError("DAT-02 split differs from its pin")
    split = json.loads(SPLIT.read_bytes())
    inventories: Json = {}
    coverage: Json = {}

    def record(stage: str, value: Json, sources: list[Path], rule: str, counts: Json,
               pinned: dict[str, str] | None = None) -> None:
        inventories[stage] = value
        coverage[stage] = {
            "sources": [{"path": str(p), **({"sha256_pinned_not_rehashed": (pinned or {})[str(p)]}
                                            if str(p) in (pinned or {}) else {"sha256": sha256_file(p)})}
                        for p in sources],
            "rule": rule,
            "counts": {**counts, **{key: len(value[key]) for key in KEYS}},
        }

    # TRAIN: every DAT-02 label except dev and final, plus every SFT row and the C18/C19 SFT replay.
    groups, packages, forms = split_groups(split, lambda label: label not in NOT_TRAIN)
    labels = sorted({g["split"] for g in split["groups"]} - set(NOT_TRAIN))
    train_rows = {row["id"]: row["package_id"] for row in complete(fields(TRAIN_ROWS, ("id", "package_id")),
                                                                    TRAIN_ROWS)}
    replay = {row["id"] for row in complete(fields(SFT_REPLAY, ("id",)), SFT_REPLAY)}
    record("train", membership([*train_rows, *replay], packages | set(train_rows.values()), groups, forms),
           [SPLIT, TRAIN_ROWS, SFT_REPLAY],
           f"DAT-02 v2 groups labelled {labels} (group IDs; pkg: names as packages; all identity forms as "
           "split identities); every Arm A SFT row (DAT10 v4 braceindent) and its package; C18 SFT replay IDs",
           {"dat02_groups": len(groups), "dat02_labels": labels, "sft_rows": len(train_rows),
            "sft_replay_rows": len(replay), "sft_replay_rows_not_in_sft": len(replay - set(train_rows))})

    groups, packages, forms = split_groups(split, lambda label: label == "dev_group")
    dev: dict[str, tuple[str, str]] = {}
    for row in jsonl(DEV75):
        dev[row["row"]["id"]] = (row["row"]["package_id"], row["source_ref"]["group_id"])
    for row in jsonl(DEV250):
        dev[row["id"]] = (row["package_id"], row["group_id"])
    record("dev", membership(dev, packages | {p for p, _ in dev.values()},
                             set(groups) | {g for _, g in dev.values()}, forms),
           [SPLIT, DEV75, DEV250],
           "DAT-02 v2 dev_group groups and identity forms; DEV75 and DEV250 row IDs, packages and groups",
           {"dat02_groups": len(groups), "dev_rows": len(dev)}, pinned={str(DEV75): DEV75_SHA256})

    rows, packages, groups, documents = set(), set(), set(), set()
    cpt_rows_without_package = 0
    for row in fields(CPT_ROWS, ("row_id", "package", "group_id", "document_id")):
        cpt_rows_without_package += row["package"] is None
        rows.add(row["row_id"])
        packages.add(row["package"])
        groups.add(row["group_id"])
        documents.add(row["document_id"])
    replay_rows = list(fields(CPT_REPLAY, ("id", "package")))
    rows |= {r["id"] for r in replay_rows}
    packages |= {r["package"] for r in replay_rows}
    record("cpt", membership(rows, packages, groups, documents), [CPT_ROWS, CPT_REPLAY],
           "every CPT-prefix-extension-v1 combined ctx16384 row (row_id, package, group_id; document_id as "
           "split identity) and the C18 CPT replay rows",
           {"documents": len(documents - {None}), "rows_without_package": cpt_rows_without_package,
            "cpt_replay_rows": len(replay_rows)})

    pool = [row["id"] for row in complete(({"id": r.get("id")} for r in jsonl(C14_POOL)), C14_POOL)]
    record("c14", membership(pool, {train_rows[i] for i in pool if i in train_rows}), [C14_POOL, TRAIN_ROWS],
           "every C14 GRPO pool row; packages joined from the Arm A SFT rows",
           {"pool_rows": len(pool), "rows_without_package": len([i for i in pool if i not in train_rows])})

    synth: dict[str, str] = {}
    for path in SYNTH_CASES:
        synth.update((row["row_id"], row["package_id"])
                     for row in complete(fields(path, ("row_id", "package_id")), path))
    record("synth", membership(synth, set(synth.values())), list(SYNTH_CASES),
           "every SYNTH v1, v2, v3 and v3.1 case row ID and seed package ID", {"cases": len(synth)})

    slices = json.loads(EXEC_SLICES.read_bytes())
    by_role: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for item in slices["slices"]:
        functions, packages, names = by_role.setdefault(item["role"], (set(), set(), set()))
        functions.update(item["function_ids"])
        packages.update(item["packages"])
        names.add(item["slice"])
    tasks: dict[str, tuple[str, str]] = {}
    for path in EXEC_TRAIN_TASKS:
        rows = complete(fields(path, ("id", "package", "function_id")), path)
        tasks.update((r["id"], (r["package"], r["function_id"])) for r in rows)
    hard_mix = json.loads((EXEC_TRAIN_TASKS[1].parent / "train.receipt.json").read_bytes())
    if sum(1 for _ in fields(EXEC_TRAIN_TASKS[1], ("id",))) != sum(hard_mix["family"].values()):
        raise ValueError("C19 hard mix row count differs from its receipt")
    functions, packages, names = by_role["train"]
    record("exec_train", membership(tasks, packages | {p for p, _ in tasks.values()}, (), names,
                                    functions | {f for _, f in tasks.values()}),
           [EXEC_SLICES, *EXEC_TRAIN_TASKS],
           "every train-role EXEC-ENV-v1 slice (function IDs, packages) plus C18 train tasks and the C19 hard mix",
           {"slices": len(names), "tasks": len(tasks), "slice_functions": len(functions),
            "task_functions_outside_train_slices": len({f for _, f in tasks.values()} - functions)})

    val = {r["id"]: (r["package"], r["function_id"])
           for r in complete(fields(EXEC_VAL_TASKS, ("id", "package", "function_id")), EXEC_VAL_TASKS)}
    record("exec_validation", membership(val, {p for p, _ in val.values()}, (), (),
                                         {f for _, f in val.values()}),
           [EXEC_VAL_TASKS], "every C18 validation task (C19 reuses it unchanged)",
           {"tasks": len(val), "val_functions_inside_train_slices":
            len({f for _, f in val.values()} & by_role["train"][0])})

    functions, packages, names = by_role["holdout"]
    record("exec_holdout", membership((), packages, (), names, functions), [EXEC_SLICES],
           "identity only: hold-out-role slice partition metadata (H000-H007 function IDs and packages, a "
           "superset of the 650 frozen functions); no hold-out task, pool row or freeze file opened",
           {"slices": len(names), "slice_functions": len(functions), "expected_functions": slices["holdout"]
            ["functions"], "expected_packages": slices["holdout"]["packages"]})

    for stage in STAGES:
        value = inventories[stage]
        if set(value) != set(KEYS) or not value["package_ids"] or (stage.startswith("exec_")
                                                                    and not value["function_ids"]):
            raise ValueError(f"inventory {stage} fails the launcher's schema")
    return inventories, coverage


# ------------------------------------------------------------------ pins

def file_pin(path: Path, expected: str | None = None) -> Json:
    if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f"pin needs an absolute real regular file: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"hash mismatch for {path}")
    return {"path": str(path), "sha256": digest}


def tree_record(root: Path) -> Json:
    """Every file, in the shape runtime.verify_tree() checks."""
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError(f"tree needs an absolute real directory: {root}")
    files = {}
    for directory, dirs, names in os.walk(root):
        base = Path(directory)
        for name in dirs + names:
            if (base / name).is_symlink():
                raise ValueError(f"symlink in pinned tree: {base / name}")
        for name in names:
            files[str((base / name).relative_to(root))] = sha256_file(base / name)
    if not files:
        raise ValueError(f"empty tree: {root}")
    return {"path": str(root), "files": dict(sorted(files.items()))}


def runtime_pins() -> Json:
    from importlib.metadata import version

    interpreter = Path(sys.executable).resolve(strict=True)
    banner = subprocess.run([str(LLAMA_SERVER), "--version"], capture_output=True, text=True, timeout=60,
                            check=False)
    if LLAMA_COMMIT not in banner.stdout + banner.stderr:
        raise ValueError("llama-server is not the pinned commit")
    return {"python_sha256": sha256_file(interpreter), "python_path": str(interpreter),
            "distributions": {name: version(name) for name in DISTRIBUTIONS},
            "llama_server": {**file_pin(LLAMA_SERVER), "commit": LLAMA_COMMIT}}


def build_pins(state: Path) -> Json:
    tokenizer_dir = state / "tokenizer"
    tokenizer = tree_record(tokenizer_dir)
    for name in TOKENIZER_FILES:
        if tokenizer["files"].get(name) != sha256_file(CAMPAIGN_TOKENIZER / name):
            raise ValueError(f"tokenizer copy differs from the campaign tokenizer: {name}")
    models = {label: tree_record(path) for label, path in MODELS.items()}
    models["c19-100-q6_k"] = file_pin(*Q6_K)
    return {"models": models, "release_artifacts": {"Q4_K_M-mixB-e4": file_pin(*MIXB_E4)},
            "tokenizer": tokenizer, "runtime": runtime_pins()}


# ------------------------------------------------------------------ output

def private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise ValueError(f"{path} must be mode 700")
    return path


def write_once(path: Path, value: object) -> Json:
    data = (json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("inventories", "pins"))
    parser.add_argument("--state", type=Path, default=STATE)
    args = parser.parse_args()
    state = private_dir(args.state.resolve())
    if args.command == "inventories":
        inventories, coverage = build_inventories()
        folder = private_dir(state / "inventories")
        out = {"memberships": write_once(folder / "memberships.json", inventories)}
        out["coverage"] = write_once(folder / "coverage.json", {"memberships_sha256": out["memberships"]["sha256"],
                                                                 "stages": coverage})
        summary = {stage: coverage[stage]["counts"] for stage in STAGES}
    else:
        pins = build_pins(state)
        folder = private_dir(state / "pins")
        out = {name: write_once(folder / f"{name}.json", value) for name, value in pins.items()}
        summary = {"models": {k: len(v.get("files", {})) or 1 for k, v in pins["models"].items()},
                   "runtime": pins["runtime"]["distributions"]}
    write_once(folder / "summary.json", {"outputs": out, "summary": summary})
    print(json.dumps({"outputs": out, "summary": summary}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
