"""Freeze gated exec-env slices (the hold-out) at a pinned gate version.

    PYTHONPATH=packages/sepalith/src python -m sepalith.exec_env.freeze --slices H000..H007 --gate c631141 [--check]

For every slice it verifies, without running anything:

- every function directory either has a ``SKIP`` file (reason recorded) or a
  ``test.R`` with a ``gate.json`` whose verdict is ``pass`` (a gate ``skip``,
  i.e. an output-only method, is recorded as skipped);
- ``gate.json`` was written by the pinned gate: ``gate_version.commit`` starts
  with ``--gate``, ``exec_env_modified`` is false, and the source fingerprint
  equals the fingerprint of the exec_env sources at that commit (recomputed
  from git, exactly as ``gate.gate_version`` computes it);
- ``gate.json`` is newer than every input it judged (``source.R``, ``meta.json``,
  ``test.R``, ``equivalent.txt``, ``probes/*.R``), so no file changed after gating;
- no ``gate.json`` reports stale or killed equivalent declarations.

With ``--check`` it only prints the problems. Otherwise, if there are none, it
writes the frozen manifest (``--out``, default
``EXEC-ENV-v1/holdout-freeze-v1.json``): per function the ids, package and
name, sha256 of ``source.R``, ``meta.json``, ``test.R``, ``frozen.R``,
``equivalent.txt``, every probe and ``gate.json``, the gate summary (assertions,
categories, mutants), and every conditioned equivalence declaration
(``SCOPED``/``BUGFIX``: untested branches that task building must avoid;
``QUIRK``: incidental behaviour); plus the skip list with reasons. It then copies
the slices (function directories and slice reports, not ``.claim``) to
``--copy-to`` (default ``EXEC-ENV-v1/holdout-freeze-v1/``) and removes write
permission from the copy. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

from sepalith.exec_env.gate import parse_declaration

DATA = Path("/mnt/e/sepalith/campaign-20260915/data-work/EXEC-ENV-v1")
SCHEMA = "sepalith.exec-env.holdout-freeze.v1"
LOW_REPLAY_CALLS = 3
KNOWN_LIMITATIONS = [
    "Gate v3's call recorder (r_diff.R) silently drops a test call whose arguments do not evaluate in the "
    "test frame (non-standard evaluation, e.g. an unevaluated `expr`); gate.json records only the calls it "
    "kept (differential.calls), not the dropped ones. Dropped calls are missing from the equivalence replay "
    "and the argument-diversity check. Functions with fewer than 3 recorded calls are listed under "
    "low_replay_coverage as a proxy.",
]
EXEC_ENV = Path(__file__).resolve().parent
INPUTS = ("source.R", "meta.json", "test.R", "equivalent.txt")
HASHED = ("source.R", "meta.json", "test.R", "frozen.R", "equivalent.txt", "gate.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_slices(spec: str) -> list[str]:
    """``H000..H007`` or ``H000,H003`` (or a mix) -> slice ids."""
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        match = re.fullmatch(r"([A-Za-z]+)(\d+)\.\.([A-Za-z]+)?(\d+)", part)
        if match:
            prefix, first, last = match.group(1), match.group(2), match.group(4)
            width = len(first)
            out += [f"{prefix}{n:0{width}d}" for n in range(int(first), int(last) + 1)]
        elif part:
            out.append(part)
    return out


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(EXEC_ENV), *args], capture_output=True, text=True, check=True).stdout


def resolve_commit(prefix: str) -> str:
    return git(["rev-parse", "--verify", f"{prefix}^{{commit}}"]).strip()


def fingerprint_at(commit: str) -> str:
    """``gate.gate_version()['fingerprint']`` for a clean checkout of ``commit``."""
    top = git(["rev-parse", "--show-toplevel"]).strip()
    relative = EXEC_ENV.relative_to(Path(top).resolve()).as_posix()
    names = [n for n in subprocess.run(["git", "-C", top, "ls-tree", "--name-only", commit, f"{relative}/"],
                                       capture_output=True, text=True, check=True).stdout.split("\n") if n]
    digest = hashlib.sha256()
    for name in sorted(names, key=lambda n: Path(n).name):
        if Path(name).suffix in (".py", ".R", ".sh"):
            content = subprocess.run(["git", "-C", top, "show", f"{commit}:{name}"], capture_output=True, check=True).stdout
            digest.update(Path(name).name.encode() + b"\0" + content + b"\0")
    return digest.hexdigest()[:16]


def function_dirs(slice_dir: Path) -> list[Path]:
    return sorted(p for p in slice_dir.iterdir() if p.is_dir() and (p / "source.R").is_file() and (p / "meta.json").is_file())


def conditioned_declarations(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or parts[0].startswith("#"):
            continue
        parsed = parse_declaration(parts[1])
        tag = re.search(r"\b(SCOPED|BUGFIX|QUIRK)\[", parts[1])
        if parsed["cond"] and tag:
            out.append({"mutant": parts[0], "tag": tag.group(1), "condition": parsed["cond"],
                        "modifiers": {"tol": parsed["tol"], "attrs": parsed["attrs"]}, "reason": parts[1][:300]})
    return out


def check_function(directory: Path, commit: str, fingerprint: str) -> tuple[dict, list[str]]:
    """(manifest record, problems) for one function directory."""
    meta = json.loads((directory / "meta.json").read_text())
    record: dict = {"function": directory.name, "package": meta.get("package"), "name": meta.get("name"),
                    "path": meta.get("path")}
    problems: list[str] = []
    if (directory / "SKIP").exists():
        record.update(status="skipped", skip_reason=(directory / "SKIP").read_text(encoding="utf-8").strip()[:500])
        return record, problems
    if not (directory / "test.R").is_file():
        record["status"] = "blocked"
        return record, [f"{directory.name}: neither SKIP nor test.R"]
    gate_path = directory / "gate.json"
    if not gate_path.is_file():
        record["status"] = "blocked"
        return record, [f"{directory.name}: test.R was never gated"]
    gate = json.loads(gate_path.read_text())
    version = gate.get("gate_version") or {}
    if version.get("commit") != commit:
        problems.append(f"{directory.name}: gated at {version.get('commit')}, not {commit[:12]}")
    if version.get("exec_env_modified"):
        problems.append(f"{directory.name}: gated with modified exec_env sources")
    if version.get("fingerprint") != fingerprint:
        problems.append(f"{directory.name}: gate fingerprint {version.get('fingerprint')} != {fingerprint}")
    gated_at = gate_path.stat().st_mtime
    inputs = [directory / n for n in INPUTS if (directory / n).exists()]
    inputs += sorted((directory / "probes").glob("*.R")) if (directory / "probes").is_dir() else []
    newer = [p.relative_to(directory).as_posix() for p in inputs if p.stat().st_mtime > gated_at + 1]
    if newer:
        problems.append(f"{directory.name}: changed after gating: {', '.join(newer)}")
    mutants = gate.get("mutants") or {}
    if mutants.get("stale_equivalent_ids") or mutants.get("declared_equivalent_but_killed"):
        problems.append(f"{directory.name}: stale or killed equivalent declarations")
    if gate.get("verdict") == "skip":
        record.update(status="skipped", skip_reason="gate: " + ";".join(gate.get("problems", [])))
    elif gate.get("verdict") != "pass":
        problems.append(f"{directory.name}: gate verdict {gate.get('verdict')}: {';'.join(gate.get('problems', []))[:200]}")
    record.update(
        status="frozen" if gate.get("verdict") == "pass" else record.get("status", "failed"),
        sha256={n: sha256_file(directory / n) for n in HASHED if (directory / n).is_file()},
        probes={p.name: sha256_file(p) for p in sorted((directory / "probes").glob("*.R"))}
        if (directory / "probes").is_dir() else {},
        gate={"checks": gate.get("checks"), "hand_written": gate.get("hand_written_checks"),
              "categories": gate.get("categories"),
              "mutants": {k: mutants.get(k) for k in ("generated", "sampled", "valid", "killed", "declared_equivalent")},
              "probes": gate.get("probes"), "warnings": gate.get("warnings", []),
              "recorded_calls": (gate.get("differential") or {}).get("calls"),
              "condition_exclusions": (gate.get("differential") or {}).get("excluded", {})},
        conditioned_declarations=conditioned_declarations(directory / "equivalent.txt"))
    if problems:
        record["status"] = "blocked"
    return record, problems


def build_manifest(root: Path, slices: Sequence[str], commit: str, fingerprint: str) -> tuple[dict, list[str]]:
    records, problems = [], []
    for slice_id in slices:
        slice_dir = root / slice_id
        if not slice_dir.is_dir():
            problems.append(f"{slice_id}: no such slice")
            continue
        for directory in function_dirs(slice_dir):
            record, issues = check_function(directory, commit, fingerprint)
            record["slice"] = slice_id
            records.append(record)
            problems += [f"{slice_id}/{issue}" for issue in issues]
    status = Counter(r.get("status") for r in records)
    untested = [{"slice": r["slice"], "function": r["function"], **d} for r in records
                for d in r.get("conditioned_declarations", []) if d["tag"] in ("SCOPED", "BUGFIX")]
    manifest = {
        "schema": SCHEMA, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gate": {"commit": commit, "fingerprint": fingerprint}, "slices": list(slices), "root": str(root),
        "counts": {"functions": len(records), **dict(status)},
        "skipped": [{"slice": r["slice"], "function": r["function"], "name": r["name"], "reason": r["skip_reason"]}
                    for r in records if r.get("status") == "skipped"],
        "untested_branches": untested,
        "low_replay_coverage": [{"slice": r["slice"], "function": r["function"], "name": r["name"],
                                 "recorded_calls": r["gate"]["recorded_calls"]}
                                for r in records if r.get("status") == "frozen"
                                and (r["gate"]["recorded_calls"] or 0) < LOW_REPLAY_CALLS],
        "known_limitations": KNOWN_LIMITATIONS,
        "functions": [r for r in records if r.get("status") == "frozen"],
    }
    return manifest, problems


def copy_read_only(root: Path, slices: Sequence[str], target: Path) -> dict:
    """Copy the slices (no ``.claim``) to ``target`` and drop write permission; returns a summary."""
    files = 0
    for slice_id in slices:
        shutil.copytree(root / slice_id, target / slice_id, ignore=shutil.ignore_patterns(".claim"))
    for path in sorted(target.rglob("*")):
        if path.is_file():
            files += 1
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for path in sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)) + [target]:
        path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    probe = next((p for p in target.rglob("*") if p.is_file()), None)
    effective = probe is not None and not os.access(probe, os.W_OK)
    return {"path": str(target), "files": files, "write_permission_removed": effective}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slices", required=True, help="e.g. H000..H007")
    parser.add_argument("--gate", required=True, help="pinned gate commit (prefix)")
    parser.add_argument("--root", type=Path, default=DATA / "authors")
    parser.add_argument("--out", type=Path, default=DATA / "holdout-freeze-v1.json")
    parser.add_argument("--copy-to", type=Path, default=DATA / "holdout-freeze-v1")
    parser.add_argument("--check", action="store_true", help="verify and report only; write nothing")
    parser.add_argument("--fingerprint", help=argparse.SUPPRESS)  # tests: skip the git lookup
    args = parser.parse_args(argv)
    slices = expand_slices(args.slices)
    commit = args.gate if args.fingerprint else resolve_commit(args.gate)
    fingerprint = args.fingerprint or fingerprint_at(commit)
    manifest, problems = build_manifest(args.root, slices, commit, fingerprint)
    print(json.dumps({"slices": slices, "gate": manifest["gate"], "counts": manifest["counts"],
                      "untested_branches": len(manifest["untested_branches"]),
                      "low_replay_coverage": len(manifest["low_replay_coverage"]), "problems": len(problems)}, indent=1))
    for problem in problems[:200]:
        print("PROBLEM", problem)
    if args.check:
        return 1 if problems else 0
    if problems:
        print("not frozen: fix the problems above (or re-gate) first")
        return 1
    if args.out.exists() or args.copy_to.exists():
        print(f"refusing to overwrite {args.out if args.out.exists() else args.copy_to}")
        return 1
    manifest["read_only_copy"] = copy_read_only(args.root, slices, args.copy_to)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    manifest_sha = sha256_file(args.out)
    args.out.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"written": str(args.out), "sha256": manifest_sha, "copy": manifest["read_only_copy"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
