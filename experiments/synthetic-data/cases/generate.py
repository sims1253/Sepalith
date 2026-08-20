#!/usr/bin/env python3
"""Case harness: `uv run python -m cases.generate --case <name> --n 500
--backend <agy|zai|opencode|openrouter> --out <path>` (run from
experiments/synthetic-data).

Per row: seeded sampler draw -> prompt from the spec's template list ->
backend.complete (paced, retried, stats) -> 3-layer gate -> row construction
-> final row-structure check -> content-hash dedup -> append with drvfs
write retries. Provenance (case, template, backend, model, full_prompt,
generated_at, license/source_url when corpus-derived, seed, corpus key)
rides on every row. Progress survives crashes via a done-key sidecar
(<out>.done.jsonl, the agy_generators convention) and resumes on rerun.

Gate layers: (1) JSON extraction from the model text, (2) the spec's target
schema (field present, string, length bounds), (3) the registered case
validator (comment gate / tidyselect tree-sitter check / the validate.py
Rscript+jarl gate for whole-snippet code cases). A layer-3 failure is
regenerated ONCE before the item is marked done-and-rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # file-run (tests): bootstrap the package root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from cases.backends import BackendError, extract_json_object, make_backend, \
    strip_fences
from cases.corpus import select_corpus
from cases.rows import build_row, normalize_target
from cases.spec import CaseSpec, SpecError, list_cases, load_case, load_spec_file
from cases.samplers import make_sampler
from cases.validators import check_row, get_validator


def _append_line(path: Path, obj: dict, tries: int = 20, wait_s: float = 30.0):
    """Append one JSONL line, riding out drvfs ENOMEM flaps (see
    _nas_write_lines in comment_to_code.py: partial progress must never
    kill a long API run)."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    for attempt in range(tries):
        try:
            with open(path, "a") as fh:
                fh.write(line)
                fh.flush()
            return
        except OSError as e:
            if attempt == tries - 1:
                print(f"  [drvfs-write] giving up on {path}: {e}", flush=True)
                return
            print(f"  [drvfs-write] {e}; retry {attempt + 1}/{tries - 1} "
                  f"in {wait_s:.0f}s", flush=True)
            time.sleep(wait_s)


def _write_json(path: Path, obj, tries: int = 20, wait_s: float = 30.0):
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    for attempt in range(tries):
        try:
            path.write_text(text)
            return
        except OSError as e:
            if attempt == tries - 1:
                print(f"  [drvfs-write] giving up on {path}: {e}", flush=True)
                return
            print(f"  [drvfs-write] {e}; retry {attempt + 1}/{tries - 1} "
                  f"in {wait_s:.0f}s", flush=True)
            time.sleep(wait_s)


def _content_hash(spec: CaseSpec, item: dict, target: str) -> str:
    norm = " ".join(target.split())
    scope = (f"{spec.case_scope()}\x00{item.get('key', '-')}\x00{norm}"
             if spec.dedup == "target+key"
             else f"{spec.case_scope()}\x00{norm}")
    return hashlib.sha1(scope.encode()).hexdigest()


def _layer2(spec: CaseSpec, obj: dict) -> tuple[bool, str, str | None]:
    """Spec schema on the parsed object: target field present, a string,
    within the length bounds. Returns (ok, reason, target)."""
    tf = spec.target_field
    if tf not in obj:
        return False, f"response lacks {tf!r}", None
    t = obj[tf]
    if not isinstance(t, str):
        return False, f"{tf!r} is {type(t).__name__}, not string", None
    d = spec.difficulty or {}
    lo = int(d.get("target_chars_min", 0) or 0)
    hi = int(d.get("target_chars_max", 0) or 0)
    if lo and len(t.strip()) < lo:
        return False, f"{tf!r} shorter than {lo} chars", None
    if hi and len(t.strip()) > hi:
        return False, f"{tf!r} longer than {hi} chars", None
    return True, "", t


def generate(spec: CaseSpec, backend_name: str, n: int, out: Path,
             seed: int = 13, verbose: bool = True) -> dict:
    rng = random.Random(seed)
    backend = make_backend(backend_name, target_key=spec.target_field, seed=seed)
    validate_target = get_validator(spec.validator)
    sampler = make_sampler(
        dict(spec.parameter_sampler, n_templates=len(spec.prompt_templates)),
        rng)
    normalizer = spec.raw.get("target_normalizer", "raw")

    print(f"case={spec.name} v{spec.version} backend={backend_name} "
          f"model={backend.model} n={n} seed={seed}")
    items = select_corpus(spec, rng, want=n)
    if not items:
        raise RuntimeError(f"corpus selector produced 0 items for {spec.name}")
    print(f"  [corpus] {len(items)} candidate items "
          f"(need {n} accepted; provenance: {spec.provenance.get('note', 'n/a')})")

    out.parent.mkdir(parents=True, exist_ok=True)
    done_path = Path(str(out) + ".done.jsonl")

    # ---- resume state -------------------------------------------------
    done: dict[str, dict] = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["key"]] = rec
            except (ValueError, KeyError):
                pass
    seen_hashes: set[str] = set()
    accepted = 0
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("content_hash"):
                    seen_hashes.add(row["content_hash"])
                accepted += 1
            except ValueError:
                pass
    print(f"  [resume] rows={accepted} done-keys={len(done)} "
          f"hashes={len(seen_hashes)}")

    prov = spec.provenance
    stats = dict(attempted=0, accepted=0, rejected_json=0, rejected_schema=0,
                 rejected_validator=0, rejected_rowcheck=0, dups=0,
                 backend_errors=0, items_skipped_done=0)
    t0 = time.time()
    last_print = 0.0
    target_n = n  # total rows the out file should hold after this run

    for item in items:
        if accepted >= target_n:
            break
        if item["key"] in done:
            stats["items_skipped_done"] += 1
            continue
        ok, rec = _attempt_item(spec, item, backend, sampler, validate_target,
                                normalizer, seen_hashes, stats, prov,
                                rng, seed)
        row = rec.pop("row", None)      # rows go to out, keys to the sidecar
        _append_line(done_path, rec)
        done[item["key"]] = rec
        if not ok:
            continue
        _append_line(out, row)
        accepted += 1
        if verbose and (accepted % 10 == 0 or time.time() - last_print > 60):
            last_print = time.time()
            print(f"  rows={accepted}/{target_n} "
                  f"stats={ {k: v for k, v in stats.items() if v} } "
                  f"backend={backend.stats_summary().get('ok')}ok "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    report = dict(case=spec.name, case_version=spec.version,
                  backend=backend_name, model=backend.model, seed=seed,
                  out=str(out), rows_total=accepted, target=target_n,
                  elapsed_s=round(time.time() - t0, 1),
                  backend_stats=backend.stats_summary(), counts=stats,
                  done_keys=len(done))
    report["throughput_per_min"] = round(
        stats["accepted"] / max(0.1, report["elapsed_s"]) * 60, 1)
    _write_json(Path(str(out) + ".stats.json"), report)
    if verbose:
        print(json.dumps(report, indent=1))
    return report


def _attempt_item(spec, item, backend, sampler, validate_target, normalizer,
                  seen_hashes, stats, prov, rng, seed) -> tuple[bool, dict]:
    """Up to 2 generations (layer-3 failures regenerate once, the
    build_synthetic convention); every terminal outcome lands in the done
    sidecar so resume never re-burns quota."""
    stats["attempted"] += 1
    last_reason = "?"
    for gen in range(2):
        draw = sampler()
        t_i = int(draw["template_index"])
        prompt = spec.fill_template(t_i, item)
        try:
            raw = backend.complete(prompt)
        except (BackendError, RuntimeError) as e:
            stats["backend_errors"] += 1
            last_reason = f"backend: {e}"[:160]
            continue
        obj = extract_json_object(strip_fences(raw))
        if obj is None:
            stats["rejected_json"] += 1
            last_reason = "layer1: no JSON object"
            continue
        ok, reason, target = _layer2(spec, obj)
        if not ok:
            stats["rejected_schema"] += 1
            last_reason = f"layer2: {reason}"
            continue
        target = normalize_target(normalizer, target)
        ok, reason = validate_target(target)
        if not ok:
            stats["rejected_validator"] += 1
            last_reason = f"layer3: {reason}"
            continue                     # regenerate once with a new draw
        h = _content_hash(spec, item, target)
        if h in seen_hashes:
            stats["dups"] += 1
            return False, dict(key=item["key"], ok=True, dup=True, hash=h,
                               ts=_now())
        seen_hashes.add(h)
        row = build_row(spec, item, target,
                        dict(model=backend.model, backend=backend.name))
        row.update(dict(
            case=spec.name, case_version=spec.version,
            template_index=t_i, backend=backend.name, model=backend.model,
            full_prompt=prompt, generated_at=_now(), seed=seed,
            corpus_key=item["key"], content_hash=h,
            license=prov.get("license"), source_url=prov.get("source_url"),
            generator=f"cases.generate:{backend.name}",
        ))
        ok, reason = check_row(row, spec.row_check)
        if not ok:
            stats["rejected_rowcheck"] += 1
            return False, dict(key=item["key"], ok=False,
                               reason=f"rowcheck: {reason}"[:160], ts=_now())
        stats["accepted"] += 1
        return True, dict(key=item["key"], ok=True, hash=h, row=row,
                          ts=_now())
    return False, dict(key=item["key"], ok=False,
                       reason=last_reason[:160], ts=_now())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m cases.generate",
        description="Generate validated synthetic-data rows from a "
                    "declarative case spec.")
    ap.add_argument("--case", help=f"case name ({', '.join(list_cases())} ...)")
    ap.add_argument("--spec", type=Path, help="path to a case spec JSON "
                                              "(alternative to --case)")
    ap.add_argument("--n", "-n", type=int, default=100,
                    help="total rows the output should hold after this run")
    ap.add_argument("--backend", default="agy",
                    choices=["agy", "zai", "opencode", "openrouter", "mock"])
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSONL (default results/cases/<case>.jsonl)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rescan", action="store_true",
                    help="ignore cached corpus scans")
    args = ap.parse_args(argv)

    try:
        spec = load_spec_file(args.spec) if args.spec else load_case(args.case)
    except SpecError as e:
        sys.exit(str(e))
    if args.rescan:
        from cases.corpus import CACHE_DIR
        for p in CACHE_DIR.glob("*.json"):
            p.unlink(missing_ok=True)
    out = args.out or (Path(__file__).resolve().parent.parent /
                       "results" / "cases" / f"{spec.name}.jsonl")
    generate(spec, args.backend, args.n, out, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
