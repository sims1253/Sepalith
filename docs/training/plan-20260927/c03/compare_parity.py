#!/usr/bin/env python3
"""Compare Kaggle fp16 (or fp32) generations with the PC bf16 references (C03 step 4).

Inputs are generation records in the shared format (``id``, ``sample``,
``generated_ids``, optional ``prompt_sha256``): the PC side from
``parity_references.py`` (``<label>/dev75-generations.jsonl``,
``<label>/train200-generations.jsonl``, ``<label>/dev75.cases/cases.json``)
and the Kaggle side from ``scripts/kaggle/infer_entry.py``
(``generations.jsonl``, ``cases.json``, ``summary.json``). An evaluator
``cases.json`` is also accepted wherever generations are expected.

Reported per set (DEV75, TRAIN-200) and combined:

- exact token-id identity per case, the identity rate, and the first
  divergence position of every differing case;
- prompt identity (``prompt_sha256``): a prompt that differs means the two
  sides tokenized differently, which is not a numerics difference;
- for DEV75 with evaluator cases on both sides: the R1 counts ``valid``,
  ``edit``, ``nofp`` and ``cap`` on each side, their differences, and the
  per-case agreement on each flag;
- non-finite values on the candidate side (``calls_with_nonfinite_logits``
  and ``nll_finite`` from its ``summary.json``).

The card's pass criteria (cards/C03-kaggle-inference.md, step 4): no NaN or
inf, each of the four DEV75 counts within 1 of the bf16 run, and at least
90% of all compared cases token-identical. The comparison also checks that
both sides used the same batching (batch size, KV cache, cap); a mismatch is
reported as not comparable rather than as a parity failure. On the CPT
parent most cases run to the cap without a stop token, so parent parity
mostly measures drift in run-on text; the SFT checkpoint is the
representative comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

IDENTITY_FLOOR = 0.90
COUNT_TOLERANCE = 1
R1_FLAGS = {"valid": "protocol_valid", "cap": "cap_hit"}


def load_records(path):
    """id -> record from a generations JSONL (sample 0) or an evaluator cases.json."""
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())["results"]
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row.get("sample", 0) == 0]
    records = {row["id"]: row for row in rows}
    if len(records) != len(rows):
        raise SystemExit(f"duplicate ids in {path}")
    return records


def first_divergence(a, b):
    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def identity(reference, candidate):
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    divergent, prompt_mismatch = [], []
    for case_id in sorted(set(reference) & set(candidate)):
        ref, cand = reference[case_id], candidate[case_id]
        if ref.get("prompt_sha256") and cand.get("prompt_sha256") and ref["prompt_sha256"] != cand["prompt_sha256"]:
            prompt_mismatch.append(case_id)
        position = first_divergence(ref["generated_ids"], cand["generated_ids"])
        if position is not None:
            divergent.append({"id": case_id, "first_divergence": position,
                              "reference_tokens": len(ref["generated_ids"]), "candidate_tokens": len(cand["generated_ids"]),
                              "reference_cap_hit": ref.get("cap_hit"), "candidate_cap_hit": cand.get("cap_hit")})
    compared = len(set(reference) & set(candidate))
    positions = [row["first_divergence"] for row in divergent]
    return {"cases": compared, "identical": compared - len(divergent),
            "identity_rate": (compared - len(divergent)) / compared if compared else None,
            "missing_in_candidate": missing, "extra_in_candidate": extra, "prompt_mismatches": prompt_mismatch,
            "first_divergence": {"min": min(positions), "median": statistics.median(positions), "max": max(positions)}
            if positions else None,
            "divergent": divergent}


def r1_from_cases(cases):
    return {"valid": sum(bool(c["protocol_valid"]) for c in cases), "cap": sum(bool(c["cap_hit"]) for c in cases),
            "edit": sum((not c["expected_noop"]) and bool(c["exact_region"]) for c in cases),
            "nofp": sum(bool(c["expected_noop"]) and bool(c["suggestion"]) for c in cases)}


def r1_comparison(reference_cases, candidate_cases):
    ids = sorted(set(reference_cases) & set(candidate_cases))
    ref = [reference_cases[i] for i in ids]
    cand = [candidate_cases[i] for i in ids]
    counts = {"reference": r1_from_cases(ref), "candidate": r1_from_cases(cand)}
    counts["difference"] = {k: counts["candidate"][k] - counts["reference"][k] for k in counts["reference"]}
    flags = {"valid": lambda c: bool(c["protocol_valid"]), "cap": lambda c: bool(c["cap_hit"]),
             "edit": lambda c: (not c["expected_noop"]) and bool(c["exact_region"]),
             "nofp": lambda c: bool(c["expected_noop"]) and bool(c["suggestion"])}
    counts["per_case_agreement"] = {name: sum(fn(a) == fn(b) for a, b in zip(ref, cand, strict=True))
                                    for name, fn in flags.items()}
    counts["cases"] = len(ids)
    counts["within_tolerance"] = all(abs(v) <= COUNT_TOLERANCE for v in counts["difference"].values())
    return counts


def candidate_runtime(summary_path):
    if summary_path is None or not Path(summary_path).is_file():
        return None
    summary = json.loads(Path(summary_path).read_text())
    result = summary.get("result") or {}
    return {"generation": summary.get("generation"), "runtime": summary.get("runtime"), "gpus": summary.get("gpus"),
            "calls_with_nonfinite_logits": result.get("calls_with_nonfinite_logits"),
            "nll_finite": result.get("nll_finite"), "model": summary.get("model")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", type=Path, required=True, help="PC model folder, e.g. <attempt>/lr1e-5-gate-268")
    p.add_argument("--reference-receipt", type=Path, help="PC references.json (batching check)")
    p.add_argument("--dev75", type=Path, help="Kaggle DEV75 run folder (generations.jsonl, cases.json, summary.json)")
    p.add_argument("--train200", type=Path, help="Kaggle TRAIN-200 run folder (generations.jsonl, summary.json)")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if not (a.dev75 or a.train200):
        p.error("give --dev75 and/or --train200")
    receipt = {"schema": "sepalith.c03.parity-comparison.v1", "reference_dir": str(a.reference_dir), "sets": {},
               "criteria": {"identity_floor": IDENTITY_FLOOR, "count_tolerance": COUNT_TOLERANCE,
                            "nonfinite": "none allowed"}}
    settings = None
    if a.reference_receipt and a.reference_receipt.is_file():
        settings = json.loads(a.reference_receipt.read_text()).get("settings")
        receipt["reference_settings"] = settings
    comparable, nonfinite, all_identical, all_cases = [], [], 0, 0
    for name, folder, reference_file in (("dev75", a.dev75, "dev75-generations.jsonl"),
                                         ("train200", a.train200, "train200-generations.jsonl")):
        if folder is None:
            continue
        reference = load_records(a.reference_dir / reference_file)
        candidate = load_records(folder / "generations.jsonl")
        entry = {"candidate_dir": str(folder), "identity": identity(reference, candidate),
                 "candidate": candidate_runtime(folder / "summary.json")}
        generation = (entry["candidate"] or {}).get("generation") or {}
        if settings:
            same = (generation.get("batch_size"), generation.get("kv_cache"), generation.get("max_new_tokens")) == (
                settings.get("batch_size"), settings.get("kv_cache"), settings.get("cap"))
            entry["same_batching"] = same
            comparable.append(same)
        if name == "dev75" and (folder / "cases.json").is_file() and (a.reference_dir / "dev75.cases/cases.json").is_file():
            entry["r1"] = r1_comparison(load_records(a.reference_dir / "dev75.cases/cases.json"),
                                        load_records(folder / "cases.json"))
        runtime = entry["candidate"] or {}
        nonfinite.append(bool(runtime.get("calls_with_nonfinite_logits")) or runtime.get("nll_finite") is False)
        all_identical += entry["identity"]["identical"]
        all_cases += entry["identity"]["cases"]
        receipt["sets"][name] = entry
    combined = all_identical / all_cases if all_cases else None
    dev = receipt["sets"].get("dev75", {})
    checks = {"no_nonfinite": not any(nonfinite),
              "dev75_counts_within_1": (dev.get("r1") or {}).get("within_tolerance"),
              "identity_at_least_90pct": combined is not None and combined >= IDENTITY_FLOOR,
              "prompts_identical": all(not s["identity"]["prompt_mismatches"] for s in receipt["sets"].values())}
    receipt["combined_identity"] = {"cases": all_cases, "identical": all_identical, "identity_rate": combined}
    receipt["checks"] = checks
    if comparable and not all(comparable):
        receipt["verdict"] = "not_comparable_batching_differs"
    elif checks["dev75_counts_within_1"] is None:
        receipt["verdict"] = "incomplete_no_dev75_counts"
    else:
        receipt["verdict"] = "pass" if all(checks.values()) else "fail"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": receipt["verdict"], "checks": checks, "combined_identity": receipt["combined_identity"],
                      "r1": (dev.get("r1") or {}).get("difference")}, sort_keys=True))
    return 0 if receipt["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
