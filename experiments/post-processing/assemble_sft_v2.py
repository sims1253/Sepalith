#!/usr/bin/env python3
"""Assemble the SFT v2 training mixture (CPU-only, single process).

Sources (all on the NAS under /mnt/h/sepalith/datasets):
  1. finish_block        sft_v1 train (ALL) / eval (unchanged, same held-out pkgs)
  2. edit_pairs          edit_pairs_v1 examples->train, eval->v2 eval (zeta2)
  3. scenarios           scenarios_v1 7 canonical files (zeta2), per-family 3% package holdout
     (covers comment_to_code_real + comment_to_code_synthetic families)
  4. pr_instructed       reviewer instruction as a "# reviewer:" comment line
  5. synthetic_analyst   comment->code style, analyst.R
  7. hidden_r_instruction 40k stratified sample (seed 5), plain alpaca-ish text
  8. paper_to_r          passed==true rows only, comment->code style

Output: /mnt/h/sepalith/datasets/sft_v2/{train,eval}.jsonl
Row schema: {text, prompt, target, family, package_or_repo, has_types: false}
Train shuffled with seed 42. Run resource-polite: nice -n 19, 1 process, no GPU.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import render_zeta2  # noqa: E402  (exact v1 renderer conventions)

NAS = Path("/mnt/h/sepalith/datasets")
OUT = NAS / "sft_v2"
UPDATED = "\n>>>>>>> UPDATED"
MAX_CHARS = 6000  # v1 convention: prompt+target char budget for edit-format rows

SCENARIO_FILES = [
    "rename_propagation.jsonl",
    "pipe_rewrite.jsonl",
    "na_rm_propagation.jsonl",
    "format_propagation.jsonl",
    "doc_sync.jsonl",
    "comment_to_code_real.jsonl",
    "comment_to_code_synthetic.jsonl",
]

stats = Counter()
notes = []


def row(prompt, target, family, pkg):
    return dict(text=prompt + target, prompt=prompt, target=target,
                family=family, package_or_repo=pkg, has_types=False)


def edit_row(ex, family, pkg):
    """zeta2 render + UPDATED-marker target; drops rows with nothing to predict
    or over the v1 char budget (removes runaway mining contexts)."""
    if not [l for l in ex.get("region_new") or [] if l.strip()]:
        stats[f"drop:{family}:empty_region_new"] += 1
        return None
    ex = dict(ex)
    ex.setdefault("suffix", [])  # scenario rows carry prefix context only
    prompt = render_zeta2(ex)
    target = "\n".join(ex["region_new"]).rstrip() + UPDATED
    if len(prompt) + len(target) > MAX_CHARS:
        stats[f"drop:{family}:over_{MAX_CHARS}"] += 1
        return None
    return row(prompt, target, family, pkg)


def comment_to_code_row(filename, comment, code, family, pkg):
    """zeta2 empty-region style: last prefix line is a comment, region = cursor."""
    parts = (["<[fim-suffix]>"] +
             [f"<[fim-prefix]><filename>{filename}"] +
             [comment.rstrip()] +
             ["<<<<<<< CURRENT", "<|user_cursor|>", "=======", "<[fim-middle]>"])
    prompt = "\n".join(parts)
    body = code.rstrip()
    if body.endswith(">>>>>>> UPDATED"):  # never emit a second marker
        body = body[: -len(">>>>>>> UPDATED")].rstrip()
        notes.append(f"{family}: stripped pre-existing UPDATED marker")
    return row(prompt, body + UPDATED, family, pkg)


def load_finish_block():
    fam = "finish_block"
    for split, path in (("train", NAS / "sft_v1/train.jsonl"),
                        ("eval", NAS / "sft_v1/eval.jsonl")):
        for line in open(path):
            r = json.loads(line)
            yield split, row(r["prompt"], r["target"], fam, r["package"])
            stats[f"{fam}:{split}"] += 1


def load_edit_pairs():
    fam = "edit_pairs"
    for split, path in (("train", NAS / "edit_pairs_v1/examples.jsonl"),
                        ("eval", NAS / "edit_pairs_v1/eval.jsonl")):
        for line in open(path):
            r = json.loads(line)
            rr = edit_row(r, fam, r["repo"])
            if rr:
                yield split, rr
                stats[f"{fam}:{split}"] += 1


def load_scenarios():
    """7 canonical files; per FAMILY hold out 3% of that family's packages
    (min 1, seed 42) so every scenario family has eval coverage."""
    recs = []
    fam_pkgs = defaultdict(set)
    for fname in SCENARIO_FILES:
        path = NAS / "scenarios_v1" / fname
        if not path.exists():
            # e.g. comment_to_code_synthetic.jsonl quarantined as .bak while
            # the (fixed) generator rebuilds it; assemble without the family.
            notes.append(f"scenarios: {fname} missing -> skipped (0 rows)")
            continue
        for line in open(path):
            r = json.loads(line)
            recs.append(r)
            fam_pkgs[r["family"]].add(r["package"])
    rng = random.Random(42)
    eval_pkgs = {}
    for fam in sorted(fam_pkgs):
        ordered = sorted(fam_pkgs[fam])
        rng.shuffle(ordered)
        n_hold = max(1, round(len(ordered) * 0.03))
        eval_pkgs[fam] = set(ordered[:n_hold])
        stats[f"scenarios:{fam}:packages_held_out"] = f"{n_hold}/{len(ordered)}"
    for r in recs:
        fam = r["family"]
        split = "eval" if r["package"] in eval_pkgs[fam] else "train"
        rr = edit_row(r, fam, r["package"])
        if rr:
            yield split, rr
            stats[f"{fam}:{split}"] += 1


def load_pr_instructed():
    """zeta2 with the edit_history section replaced by a reviewer comment line.

    Cursor follows the edit-pair mining convention: first line where
    region_old differs from region_new (fallback: last line of region_old).
    """
    fam = "pr_instructed"
    n_multiline = 0
    for line in open(NAS / "pr_instructed_v1/pilot.jsonl"):
        r = json.loads(line)
        instr = r["instruction"].strip()
        if "\n" in instr:
            n_multiline += 1
            instr = " ".join(instr.split())  # keep it one comment line
        ro, rn = r["region_old"], r["region_new"]
        fd = next((i for i in range(min(len(ro), len(rn))) if ro[i] != rn[i]), None)
        ex = dict(suffix=[], path=r["path"], event_diff="",
                  prefix=r["prefix"] + [f"# reviewer: {instr}"],
                  region_old=ro, region_new=rn,
                  cursor_idx=fd if fd is not None else len(ro) - 1)
        rr = edit_row(ex, fam, r["repo"])
        if rr:
            yield "train", rr
            stats[f"{fam}:train"] += 1
    if n_multiline:
        notes.append(f"pr_instructed: {n_multiline} instruction(s) contained "
                     f"newlines, whitespace-collapsed to one comment line")


def load_synthetic_analyst():
    fam = "synthetic_analyst"
    for line in open(NAS / "synthetic_analyst_v1/analyst_scripts.jsonl"):
        r = json.loads(line)
        rr = comment_to_code_row("analyst.R", f"# {r['intent'].strip()}",
                                 r["code"], fam, "synthetic_analyst_v1")
        if rr:
            yield "train", rr
            stats[f"{fam}:train"] += 1


def load_hidden_r_instruction(n_total=40_000, n_eval=1_000, seed=5):
    """Stratified sample: ALL of codex_r_strict + random ling_coder fill.

    Plain-text general-instruction tail (NOT zeta2). package_or_repo is the
    per-row source id (mid / dataset_row) — these rows have no repo concept.
    Eval holdout of n_eval is drawn proportionally per stratum (seed 5).
    """
    fam = "hidden_r_instruction"

    def render(inp, outp):
        prompt = f"### Instruction:\n{inp.strip()}\n\n### Response:\n"
        return row(prompt, outp.rstrip() + "\n", fam, None)  # pkg filled by caller

    codex = []
    for line in open(NAS / "hidden_r_instruction_v1/codex_r_strict.jsonl"):
        r = json.loads(line)
        rr = render(r["input"], r["output"])
        rr["package_or_repo"] = r["dataset_row"]
        codex.append(rr)
    ling = []
    for line in open(NAS / "hidden_r_instruction_v1/ling_coder_r.jsonl"):
        r = json.loads(line)
        inp = r["messages"][0]["content"]
        outp = r["messages"][-1]["content"]
        rr = render(inp, outp)
        rr["package_or_repo"] = r["mid"]
        ling.append(rr)

    stats["hidden_r_instruction:ling_pool"] = len(ling)
    rng = random.Random(seed)
    rng.shuffle(ling)
    rng.shuffle(codex)
    # proportional stratified holdout: n_eval/n_total per stratum (first k of
    # the seeded shuffles above -> reproducible)
    codex_all = codex
    ling_take = ling[: n_total - len(codex_all)]
    frac = n_eval / n_total
    ev_codex = round(len(codex_all) * frac)
    ev_ling = round(len(ling_take) * frac)
    eval_rows = codex_all[:ev_codex] + ling_take[:ev_ling]
    eval_ids = {id(r) for r in eval_rows}
    train_rows = [r for r in codex_all + ling_take if id(r) not in eval_ids]
    stats[f"{fam}:eval_codex"] = ev_codex
    stats[f"{fam}:eval_ling"] = ev_ling
    for r in train_rows:
        stats[f"{fam}:train"] += 1
        yield "train", r
    for r in eval_rows:
        stats[f"{fam}:eval"] += 1
        yield "eval", r


def load_paper_to_r():
    fam = "paper_to_r"
    for line in open(NAS / "paper_to_r_pilot/examples.jsonl"):
        r = json.loads(line)
        if not r.get("passed"):
            continue
        rr = comment_to_code_row("method.R",
                                 f"# {r['method']}: {r['property'].strip()}",
                                 r["implementation"], fam, r["method"])
        if rr:
            yield "train", rr
            stats[f"{fam}:train"] += 1


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output dir (default: the live sft_v2; use e.g. "
                         "sft_v2_clean to rebuild without touching it)")
    args = ap.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    train, evals = [], []
    for loader in (load_finish_block, load_edit_pairs, load_scenarios,
                   load_pr_instructed, load_synthetic_analyst,
                   load_hidden_r_instruction, load_paper_to_r):
        for split, r in loader():
            (train if split == "train" else evals).append(r)

    # ---- dedup on text across the WHOLE mixture (train+eval), keep first ----
    seen, uniq_train, uniq_eval, n_dup = set(), [], [], 0
    for bucket, src in (("train", train), ("eval", evals)):
        for r in src:
            h = hash(r["text"])
            if h in seen:
                n_dup += 1
                stats[f"dup:{r['family']}"] += 1
                continue
            seen.add(h)
            (uniq_train if bucket == "train" else uniq_eval).append(r)
    train, evals = uniq_train, uniq_eval
    stats["dropped_duplicate_text"] = n_dup

    # ---- validation ----
    assert all(r["target"].strip() for r in train + evals), "empty target found"
    train_pkgs = defaultdict(set)
    for r in train:
        train_pkgs[r["family"]].add(r["package_or_repo"])
    overlap = {}
    for fam in sorted(train_pkgs):
        ev = {r["package_or_repo"] for r in evals if r["family"] == fam}
        ov = ev & train_pkgs[fam]
        if ov:
            overlap[fam] = len(ov)
    assert not overlap, f"eval/train package overlap: {overlap}"

    random.Random(42).shuffle(train)

    with open(OUT / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "eval.jsonl", "w") as f:
        for r in evals:
            f.write(json.dumps(r) + "\n")

    # ---- report ----
    fam_tr = Counter(r["family"] for r in train)
    fam_ev = Counter(r["family"] for r in evals)
    lengths = sorted(len(r["text"]) for r in train)
    per_fam_len = {f: sorted(len(r["text"]) for r in train if r["family"] == f)
                   for f in sorted(fam_tr)}
    eval_lengths = sorted(len(r["text"]) for r in evals)
    report = dict(
        total_train=len(train), total_eval=len(evals),
        families={f: dict(train=fam_tr.get(f, 0), eval=fam_ev.get(f, 0))
                  for f in sorted(set(fam_tr) | set(fam_ev))},
        train_len_chars=dict(p50=pct(lengths, 50), p95=pct(lengths, 95),
                             p99=pct(lengths, 99), max=lengths[-1] if lengths else 0),
        eval_len_chars=dict(p50=pct(eval_lengths, 50), p95=pct(eval_lengths, 95),
                            p99=pct(eval_lengths, 99), max=eval_lengths[-1] if eval_lengths else 0),
        per_family_train_p50_p95={f: [pct(v, 50), pct(v, 95)]
                                  for f, v in per_fam_len.items()},
        checks=dict(duplicate_text_rows_dropped=n_dup,
                    empty_targets=0, eval_train_pkg_overlap=overlap),
        notes=notes,
    )
    (OUT / "stats.json").write_text(json.dumps({"report": report,
                                                "source_counts": dict(stats)}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
