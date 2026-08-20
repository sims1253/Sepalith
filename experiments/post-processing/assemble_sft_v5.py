#!/usr/bin/env python3
"""Assemble the SFT v5 training mixture (CPU-only, single process).

THE v5 TARGET CONVENTION (user-decided, final): the target is ALWAYS only
what comes after the cursor (suffix completion). The old finish-block
convention of re-emitting the whole region including the typed partial is
retired. Concretely, vs sft_v2/v3:

  * finish_block is RE-RENDERED from the source records
    (experiments/synthetic-data/finish_block_sample.jsonl), not copied from
    sft_v1. For kind=mid_body the head (the partial body lines already in
    region_old, cursor at their end) is defensively stripped off the target:
    target_new = full_target[len(head):]; rows where nothing remains after
    the strip are dropped and counted. kind=signature is unchanged. (On the
    current corpus the generator already emitted target=tail, so the strip
    converts 0 rows — the counts in stats.json prove it either way.)
  * every zeta2 edit family (edit_pairs, pr_instructed, the scenario
    families) is rendered with the cursor at the FIRST CHANGED line of the
    region and the target stripped of the unchanged head lines that sit
    above the cursor (they were re-emitted in v2). Scenario rows stored a
    character-offset cursor_idx, which render_zeta2 silently read as a line
    index — their cursor marker was LOST in v2; v5 always places a visible
    line-index cursor.
  * NEW family no_op (experiments/synthetic-data/suffix_scenarios.py):
    empty targets are allowed ONLY here (after a function's closing brace:
    stop immediately; between functions: one blank line, then stop).
  * NEW family mid_roxygen: cursor inside a roxygen block, target = the
    remaining roxygen lines (native suffix convention).
  * families whose region above the cursor is empty by construction
    (comment_to_code, comment_drafting, roxygen_drafting, synthetic_analyst,
    paper_to_r) already complied and are rendered as in v2. Verdicts are
    recorded per family in stats.json.
  * hidden_r_instruction has no cursor/edit format at all (plain text) and
    is carried over unchanged.

Sources (all on the NAS under /mnt/h/sepalith/datasets):
  1. finish_block        source records, v1 dedup + v1 package split (seed 11)
  2. edit_pairs          edit_pairs_v1 examples->train, eval->v2 eval (zeta2)
  3. scenarios           scenarios_v1 canonical files (incl. no_op +
                         mid_roxygen), per-family 3% package holdout
  4. pr_instructed       reviewer instruction as a "# reviewer:" comment line
  5. synthetic_analyst   comment->code style, analyst.R
  6. hidden_r_instruction 40k stratified sample (seed 5), plain alpaca text
  7. paper_to_r          passed==true rows only, comment->code style

Output: /mnt/h/sepalith/datasets/sft_v5/{train,eval}.jsonl + stats.json
Row schema: {text, prompt, target, family, package_or_repo, has_types: false}
Train shuffled with seed 42. Run resource-polite: nice -n 19, 1 process.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import render_zeta2  # noqa: E402  (exact v1 renderer conventions)

REPO = Path("/home/m0hawk/Documents/Sepalith")
NAS = Path("/mnt/h/sepalith/datasets")
OUT = NAS / "sft_v5"
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
    "comment_to_code_gemini.jsonl",  # gemini-3.7-flash harvest, own family
    "comment_drafting.jsonl",
    "comment_insert.jsonl",
    "roxygen_drafting.jsonl",
    "no_op.jsonl",          # v5: the eagerness fix (empty targets allowed)
    "mid_roxygen.jsonl",    # v5: suffix-convention roxygen continuation
    # cases wave1 (experiments/synthetic-data/cases): entries containing "/"
    # are NAS-relative (resolved against NAS itself, not scenarios_v1/). Same
    # scenarios_v1 edit-format schema (prefix/region_old==[""]/region_new/
    # cursor_idx==0/event_diff==""), so edit_row renders them unchanged with
    # fd=0 (target = region_new, the verbatim corpus remainder).
    "cases_v1/namespace_qualify_propagation.jsonl",
    "cases_v1/pipe_chain_link.jsonl",
    "cases_v1/pkg_metadata_sync.jsonl",
    "cases_v1/expectation_completion.jsonl",
    "cases_v1/trycatch_handler_completion.jsonl",
    "cases_v1/mid_body_edit.jsonl",   # one changed line mid-function; suffix
                                      # pins the post-change function remainder
    # retyping families (edit-stream ideas): the removed-block continuum.
    # astfim_partial derives train rows from the astfim_v1 FIXED TRAIN split
    # (parent package holdout respected); its 100-row eval companion lives at
    # cases_v1/astfim_partial_eval.jsonl but stays UNREGISTERED — this
    # assembler takes train files only and cuts its own 3% per-family package
    # holdout, so a per-family eval input has no slot here.
    "cases_v1/astfim_partial.jsonl",
    "cases_v1/removed_block_comment.jsonl",  # dev one-liner marks the site;
                                      # target re-inserts the removed block
    # backend variants (same families as their mains, one stamp per family)
    # + the agy proof families, kept for diversity
    "cases_v1/expectation_completion_zai.jsonl",
    "cases_v1/trycatch_handler_completion_zai.jsonl",
    "cases_v1/comment_styles_zai.jsonl",
    "cases_v1/tidyselect_completion.jsonl",
    "cases_v1/comment_to_code_styles.jsonl",
]

# Families can outgrow their useful mixture share. Cap a family's rows
# BEFORE the package holdout; sampling is seeded so reruns are stable.
FAMILY_CAPS = {
    "roxygen_drafting": 40000,
    "no_op": 8000,
    "mid_roxygen": 10000,
}

stats = Counter()
notes = []
# suffix-convention verification bookkeeping (deliverable: per-family verdict)
sfx = defaultdict(lambda: Counter())


def row(prompt, target, family, pkg):
    return dict(text=prompt + target, prompt=prompt, target=target,
                family=family, package_or_repo=pkg, has_types=False)


def content_of(target):
    """Target text with the >>>>>>> UPDATED terminator removed."""
    body = target
    if body.endswith(">>>>>>> UPDATED"):
        body = body[: -len(">>>>>>> UPDATED")]
    return body.rstrip("\n")


def edit_row(ex, family, pkg, fd=None, cursor_after=None):
    """zeta2 render under the SUFFIX convention.

    The cursor sits at the first CHANGED line of the region (line index
    `cursor_after` gets the marker appended); the target is region_new from
    that line on — the unchanged head lines above the cursor are NOT
    re-emitted (v2 predicted the whole region_new). `fd`/`cursor_after`
    override the derived placement for families that carry their own
    convention (no_op, mid_roxygen).
    """
    sfx[family]["rows"] += 1
    ro, rn = ex.get("region_old") or [], ex.get("region_new") or []
    if family != "no_op" and not [l for l in rn if l.strip()]:
        stats[f"drop:{family}:empty_region_new"] += 1
        sfx[family]["dropped_empty"] += 1
        return None
    if fd is None:  # first differing line; prefix-equal -> whole old region
        fd = next((i for i in range(min(len(ro), len(rn)))
                   if ro[i] != rn[i]), min(len(ro), len(rn)))
    if fd > 0:
        sfx[family]["head_stripped_rows"] += 1
        sfx[family]["head_lines_stripped"] += fd
    tgt_lines = rn[fd:]
    if family == "no_op":
        # no rstrip: a blank line target must stay a real blank line
        if tgt_lines and all(not l.strip() for l in tgt_lines):
            target = "\n" * len(tgt_lines) + UPDATED
        else:
            target = "\n".join(tgt_lines) + UPDATED
    else:
        target = "\n".join(tgt_lines).rstrip() + UPDATED
        if not content_of(target).strip():
            stats[f"drop:{family}:empty_after_head_strip"] += 1
            sfx[family]["dropped_empty"] += 1
            return None
    ex = dict(ex)
    ex.setdefault("suffix", [])  # scenario rows carry prefix context only
    ex["cursor_idx"] = cursor_after if cursor_after is not None \
        else max(fd - 1, 0)
    prompt = render_zeta2(ex)
    if len(prompt) + len(target) > MAX_CHARS:
        stats[f"drop:{family}:over_{MAX_CHARS}"] += 1
        return None
    return row(prompt, target, family, pkg)


def comment_to_code_row(filename, comment, code, family, pkg):
    """zeta2 empty-region style: last prefix line is a comment, region = cursor."""
    sfx[family]["rows"] += 1
    parts = (["<[fim-suffix]>"] +
             [f"<[fim-prefix]><filename>{filename}"] +
             [comment.rstrip()] +
             ["<<<<<<< CURRENT", "<|user_cursor|>", "=======", "<[fim-middle]>"])
    prompt = "\n".join(parts)
    body = code.rstrip()
    if body.endswith(">>>>>>> UPDATED"):  # never emit a second marker
        body = body[: -len(">>>>>>> UPDATED")].rstrip()
        notes.append(f"{family}: stripped pre-existing UPDATED marker")
    if not body.strip():
        stats[f"drop:{family}:empty_target"] += 1
        sfx[family]["dropped_empty"] += 1
        return None
    return row(prompt, body + UPDATED, family, pkg)


# ---------------------------------------------------------------------------
# 1. finish_block: suffix-convention RE-RENDER from the source records
# ---------------------------------------------------------------------------

FINISH_SRC = REPO / "experiments/synthetic-data/finish_block_sample.jsonl"
MAX_CHARS_V1 = 6000
MIN_TARGET_CHARS = 30  # v1 convention


def render_finish_block(rec):
    """format_sft_v1.py render, with the v5 mid_body head-strip.

    Returns (prompt, target, converted) or (None, None, False) when nothing
    remains after the strip (row must be dropped).
    """
    prefix_lines = rec["prefix"].splitlines()
    target = rec["target"]
    # region content the user has so far: empty for signature kind; head for
    # mid_body — the cursor marker goes after the LAST head line
    if rec["kind"] == "signature":
        region = ["<|user_cursor|>"]
        converted = False
    else:
        head = rec["prefix"].split("{\n", 1)[-1] if "{\n" in rec["prefix"] else ""
        head_lines = head.splitlines()
        region = [l + ("\n" + "<|user_cursor|>" if i == len(head_lines) - 1
                       else "") for i, l in enumerate(head_lines)] \
            or ["<|user_cursor|>"]
        # v5 suffix convention: the target must not re-emit the head the
        # user already typed. Old-style records carried the FULL body here;
        # strip the duplicated head lines defensively (comparison is
        # \r-tolerant: CRLF sources leave \r residue in raw targets).
        # (head "" -> [""] is the no-"{\n" fallback, not a real head: never
        # strip it.)
        tgt_lines = target.splitlines()
        converted = False
        if head_lines and head_lines != [""] \
                and [l.rstrip("\r") for l in tgt_lines[:len(head_lines)]] \
                == [l.rstrip("\r") for l in head_lines]:
            tgt_lines = tgt_lines[len(head_lines):]
            converted = True
            if not tgt_lines:
                return None, None, False
            target = "\n".join(tgt_lines)
    prompt_lines = (["<[fim-suffix]>"] +
                    [f"<[fim-prefix]><filename>{rec['package']}/{rec['path']}"] +
                    prefix_lines + ["<<<<<<< CURRENT"] + region +
                    ["=======", "<[fim-middle]>"])
    return "\n".join(prompt_lines), target.rstrip() + UPDATED, converted


def load_finish_block():
    fam = "finish_block"
    recs = [json.loads(l) for l in open(FINISH_SRC)]
    # exact-target dedup (v1 convention, keep first)
    seen, uniq = set(), []
    for r in recs:
        h = " ".join(r["target"].split())
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    stats[f"{fam}:source"] = len(recs)
    stats[f"{fam}:after_dedup"] = len(uniq)
    # v1 package-level split (5% of packages, seed 11) — kept identical so
    # v5 evals stay comparable with v1-v4
    pkgs = sorted({r["package"] for r in uniq})
    rng = random.Random(11)
    rng.shuffle(pkgs)
    eval_pkgs = set(pkgs[: max(1, len(pkgs) // 20)])
    n_mid = n_conv = n_drop_empty = 0
    for r in uniq:
        prompt, target, converted = render_finish_block(r)
        if prompt is None:
            if r["kind"] == "mid_body":
                n_drop_empty += 1
            stats[f"drop:{fam}:{r['kind']}:empty_after_head_strip"] += 1
            continue
        if r["kind"] == "mid_body":
            n_mid += 1
            n_conv += converted
        if len(prompt) + len(target) > MAX_CHARS_V1:
            stats[f"drop:{fam}:over_{MAX_CHARS_V1}"] += 1
            continue
        if len(target) < MIN_TARGET_CHARS:
            stats[f"drop:{fam}:short_target"] += 1
            continue
        split = "eval" if r["package"] in eval_pkgs else "train"
        yield split, row(prompt, target, fam, r["package"])
        stats[f"{fam}:{split}"] += 1
    stats[f"{fam}:mid_body_rows"] = n_mid
    stats[f"{fam}:mid_body_converted"] = n_conv
    stats[f"{fam}:mid_body_dropped_empty"] = n_drop_empty
    stats[f"{fam}:eval_packages"] = len(eval_pkgs)
    sfx[fam].update(rows=n_mid + sum(1 for r in uniq if r["kind"] == "signature"),
                    head_stripped_rows=n_conv, dropped_empty=n_drop_empty)


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
    """Canonical scenario files; per FAMILY hold out 3% of that family's
    packages (min 1, seed 42) so every scenario family has eval coverage."""
    recs = []
    fam_pkgs = defaultdict(set)
    for fname in SCENARIO_FILES:
        # wave1 cases live under NAS/cases_v1: entries containing "/" are
        # NAS-relative; plain filenames stay under scenarios_v1/
        path = (NAS / fname) if "/" in fname else (NAS / "scenarios_v1" / fname)
        if not path.exists():
            # e.g. comment_to_code_synthetic.jsonl quarantined as .bak while
            # the (fixed) generator rebuilds it; assemble without the family.
            notes.append(f"scenarios: {fname} missing -> skipped (0 rows)")
            continue
        by_fam = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            # comment_to_code_gemini rows carry prefix: null (comment at file
            # start); the renderer needs a list
            if r.get("prefix") is None:
                r["prefix"] = []
            by_fam[r["family"]].append(r)
        for fam, rows in by_fam.items():
            cap = FAMILY_CAPS.get(fam)
            if cap is not None and len(rows) > cap:
                notes.append(f"scenarios: {fam} capped {len(rows)} -> {cap} "
                             f"(seeded sample)")
                rng = random.Random(123)
                idx = sorted(rng.sample(range(len(rows)), cap))
                rows = [rows[i] for i in idx]
            recs.extend(rows)
            for r in rows:
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
        if fam == "no_op":
            # target built by the family's own convention: empty / one blank
            rr = edit_row(r, fam, r["package"], fd=0,
                          cursor_after=r.get("cursor_idx", 0))
        elif fam == "mid_roxygen":
            # region_new already holds ONLY the lines after the cursor (the
            # stored cursor_idx is the line the cursor follows); a generic
            # first-diff search would misplace it on repeated roxygen lines
            rr = edit_row(r, fam, r["package"], fd=0,
                          cursor_after=r.get("cursor_idx", 0))
        else:
            rr = edit_row(r, fam, r["package"])
        if rr:
            yield split, rr
            stats[f"{fam}:{split}"] += 1


def load_pr_instructed():
    """zeta2 with the edit_history section replaced by a reviewer comment line.

    Cursor follows the v5 suffix convention: first line where region_old
    differs from region_new (fallback: last line of region_old); the target
    is region_new from that line on.
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
        fd = next((i for i in range(min(len(ro), len(rn))) if ro[i] != rn[i]),
                  min(len(ro), len(rn)))
        ex = dict(suffix=[], path=r["path"], event_diff="",
                  prefix=r["prefix"] + [f"# reviewer: {instr}"],
                  region_old=ro, region_new=rn, cursor_idx=max(fd - 1, 0))
        rr = edit_row(ex, fam, r["repo"], fd=fd, cursor_after=max(fd - 1, 0))
        if rr:
            yield "train", rr
            stats[f"{fam}:train"] += 1
    if n_multiline:
        notes.append(f"pr_instructed: {n_multiline} instruction(s) contained "
                     f"newlines, whitespace-collapsed to one comment line")


def load_synthetic_analyst():
    fam = "synthetic_analyst"
    # analyst_scripts.jsonl: the detached 3-source generator (grow-only);
    # analyst_direct.jsonl: the burst generator (deduped on merge);
    # analyst_gemini.jsonl: the agy gemini-3.7-flash harvest (same schema)
    for fname in ("analyst_scripts.jsonl", "analyst_direct.jsonl",
                  "analyst_gemini.jsonl"):
        for line in open(NAS / "synthetic_analyst_v1" / fname):
            r = json.loads(line)
            rr = comment_to_code_row("analyst.R", f"# {r['intent'].strip()}",
                                     r["code"], fam, "synthetic_analyst_v1")
            if rr:
                yield "train", rr
                stats[f"{fam}:train"] += 1


def load_hidden_r_instruction(n_total=40_000, n_eval=1_000, seed=5):
    """Stratified sample: ALL of codex_r_strict + random ling_coder fill.

    Plain-text general-instruction tail (NOT zeta2, no cursor concept).
    package_or_repo is the per-row source id (mid / dataset_row) — these
    rows have no repo concept. Eval holdout of n_eval is drawn
    proportionally per stratum (seed 5).
    """
    fam = "hidden_r_instruction"

    def render(inp, outp):
        prompt = f"### Instruction:\n{inp.strip()}\n\n### Response:\n"
        return row(prompt, outp.rstrip() + "\n", fam, None)  # pkg by caller

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
    # proportional stratified holdout: n_eval/n_total per stratum (first k
    # of the seeded shuffles above -> reproducible)
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


ALREADY_COMPLIANT = {
    "comment_to_code_real": "compliant: empty region above the cursor "
                            "(region_old=[''], cursor line 0); target is "
                            "generated code only, no re-emitted partial",
    "comment_to_code_synthetic": "compliant: same empty-region construction",
    "comment_drafting": "compliant: empty region above the cursor; target "
                        "is the drafted comment only",
    "roxygen_drafting": "compliant: empty region above the cursor; target "
                        "is the drafted roxygen block only",
    "synthetic_analyst": "compliant: comment_to_code_row — cursor on an "
                         "empty region, target is generated code only",
    "paper_to_r": "compliant: comment_to_code_row — cursor on an empty "
                  "region, target is generated code only",
    "hidden_r_instruction": "n/a: plain-text instruction family, no "
                            "cursor/edit format",
    "no_op": "compliant by construction: target empty (after_close_brace) "
             "or a single blank line (blank_between)",
    "mid_roxygen": "compliant by construction: target = the roxygen lines "
                   "after the cursor only",
}


def suffix_report():
    """Per-family verdict for the suffix-convention verification."""
    out = {}
    for fam in sorted(set(sfx) | set(ALREADY_COMPLIANT)):
        c = sfx.get(fam, Counter())
        if fam in ("edit_pairs", "pr_instructed"):
            verdict = ("re-emitted the unchanged head above the cursor in v2; "
                       "v5 strips it (target = region_new from the first "
                       "changed line on)")
        elif fam == "finish_block":
            verdict = ("mid_body re-rendered under the suffix convention: "
                       "head kept in region_old, target stripped of any "
                       "re-emitted head lines")
        elif fam in ("rename_propagation", "pipe_rewrite", "na_rm_propagation",
                     "format_propagation", "doc_sync"):
            verdict = ("compliant at line granularity after the v5 cursor "
                       "fix: v2 read the stored char-offset cursor_idx as a "
                       "line index and silently dropped the cursor marker; "
                       "v5 places it at the first changed line and strips "
                       "duplicated head lines (doc_sync/format rows had them)")
        else:
            verdict = ALREADY_COMPLIANT.get(fam, "")
        out[fam] = dict(verdict=verdict, **{k: v for k, v in sorted(c.items())})
    return out


def spot_check(train, evals):
    """Print converted/representative rows for manual inspection."""
    print("\n--- spot check: 3 finish_block mid_body rows (target must NOT "
          "re-emit the head above the cursor) ---")
    shown = 0
    for r in train + evals:
        if r["family"] != "finish_block" or shown >= 3:
            continue
        p = r["prompt"]
        region = p.split("<<<<<<< CURRENT\n", 1)[1].split("\n=======", 1)[0]
        if region == "<|user_cursor|>":
            continue  # signature kind: no head above the cursor
        head = region.rsplit("\n<|user_cursor|>", 1)[0]
        head_lines = head.splitlines()
        tgt_first = r["target"].splitlines()[0] if r["target"].splitlines() \
            else ""
        print(f"[{shown + 1}] head lines above cursor: {len(head_lines)} "
              f"(last: {head_lines[-1][:60]!r}) | target line 1: "
              f"{tgt_first[:60]!r} | target re-emits head tail: "
              f"{tgt_first == (head_lines[-1] if head_lines else None)}")
        shown += 1
    print(f"finish_block mid_body: rows={stats['finish_block:mid_body_rows']} "
          f"converted={stats['finish_block:mid_body_converted']} "
          f"dropped_empty={stats['finish_block:mid_body_dropped_empty']} "
          f"(the corpus generator already emitted tail-only targets, so a "
          f"conversion count of 0 is the expected verdict)")
    print("\n--- spot check: 2 no_op rows (one of each kind) ---")
    picked = {}
    for r in train:
        if r["family"] == "no_op" and r["target"] not in picked:
            picked[r["target"]] = r
            if len(picked) == 2:
                break
    for i, (tgt, r) in enumerate(sorted(picked.items(),
                                        key=lambda kv: len(kv[0]))):
        kind = "after_close_brace" if tgt == UPDATED else "blank_between"
        print(f"[{i + 1}] kind={kind} target={tgt!r}")
        print("    prompt tail:", repr(r["prompt"][-220:]))


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output dir (default: the live sft_v5)")
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

    # ---- validation: empty targets allowed ONLY for family=no_op ----
    empty_by_fam = Counter()
    for r in train + evals:
        if not content_of(r["target"]).strip():
            empty_by_fam[r["family"]] += 1
    bad = {f: n for f, n in empty_by_fam.items() if f != "no_op"}
    assert not bad, f"empty targets outside no_op: {bad}"
    stats["empty_target_rows_no_op"] = empty_by_fam.get("no_op", 0)
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
                            p99=pct(eval_lengths, 99),
                            max=eval_lengths[-1] if eval_lengths else 0),
        per_family_train_p50_p95={f: [pct(v, 50), pct(v, 95)]
                                  for f, v in per_fam_len.items()},
        checks=dict(duplicate_text_rows_dropped=n_dup,
                    empty_targets_outside_no_op=0,
                    empty_target_rows_no_op=empty_by_fam.get("no_op", 0),
                    eval_train_pkg_overlap=overlap),
        suffix_convention=suffix_report(),
        notes=notes,
    )
    (OUT / "stats.json").write_text(json.dumps({"report": report,
                                                "source_counts": dict(stats)},
                                               indent=1))
    print(json.dumps(report, indent=1))
    spot_check(train, evals)
    print(f"\nper-family train/eval counts:")
    for f in sorted(set(fam_tr) | set(fam_ev)):
        print(f"  {f:28s} train={fam_tr.get(f, 0):7d} eval={fam_ev.get(f, 0):6d}")


if __name__ == "__main__":
    main()
