#!/usr/bin/env python3
"""Build intent_suite_v1.jsonl: Zed-style plain-English intent-assertion cases.

Two sources:
  1. Five hand-written live cases from the 2026-08-19 extension session
     (sd-line duplication, post-brace eagerness, mid-roxygen, mom-glue,
     comment-to-code) — see docs/research/2026-08-19-night-results.md.
  2. Rows sampled from the held-out sft_v3 eval split across the scenario
     families, reconstructed into the extension's suffix-completion input
     shape by splitting each SFT prompt at its markers.

Input shape (what the extension's buildPrompt consumes):
  filename            relative path of the file being edited
  prefix_lines        lines above the cursor line
  cursor_partial      text typed on the cursor line before the cursor
  suffix_lines        after-cursor text of the cursor line, then the lines below
  edit_history_lines  optional recent-diff body (scenario families keep the
                      edit history their intent depends on; live cases have none)

Cursor placement when the SFT row has no explicit <|user_cursor|> marker
(rename/pipe/doc_sync/format rows without one): the cursor goes at the END of
the region's last line — the typed partial is the current text of the line the
edit targets. Rows with the marker use it directly; region lines after the
cursor line move to the head of the suffix (they are below the cursor).

Prefix/suffix are bounded the way the extension bounds them (6000 chars for
prefix+suffix, prefix truncated from its start) with the suffix additionally
capped at 2500 chars from its head so prompts fit -c 8192 with margin.

gt_completion (parsed target) is stored per derived row — the runner uses one
as its ground-truth calibration anchor.
"""
import json
import random
import re
from pathlib import Path

EVAL = Path("/mnt/h/sepalith/datasets/sft_v3/eval.jsonl")
OUT = Path(__file__).resolve().parent / "intent_suite_v1.jsonl"
SEED = 20260819

MAX_TOTAL = 6000   # extension's prefix+suffix char budget
MAX_SUFFIX = 2500  # head-cap so prompts stay well inside -c 8192

SAMPLE = {  # family -> n to sample from held-out eval
    "rename_propagation": 8,
    "pipe_rewrite": 4,
    "format_propagation": 6,
    "doc_sync": 5,
    "roxygen_drafting": 5,
    "comment_drafting": 4,
    "finish_block": 7,
}

IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9._]{2,}\b")
STOPWORDS = set("""function if else for while return TRUE FALSE NULL NA
library require namespace package::""".split())


def parse_prompt(prompt):
    """Split an SFT prompt at its markers into the extension input shape."""
    lines = prompt.split("\n")
    try:
        i_pre = next(i for i, l in enumerate(lines) if l.startswith("<[fim-prefix]>"))
    except StopIteration:
        return None
    suffix = lines[1:i_pre]  # after the leading <[fim-suffix]> line

    # the marker line itself carries the first <filename> tag: either the
    # edit_history pseudo-file or the real file being edited
    head = lines[i_pre][len("<[fim-prefix]>"):]
    edit_history = []
    if head.strip() == "<filename>edit_history":
        rest = lines[i_pre + 1:]
        try:
            j = next(i for i, l in enumerate(rest) if l.startswith("<filename>"))
        except StopIteration:
            return None
        edit_history = rest[:j]
        while edit_history and not edit_history[-1].strip():
            edit_history.pop()
        while edit_history and not edit_history[0].strip():
            edit_history.pop(0)
        rest = rest[j:]
        head = rest[0]
    else:
        rest = lines[i_pre + 1:]

    if not head.startswith("<filename>"):
        return None
    filename = head[len("<filename>"):]
    rest = rest[1:]

    try:
        i_cur = next(i for i, l in enumerate(rest) if l.strip() == "<<<<<<< CURRENT")
        i_eq = next(i for i, l in enumerate(rest) if l.strip() == "=======")
    except StopIteration:
        return None
    prefix = rest[1:i_cur]  # drop the <filename> line
    region = rest[i_cur + 1:i_eq]
    if "<[fim-middle]>" not in rest[i_eq + 1:i_eq + 3]:
        return None

    cidx = [i for i, l in enumerate(region) if "<|user_cursor|>" in l]
    if cidx:
        ci = cidx[0]
        before, after = region[ci].split("<|user_cursor|>", 1)
        prefix += region[:ci]
        below = region[ci + 1:]
    else:
        if not region:
            return None
        prefix += region[:-1]
        before, after = region[-1], ""
        below = []
    suffix = ([after] if after else [""]) + below + suffix
    return dict(filename=filename, prefix_lines=prefix, cursor_partial=before,
                suffix_lines=suffix, edit_history_lines=edit_history)


def parse_target(target):
    """Target text -> completion lines via the extension's parse rules."""
    text = target.split(">>>>>>> UPDATED", 1)[0]
    marker = re.compile(r"^\s*(<<<<<<<\s*CURRENT|=======|>>>>>>>\s*UPDATED|<\[fim-(middle|prefix|suffix)\]>|<\|user_cursor\|>)\s*$")
    lines = [l.replace("\r", "") for l in text.split("\n")]
    lines = [l for l in lines if not marker.match(l)]
    lines = [l.replace("<|user_cursor|>", "") for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def bound(inp):
    """Extension-style budget: suffix counts first, prefix truncated from start."""
    s = inp["suffix_lines"]
    kept, used = [], 0
    for l in s:
        if used + len(l) + 1 > MAX_SUFFIX:
            break
        kept.append(l)
        used += len(l) + 1
    s = kept
    p = inp["prefix_lines"]
    budget = max(0, MAX_TOTAL - sum(len(l) + 1 for l in s))
    kept, used = [], 0
    for l in reversed(p):
        if used + len(l) + 1 > budget:
            break
        kept.append(l)
        used += len(l) + 1
    inp["prefix_lines"] = list(reversed(kept))
    inp["suffix_lines"] = s
    return inp


def diff_names(edit_history):
    """Identifiers added/removed by the edit-history diff (for rename assertions)."""
    minus, plus = [], []
    for l in edit_history:
        if l.startswith("-") and not l.startswith("---"):
            minus += [t for t in IDENT.findall(l) if t not in STOPWORDS]
        elif l.startswith("+") and not l.startswith("+++"):
            plus += [t for t in IDENT.findall(l) if t not in STOPWORDS]
    added = [t for t in plus if t not in minus]
    removed = [t for t in minus if t not in plus]
    uniq = lambda xs: sorted(set(xs))
    a, r = uniq(added), uniq(removed)
    if len(a) == 1 and len(r) == 1:
        return r[0], a[0]
    return None


ASSERTIONS = {
    "pipe_rewrite": "the completion rewrites the current line(s) with the new pipe operator shown in the edit history, replacing the old pipe and preserving the rest of the expression",
    "format_propagation": "the completion applies the same reformatting shown in the edit history to the current line(s): the same code restyled (e.g. one argument per line), not different logic",
    "doc_sync": "the completion updates the roxygen documentation to reflect the signature change in the edit history (e.g. documents the new or changed argument) while keeping the existing #' lines",
    "roxygen_drafting": "the completion drafts roxygen documentation lines starting with #' (with tags such as @param/@return/@details where fitting) for the function below the cursor, not raw function-body code",
    "comment_drafting": "the completion drafts a short plain # comment describing the code below the cursor — a comment, not executable code",
    "finish_block": "the completion continues the unfinished function or block with valid R statements that plausibly implement it (no prompt markers, no echoed diff)",
}


def family_assertion(family, edit_history, target):
    if family == "rename_propagation":
        names = diff_names(edit_history)
        if names:
            old, new = names
            return (f"the completion's predicted line reflects the rename shown in "
                    f"the edit history: it uses the new name ({new}); a line that "
                    f"ignores the rename and keeps only the old name ({old}) or is "
                    f"unrelated fails (the old name may remain where the original "
                    f"line legitimately kept it, e.g. on the right-hand side)")
        return ("the completion's predicted line reflects the rename shown in the "
                "edit history: it uses the new name; keeping only the old name or "
                "unrelated content fails")
    return ASSERTIONS[family]


def live_cases():
    sd = dict(
        id="live-sd-line",
        input={"filename": "R/summarize.R",
               "prefix_lines": ["summarize_vec <- function(x) {",
                                "  # drop missing values, then compute both moments",
                                "  mean_value <- mean(x, na.rm = TRUE)"],
               "cursor_partial": "  sd_value <- sd(x, na.",
               "suffix_lines": ["", "]"]},
        assertion="the completion continues the current line, starting with something like rm = TRUE), and does NOT re-emit the line from its beginning",
        source="live:2026-08-19:sd-line-duplication", family="live")
    sd["input"]["suffix_lines"] = ["", "}"]
    post_brace = dict(
        id="live-post-brace",
        input={"filename": "R/summarize.R",
               "prefix_lines": ["summarize_vec <- function(x) {",
                                "  mean_value <- mean(x, na.rm = TRUE)",
                                "  sd_value <- sd(x, na.rm = TRUE)",
                                "  list(mean = mean_value, sd = sd_value)"],
               "cursor_partial": "}",
               "suffix_lines": [""]},
        assertion="the completion is empty or whitespace/blank-line only; it must NOT invent new function body code",
        source="live:2026-08-19:post-brace-eagerness", family="live")
    mid_roxy = dict(
        id="live-mid-roxygen",
        input={"filename": "R/summarize.R",
               "prefix_lines": ["#' Summarize a numeric vector",
                                "#'",
                                "#' @param x numeric vector to summarize"],
               "cursor_partial": "#' @param na_rm logical; if TRUE, missing values are dropped before summarizing",
               "suffix_lines": ["", "#' @return a list with elements mean and sd", "#' @export",
                                "summarize_vec <- function(x, na_rm = TRUE) {",
                                "  mean_value <- mean(x, na.rm = TRUE)",
                                "  sd_value <- sd(x, na.rm = TRUE)",
                                "  list(mean = mean_value, sd = sd_value)",
                                "}"]},
        assertion="the completion continues the roxygen block (more #' lines), not function body code",
        source="live:2026-08-19:mid-roxygen", family="live")
    mom_glue = dict(
        id="live-mom-glue",
        input={"filename": "R/mom.R",
               "prefix_lines": ["#' Estimate the dispersion parameter", "#'"],
               "cursor_partial": "#' Method-of-moments estimate from the squared residuals",
               "suffix_lines": ["#'",
                                "#' @param x numeric vector of observations",
                                "#' @param mu the mean around which dispersion is computed",
                                "#' @return a single numeric dispersion estimate",
                                "mom_dispersion <- function(x, mu) {",
                                "  mean((x - mu)^2)",
                                "}"]},
        assertion="the completion stays on the comment format (#' or nothing), never raw code on the same line",
        source="live:2026-08-19:mom-glue", family="live")
    c2c = dict(
        id="live-comment-to-code",
        input={"filename": "R/geom.R",
               "prefix_lines": ["# Computes the geometric mean of a numeric vector, ignoring non-positive values"],
               "cursor_partial": "",
               "suffix_lines": ["", "#' next: harmonic mean"]},
        assertion="the completion starts an R function signature (<- function(...) or similar)",
        source="live:2026-08-19:comment-to-code", family="live")
    return [sd, post_brace, mid_roxy, mom_glue, c2c]


def main():
    rows = [json.loads(l) for l in open(EVAL)]
    by_fam = {}
    for i, r in enumerate(rows):
        by_fam.setdefault(r["family"], []).append((i, r))

    rng = random.Random(SEED)
    cases = live_cases()

    for fam, n in SAMPLE.items():
        pool = by_fam.get(fam, [])
        picked, seen_pkgs = [], set()
        order = list(pool)
        rng.shuffle(order)
        for i, r in order:
            if len(picked) >= n:
                break
            inp = parse_prompt(r["prompt"])
            if inp is None or not inp["filename"].endswith(".R"):
                continue
            bound(inp)
            if not inp["prefix_lines"] and not inp["cursor_partial"]:
                continue
            gt = parse_target(r["target"])
            if not gt:
                continue
            pkg = r.get("package_or_repo") or "?"
            picked.append(dict(
                id=f"{fam}-{i}",
                input=inp,
                assertion=family_assertion(fam, inp["edit_history_lines"], gt),
                source=f"sft_v3/eval.jsonl:{fam}#row{i}:{pkg}",
                family=fam, gt_completion=gt))
            seen_pkgs.add(pkg)
        cases += picked
        print(f"{fam}: sampled {len(picked)}/{n} (packages: {len(seen_pkgs)})", flush=True)

    with open(OUT, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
