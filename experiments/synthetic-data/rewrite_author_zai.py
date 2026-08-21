#!/usr/bin/env python3
"""rewrite_author_zai.py — the REAL author-under-constraint loop (zai backend).

Replaces the mock author of rewrite_maint_proto.py's `constraint` command with
glm-5.3 and mass-produces validated rewrite/fix-issue rows:

  mine      (wave 1) corpus functions with mechanical lint findings; the
            author-LLM must fix exactly the listed findings, minimal diff.
  inject    (wave 1) dirty-twin injection (T/F, paste-sep, sapply, 1:length);
            target = the corpus original (tier-1 exact GT).
  buinject  (wave 2) fix-issue arm: injected defects (boundary operator,
            char-swap, wrong variable); author produces the FIX, target =
            corpus original, trivially exact validator.

Every authored result goes through the FULL gate stack proven with the mock
(parse / splice / diff-minimality / LOC / lint-delta / jarl-agreement /
gt-corpus-exact on inject arms / row_check / behavior where callable).
Rows use the cases-adjacent schema WITH the parent-link contract (content-
hash base_sample_id + rule/spec id + version) and family names
rewrite_lint_fix (mine+inject) / fix_issue_inject (buinject).

Resume-safe: <out>.done.jsonl sidecar (agy_generators convention; terminal
outcomes only — backend/network errors are retried on rerun). Crash-safe
partials, drvfs write retries, batched stats logging, <=4 in flight, global
429 circuit breaker with exponential pauses (shared quota with other
consumers — never starve them).

Usage (system python3 from experiments/synthetic-data):
  python3 rewrite_author_zai.py mine [--time-budget 1800 ...]
  python3 rewrite_author_zai.py author --arms mine,inject [--max N]
  python3 rewrite_author_zai.py author --arms buinject
  python3 rewrite_author_zai.py stats          # print current stats files

ZAI_API_KEY comes from the environment (exported from ~/.zshrc).
"""
from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # scenarios/, cases/, proto

import scenarios as S                             # noqa: E402
import cases.corpus as C                          # noqa: E402
import cases.validators as V                      # noqa: E402
from cases.compound import BaseSample             # noqa: E402
from cases.backends import (BackendError, ZaiBackend, extract_json_object,
                            strip_fences)         # noqa: E402
import rewrite_maint_proto as RW                  # noqa: E402

OUT_DIR = HERE / "results" / "rewrite_author_zai"
POOL_PATH = OUT_DIR / "spec_pool.jsonl"
JARL_TMP = RW.JARL_TMP                            # shared tmpfs workdir

DATASETS = Path("/mnt/h/sepalith/datasets/cases_v1")
FAMILY_OUT = {                                    # arm -> (family, out file)
    "mine": ("rewrite_lint_fix", DATASETS / "rewrite_lint_fix.jsonl"),
    "inject": ("rewrite_lint_fix", DATASETS / "rewrite_lint_fix.jsonl"),
    "buinject": ("fix_issue_inject", DATASETS / "fix_issue_inject.jsonl"),
}

# jarl rule aliases for detector agreement (mirrors run_case in the proto)
ALIAS = {"seq_safety": "seq", "class_equals": "class_equals",
         "true_false_symbol": "true_false_symbol",
         "paste0_sep": None, "sapply_vapply": None,
         "boundary_operator": None, "char_swap": None, "wrong_variable": None}

# user-facing issue text per (arm, rule) — never leaks the exact fix
PROMPT_RATIONALE = {
    ("mine", "true_false_symbol"): "T/F can be shadowed; spell the logical",
    ("mine", "seq_safety"): "1:x yields c(1,0) when x is empty; use the safe idiom",
    ("mine", "paste0_sep"): 'paste(x, y, sep="") is paste0(x, y)',
    ("mine", "sapply_vapply"): "sapply is type-unsafe here; vapply fails loudly on type drift",
    ("mine", "class_equals"): 'class(x)=="foo" breaks on S3 subclasses; use inherits()',
    ("inject", "true_false_symbol"): "T/F can be shadowed; spell the logical",
    ("inject", "paste0_sep"): 'paste(x, y, sep="") is paste0(x, y)',
    ("inject", "sapply_vapply"): "sapply is type-unsafe here; vapply fails loudly on type drift",
    ("inject", "seq_safety"): "1:x yields c(1,0) when x is empty; use the safe idiom",
    ("buinject", "boundary_operator"): "off-by-one: this loop/if boundary comparison uses the wrong operator",
    ("buinject", "char_swap"): "typo: this identifier is a misspelling of a name declared in this function",
    ("buinject", "wrong_variable"): "wrong variable: this identifier should be a different local of this function",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# drvfs-safe writers (cases.generate conventions)
# ---------------------------------------------------------------------------

def _append_line(path: Path, obj: dict, tries: int = 20, wait_s: float = 30.0):
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
            print(f"  [drvfs-write] {e}; retry {attempt + 1} in {wait_s:.0f}s",
                  flush=True)
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
            time.sleep(wait_s)


# ---------------------------------------------------------------------------
# PHASE 1: mine the spec pool (deterministic, zero quota)
# ---------------------------------------------------------------------------

def cmd_mine(args) -> int:
    rng = random.Random(args.seed)
    existing: set[str] = set()
    if POOL_PATH.exists() and not args.fresh:
        for line in POOL_PATH.read_text().splitlines():
            try:
                existing.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
    pool_pkgs = RW.sample_packages(rng, args.tidy_packages, args.random_packages)
    print(f"[mine] pool: {len(pool_pkgs)} packages "
          f"(tidy={args.tidy_packages} rest={args.random_packages}, "
          f"seed={args.seed}); {len(existing)} specs already in pool")

    quota = dict(mine=args.mine_specs, inject=args.inject_specs,
                 buinject=args.buinject_specs)
    stats = dict(functions_scanned=0, specs_emitted=0, per_rule=Counter(),
                 per_arm=Counter(), findings_total=0, packages_seen=set())
    site = len(existing)
    specs_buf: list[dict] = []
    t0 = time.time()

    def emit(bs: BaseSample, findings: list[dict], arm: str) -> None:
        nonlocal site
        b = bs.b
        bs_base_id = RW.base_sample_id(bs)
        start = bs.fn.parent.start_byte \
            if (bs.fn.parent is not None
                and bs.fn.parent.type == "binary_operator") \
            else bs.fn.start_byte
        suffix = f":{findings[0]['sb']}" if arm == "buinject" else ""
        spec = dict(
            id=f"rw:{arm}:{bs_base_id}{suffix}", arm=arm,
            base_sample_id=bs_base_id, package=b.package, path=b.rel,
            fn_head=b.line_str(bs.head_row), rows=[bs.top_row, bs.r1],
            start=start, end=bs.fn.end_byte,
            top_row=bs.top_row, head_row=bs.head_row,
            nlines=b.nlines(),
            behavior_call=RW.fn_signature_callable(bs),
            findings=[dict(rule=f["rule"], row=f["row"],
                           erow=f.get("erow", f["row"]), sb=f["sb"], eb=f["eb"],
                           old=f["old"], new=f["new"], fix=f.get("fix"),
                           col=b.rowcol(f["sb"])[1], snippet=f["old"],
                           rationale=f["rationale"])
                      for f in findings],
            src_b64=base64.b64encode(b.src).decode("ascii"),
            generated_at=_now())
        specs_buf.append(spec)
        site += 1
        stats["specs_emitted"] += 1
        stats["per_arm"][arm] += 1
        stats["findings_total"] += len(findings)
        for f in findings:
            stats["per_rule"][f["rule"]] += 1

    try:
        for b in C.iter_bundles_highest(pool_pkgs, rng):
            if all(v <= 0 for v in quota.values()) or \
                    time.time() - t0 > args.time_budget:
                break
            stats["packages_seen"].add(b.package)
            for fn in (n for n in V._walk(b.tree.root_node)
                       if n.type == "function_definition"):
                if all(v <= 0 for v in quota.values()) or \
                        time.time() - t0 > args.time_budget:
                    break
                geom = C._fn_body(b, fn)
                if geom is None:
                    continue
                _body, _head, _r0, _r1, nb = geom
                if not 6 <= len(nb) <= 60:
                    continue
                stats["functions_scanned"] += 1
                try:
                    bs = BaseSample(b, fn, site)
                except ValueError:
                    continue

                def ok_span(f: dict) -> bool:
                    return b.line_str(f["row"]).strip() and \
                        f.get("erow", f["row"]) - f["row"] < 8

                # arm 1: mine real lint findings (all findings, one spec)
                if quota["mine"] > 0:
                    findings = RW.drop_overlaps(
                        [f for f in RW.detect_findings(bs) if ok_span(f)])
                    if findings:
                        emit(bs, findings[:6], "mine")
                        quota["mine"] -= 1
                        continue
                # arm 2: inject a dirty twin (reverse strip)
                if quota["inject"] > 0:
                    findings = RW.drop_overlaps(
                        [f for f in RW.detect_injectable(bs) if ok_span(f)])
                    if findings:
                        emit(bs, findings[:4], "inject")
                        quota["inject"] -= 1
                        continue
                # arm 3: bug injection, ONE defect per spec
                if quota["buinject"] > 0:
                    one = RW.drop_overlaps(
                        [f for f in RW.detect_bug_injectable(bs) if ok_span(f)])
                    for f in one[:args.bug_per_fn]:
                        if quota["buinject"] <= 0:
                            break
                        emit(bs, [f], "buinject")
                        quota["buinject"] -= 1
    except KeyboardInterrupt:
        print("[mine] interrupted; keeping buffered specs", flush=True)

    new = [s for s in specs_buf if s["id"] not in existing]
    dups = len(specs_buf) - len(new)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(POOL_PATH, "a") as fh:                      # local fs, no drvfs
        for s in new:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    total = len(existing) + len(new)
    report = dict(ts=_now(), seed=args.seed, elapsed_s=round(time.time() - t0, 1),
                  functions_scanned=stats["functions_scanned"],
                  packages_seen=len(stats["packages_seen"]),
                  new_specs=len(new), duplicate_ids_skipped=dups,
                  pool_total=total,
                  quota_left=dict(quota),
                  per_arm=dict(stats["per_arm"]),
                  per_rule=dict(stats["per_rule"].most_common()))
    _write_json(OUT_DIR / "mine_report.json", report)
    print(f"[mine] +{len(new)} specs ({dups} dups skipped) in "
          f"{report['elapsed_s']}s; pool now {total}; "
          f"per_arm={report['per_arm']} quota_left={quota}")
    return 0


# ---------------------------------------------------------------------------
# authoring: prompt construction + response extraction
# ---------------------------------------------------------------------------

PROMPT_HEADER = """You are fixing R code. Rewrite the function below, fixing EXACTLY the issue(s) listed and changing nothing else: do not reformat, do not re-indent untouched lines, do not rename anything beyond the fixes, do not add or remove comments or blank lines. The number of non-blank lines must not increase.

Issues (line numbers count from the function's first line, 1-based):
"""


def build_prompt(ctx: dict, feedback: str | None = None) -> str:
    issues = []
    for f in ctx["findings"]:
        why = PROMPT_RATIONALE.get((ctx["arm"], f["rule"]), f["rationale"])
        cur = f["new"] if ctx["arm"] in ("inject", "buinject") else f["old"]
        line = f["row"] - ctx["top_row"] + 1
        issues.append(f"- line {line}, col {f['col']} [{f['rule']}] "
                      f"`{cur}`: {why}")
    p = PROMPT_HEADER + "\n".join(issues)
    if feedback:
        p += (f"\n\nNOTE: a previous attempt was REJECTED by the verifier "
              f"because: {feedback} Carefully apply ONLY the listed fixes "
              f"and keep every other line exactly as given.")
    p += ("\n\nFunction:\n```r\n" + ctx["dirty_fn"] + "\n```\n\n"
          "Return ONLY a JSON object: "
          '{"function": "<the complete fixed function, including its '
          '`name <- function(...)` assignment, exactly as given except for '
          'the fixes>"}')
    return p


_FNSTART_RE = re.compile(
    r"(?m)^[ \t]*(?:[A-Za-z.][\w.$]*\s*(?:<-|=)\s*function\s*\(|function\s*\()")


def extract_fn_text(raw: str) -> tuple[str | None, str]:
    """Layer 1: JSON {'function': text}; fallback: slice the raw text from
    the first assignment/function line to the last '}' line."""
    obj = extract_json_object(strip_fences(raw))
    if obj is not None and isinstance(obj.get("function"), str) \
            and "function" in obj["function"]:
        return obj["function"], "json"
    t = strip_fences(raw)
    m = _FNSTART_RE.search(t)
    if m:
        lines = t[m.start():].split("\n")
        braces = [i for i, l in enumerate(lines) if l.rstrip().endswith("}")]
        if braces:
            return "\n".join(lines[:braces[-1] + 1]), "rawslice"
    return None, "none"


def _rnorm(t: str) -> list[str]:
    return [l.rstrip() for l in t.replace("\r\n", "\n").split("\n")]


def _norm_trim(t: str) -> list[str]:
    ls = _rnorm(t)
    while ls and not ls[0].strip():
        ls.pop(0)
    while ls and not ls[-1].strip():
        ls.pop()
    return ls


class ZaiAuthorBackend(ZaiBackend):
    """glm-5.3 authoring config: bigger budget for whole-function output,
    low temperature for minimal diffs; same pacing/retry/stats."""
    name = "zai"
    model = "glm-5.3"

    def _payload(self, prompt: str) -> dict:
        return {
            "model": "glm-5.3",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": 3500,
            "temperature": 0.3,
        }


# ---------------------------------------------------------------------------
# per-spec context + gates (the FULL stack from the proven mock loop)
# ---------------------------------------------------------------------------

def build_ctx(spec: dict) -> dict:
    src = base64.b64decode(spec["src_b64"])
    findings = spec["findings"]
    if spec["arm"] == "mine":
        dirty = src
    else:
        dirty = RW.apply_fixes(src, findings)
    start = spec["start"] + (0 if spec["arm"] == "mine"
                             else RW._span_delta(findings, spec["start"]))
    end = spec["end"] + (0 if spec["arm"] == "mine"
                         else RW._span_delta(findings, spec["end"]))
    return dict(spec=spec, arm=spec["arm"], findings=findings, src=src,
                dirty=dirty, start=start, end=end,
                top_row=spec["top_row"], head_row=spec["head_row"],
                dirty_fn=dirty[start:end].decode("utf-8", "replace"),
                corpus_fn=src[spec["start"]:spec["end"]]
                .decode("utf-8", "replace"))


def _tmp_r(prefix: str, text: str) -> Path:
    JARL_TMP.mkdir(parents=True, exist_ok=True)
    fh = tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     prefix=prefix, dir=str(JARL_TMP))
    fh.write(text)
    fh.close()
    return Path(fh.name)


def run_gates(ctx: dict, authored: str, site: int) -> dict:
    """All gates, mirroring rewrite_maint_proto.run_case for a REAL author."""
    g: dict = {}
    arm, spec = ctx["arm"], ctx["spec"]
    dirty_lines = _rnorm(ctx["dirty_fn"])
    auth_lines = _rnorm(authored)
    corpus_lines = _rnorm(ctx["corpus_fn"])

    # G7 corpus-exact GT (inject arms) — cheap first
    if arm in ("inject", "buinject"):
        g["gt_corpus_exact"] = auth_lines == corpus_lines

    # G4 LOC: never longer
    nb_d = sum(1 for l in dirty_lines if l.strip())
    nb_a = sum(1 for l in auth_lines if l.strip())
    g["loc"] = nb_a <= nb_d
    g["loc_delta"] = nb_a - nb_d

    # G3 diff minimality: changed lines within the finding covers
    sm = difflib.SequenceMatcher(a=dirty_lines, b=auth_lines, autojunk=False)
    changed = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "equal":
            changed.update(range(i1, max(i1 + 1, i2)))
    cover = RW._cover_rows(ctx["findings"], ctx["top_row"])
    g["diff_minimal"] = changed <= cover
    if not g["diff_minimal"]:
        g["diff_outside"] = sorted(changed - cover)[:6]

    # G1 parse (Rscript + tree-sitter fragment check)
    path = _tmp_r(f"authz_{arm}_{site}_", authored)
    try:
        try:
            r = subprocess.run(["Rscript", "-e",
                                f"invisible(parse('{path}'))"],
                               capture_output=True, text=True, timeout=30)
            g["parse"] = V.fragment_clean(authored) and r.returncode == 0
        except subprocess.TimeoutExpired:
            g["parse"] = False
    finally:
        path.unlink(missing_ok=True)

    # G2 splice: the whole FILE re-parses with the authored function in place
    spliced = ctx["dirty"][:ctx["start"]] + authored.encode("utf-8") + \
        ctx["dirty"][ctx["end"]:]
    tb = S.parser.parse(spliced)
    g["splice"] = not tb.root_node.has_error and \
        not any(n.type == "ERROR" or n.is_missing for n in V._walk(tb.root_node))
    g["_spliced"] = spliced          # for row construction (not emitted)

    # G5/G6 jarl lint-delta + detector agreement (tmpfs copies)
    bf = _tmp_r(f"authz_b_{site}_", ctx["dirty_fn"])
    af = _tmp_r(f"authz_a_{site}_", authored)
    try:
        diags_b = RW.jarl_json([bf])
        diags_a = RW.jarl_json([af])
    finally:
        bf.unlink(missing_ok=True)
        af.unlink(missing_ok=True)
    keys_b = {(d["message"]["name"], d["location"]["row"]) for d in diags_b}
    keys_a = {(d["message"]["name"], d["location"]["row"]) for d in diags_a}
    agree = 0
    for f in ctx["findings"]:
        rule = ALIAS.get(f["rule"], f["rule"])
        if rule is None:
            continue
        if (rule, f["row"] - ctx["top_row"] + 1) in keys_b:
            agree += 1
    checkable = [f for f in ctx["findings"] if ALIAS.get(f["rule"], f["rule"])]
    g["jarl_agreement"] = (agree == len(checkable)) if checkable else True
    g["jarl_agreement_n"] = f"{agree}/{len(checkable)}"
    targeted = {(ALIAS.get(f["rule"], f["rule"]),
                 f["row"] - ctx["top_row"] + 1) for f in checkable}
    g["lint_delta"] = targeted.isdisjoint(keys_a) and (keys_a - keys_b == set())
    g["lint_new"] = sorted(f"{r}@{row}" for r, row in (keys_a - keys_b))[:5]

    return g


def behavior_gate(ctx: dict, authored: str) -> tuple[str, str]:
    """'passed'/'failed'/'skipped(...)' — mine arm only, where callable."""
    if ctx["arm"] != "mine":
        return "skipped(n/a)", ""
    call = ctx["spec"].get("behavior_call")
    if not call:
        return "skipped(not-callable)", ""
    out_b = RW.behavior_probe(ctx["dirty_fn"], call)
    out_a = RW.behavior_probe(authored, call)
    if out_b is None or out_a is None:
        return "skipped(timeout/infra)", ""
    if out_b == out_a:
        return "passed", ""
    if RW.behavior_probe(ctx["dirty_fn"], call) != out_b:
        return "skipped(nondeterministic)", ""
    return "failed", json.dumps([out_b[:120], out_a[:120]])


def map_region(dirty_lines: list[str], auth_lines: list[str],
               r0: int, span: int) -> tuple[int, int]:
    """Authored-line range corresponding to dirty-local rows [r0, r0+span),
    anchored on SequenceMatcher matching blocks: a replaced line's authored
    replacement is the gap between the previous block's end and the next
    block's start, so one finding never bleeds into a neighbor's fix."""
    sm = difflib.SequenceMatcher(a=dirty_lines, b=auth_lines, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]

    def boundary(k: int, left: bool) -> int:
        for i, j, n in blocks:
            if i <= k <= i + n:
                return j + (k - i)
        if left:      # attach gap content to the finding: start at prev end
            prev = [j + n for i, j, n in blocks if i + n <= k]
            return max(prev) if prev else 0
        nxt = [j for i, j, n in blocks if i >= k]
        return min(nxt) if nxt else len(auth_lines)

    js = boundary(r0, left=True)
    je = boundary(r0 + span, left=False)
    return js, max(je, js + 1)


# Was the finding ACTUALLY fixed? (jarl has no counterpart rule for
# paste0_sep/sapply_vapply/bug rules, so lint_delta is blind there — this is
# the targeted semantic check on the row's changed region.)
def fix_applied(f: dict, arm: str, old_text: str, new_text: str) -> bool:
    if arm in ("inject", "buinject"):
        return bool(f.get("fix")) and f["fix"] in new_text \
            and new_text != old_text
    rule = f["rule"]
    if rule == "true_false_symbol":
        want = "TRUE" if f["old"] == "T" else "FALSE"
        return want in new_text and \
            not re.search(r"(?<![\w.])[TF](?![\w.(])", new_text)
    if rule == "paste0_sep":
        return "paste0(" in new_text and not re.search(r"\bpaste\s*\(", new_text)
    if rule == "sapply_vapply":
        fv = re.search(r"(logical|numeric|integer|character)\(1\)$",
                       f.get("new") or "")
        return "vapply(" in new_text and \
            not re.search(r"\bsapply\s*\(", new_text) and \
            (fv is None or fv.group(0) in new_text)
    if rule == "seq_safety":
        return ("seq_along(" in new_text or "seq_len(" in new_text) and \
            not re.search(r"1:\s*(?:length|nrow|ncol|NROW|NCOL)\s*\(", new_text)
    if rule == "class_equals":
        return "inherits(" in new_text
    return new_text != old_text


def build_rows(ctx: dict, authored: str, spliced: bytes, prompt: str,
               extract_kind: str) -> list[dict]:
    """ONE row per finding, self-consistent: region_old is the flagged span
    of the DIRTY function, region_new the corresponding authored span, and
    prefix/suffix come from the AUTHORED (spliced) file so every row's
    prefix+region_new+suffix is exactly the accepted fixed file."""
    spec = ctx["spec"]
    dirty_full = ctx["dirty"].decode("utf-8", "replace").split("\n")
    auth_lines = _rnorm(authored)
    spliced_lines = spliced.decode("utf-8", "replace").split("\n")
    family, _out = FAMILY_OUT[ctx["arm"]]
    rows = []
    for f in ctx["findings"]:
        f_row0, f_erow = f["row"], f["erow"]
        line_delta = f["new"].count("\n") - f["old"].count("\n")
        js, je = map_region(_rnorm(ctx["dirty_fn"]), auth_lines,
                            f_row0 - ctx["top_row"], f_erow - f_row0 + 1)
        old_lines = [dirty_full[r] for r in range(f_row0, f_erow + 1)]
        new_lines = auth_lines[js:je]
        if ctx["arm"] == "mine":
            one = RW.apply_fixes(ctx["src"], [f])
            fixed = one.decode("utf-8", "replace").split("\n")
            corpus_target = "\n".join(
                fixed[r] for r in range(f_row0, f_erow + 1 + line_delta))
        else:
            corpus_target = "\n".join(
                ctx["src"].decode("utf-8", "replace").split("\n")[r]
                for r in range(f_row0, f_erow + 1))
        prefix = spliced_lines[max(0, ctx["head_row"] - 8):
                                ctx["top_row"] + js]
        fn_end_new = ctx["top_row"] + len(auth_lines) - 1
        suffix = spliced_lines[ctx["top_row"] + je:
                               min(len(spliced_lines), fn_end_new + 9)]
        chash = hashlib.sha1(
            f"{family}\x00{spec['base_sample_id']}\x00{f['rule']}\x00"
            f"{f['sb']}".encode()).hexdigest()
        rows.append(dict(
            family=family, transform=f"fix_{f['rule']}", arm=ctx["arm"],
            transform_id=f"{family}/{f['rule']}@1",
            base_sample_id=spec["base_sample_id"],
            derivation=dict(rule_id=f"rewrite/{f['rule']}", rule_version=1,
                            arm=ctx["arm"], site=f["sb"], spec_id=spec["id"]),
            package=spec["package"], path=spec["path"], row=f_row0,
            prefix=prefix or [""], region_old=old_lines, region_new=new_lines,
            suffix=suffix, cursor_idx=0, event_diff="",
            note=(f"rewrite the flagged line(s): {f['rationale']} "
                  f"(fix exactly this finding, touch nothing else)"),
            case="rewrite_author_zai", backend="zai", model="glm-5.3",
            full_prompt=prompt, generated_at=_now(),
            corpus_target=corpus_target,
            model_target="\n".join(new_lines),
            extract_kind=extract_kind,
            content_hash=chash, constraint_spec=spec["id"],
            determinism="D3 author-LLM (gated)"))
    return rows


GATE_ORDER = ("gt_corpus_exact", "loc", "diff_minimal", "parse", "splice",
              "jarl_agreement", "lint_delta", "fix_applied", "row_check")


def gate_summary(gates: dict, behavior: str) -> str:
    fails = [k for k in GATE_ORDER if gates.get(k) is False]
    if behavior == "failed":
        fails.append("behavior")
    if not fails:
        return ""
    detail = ""
    if fails[0] == "diff_minimal" and gates.get("diff_outside"):
        detail = f" changed lines outside covers {gates['diff_outside']}"
    if fails[0] == "lint_delta" and gates.get("lint_new"):
        detail = f" new lint {gates['lint_new']}"
    return f"{','.join(fails)};" + detail


# ---------------------------------------------------------------------------
# PHASE 2: the author loop (resume-safe, <=4 in flight, 429 breaker)
# ---------------------------------------------------------------------------

class Breaker:
    """Global 429 circuit: surfacing rate errors (after the backend's own 3
    retries) escalate a shared pause; sustained starvation stops the run."""

    def __init__(self, max_rounds: int = 4):
        self.lock = threading.Lock()
        self.consecutive = 0
        self.rounds = 0
        self.max_rounds = max_rounds
        self.pause_until = 0.0
        self.stop = threading.Event()
        self.rate_errors = 0

    def wait_turn(self):
        while True:
            if self.stop.is_set():
                return False
            with self.lock:
                until = self.pause_until
            if time.time() >= until:
                return True
            time.sleep(min(30.0, until - time.time() + 0.5))

    def report(self, ok: bool, rate_error: bool):
        with self.lock:
            if rate_error:
                self.rate_errors += 1
                self.consecutive += 1
                if self.consecutive >= 4:
                    self.rounds += 1
                    if self.rounds > self.max_rounds:
                        self.stop.set()
                    else:
                        pause = min(600.0, 60.0 * (2 ** (self.rounds - 1)))
                        self.pause_until = time.time() + pause
                        print(f"  [breaker] rate errors x{self.consecutive}: "
                              f"pausing {pause:.0f}s (round {self.rounds}/"
                              f"{self.max_rounds})", flush=True)
                    self.consecutive = 0
            elif ok:
                self.consecutive = 0


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["key"]] = rec
            except (ValueError, KeyError):
                pass
    return done


def load_hashes(path: Path) -> set[str]:
    hs: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                h = json.loads(line).get("content_hash")
                if h:
                    hs.add(h)
            except ValueError:
                pass
    return hs


def cmd_author(args) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in FAMILY_OUT:
            sys.exit(f"unknown arm {a!r}; known: {sorted(FAMILY_OUT)}")
    if not POOL_PATH.exists():
        sys.exit(f"no spec pool at {POOL_PATH}; run `mine` first")
    specs = []
    for line in POOL_PATH.read_text().splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue
        if s.get("arm") in arms:
            specs.append(s)
    specs.sort(key=lambda s: (s["arm"] != "mine", s["arm"] != "inject", s["id"]))

    outs = {}
    for arm in arms:
        family, out = FAMILY_OUT[arm]
        done_path = Path(str(out) + ".done.jsonl")
        outs[out] = dict(family=family, out=out, done_path=done_path,
                         done=load_done(done_path),
                         hashes=load_hashes(out), rows=0,
                         stats=dict(attempted=0, accepted=0, retried=0,
                                    backend_error=0, layer1_json=0,
                                    extract_rawslice=0, gate_rejects=Counter(),
                                    behavior_failed=0, dups=0,
                                    per_arm=Counter()))
    pending = [s for s in specs
               if s["id"] not in outs[FAMILY_OUT[s["arm"]][1]]["done"]]
    print(f"[author] arms={arms} pool={len(specs)} pending={len(pending)} "
          f"(done keys: " +
          ", ".join(f"{Path(k).name}={len(v['done'])}"
                    for k, v in outs.items()) + ")")
    if args.max > 0:
        pending = pending[:args.max]

    backend = ZaiAuthorBackend()
    breaker = Breaker()
    stats = dict(started=_now(), arms=arms, backend="zai", model=backend.model,
                 workers=args.workers, pool=len(specs), pending=len(pending))
    site_lock, site_ctr = threading.Lock(), [0]
    stats_lock = threading.Lock()

    def bump(entry: dict, key: str, n: int = 1):
        with stats_lock:
            entry["stats"][key] += n

    t0 = time.time()
    last_log = 0.0
    stop_hard = threading.Event()          # backend-failure abort
    be_fail_lock, be_fail = threading.Lock(), [0]

    def process(spec: dict) -> dict:
        """One spec: up to 2 generations; returns the terminal outcome."""
        ctx = build_ctx(spec)
        with site_lock:
            site_ctr[0] += 1
            site = site_ctr[0]
        outs_entry = outs[FAMILY_OUT[spec["arm"]][1]]
        bump(outs_entry, "attempted")
        feedback = None
        last_raw_kind = "?"
        first_prompt = ""
        for attempt in (1, 2):
            if not breaker.wait_turn() or stop_hard.is_set():
                return dict(kind="aborted", spec=spec)
            prompt = build_prompt(ctx, feedback)
            if attempt == 1:
                first_prompt = prompt
            try:
                raw = backend.complete(prompt)
                breaker.report(ok=True, rate_error=False)
            except BackendError as e:
                breaker.report(ok=False, rate_error=(e.kind == "rate"))
                bump(outs_entry, "backend_error")
                with be_fail_lock:
                    be_fail[0] += 1
                    hard = be_fail[0] >= 15
                if hard:
                    stop_hard.set()
                return dict(kind="backend_error", spec=spec, err=str(e)[:160])
            with be_fail_lock:
                be_fail[0] = 0
            fn_text, kind = extract_fn_text(raw)
            last_raw_kind = kind
            if fn_text is None:
                bump(outs_entry, "layer1_json")
                feedback = "the response was not parseable (return ONLY the JSON object)."
                continue
            fn_text = "\n".join(_norm_trim(fn_text))
            if kind == "rawslice":
                bump(outs_entry, "extract_rawslice")
            gates = run_gates(ctx, fn_text, site)
            beh, beh_detail = behavior_gate(ctx, fn_text)
            gates["behavior"] = beh
            rows = build_rows(ctx, fn_text, gates["_spliced"],
                              first_prompt, kind)
            gates["fix_applied"] = all(
                fix_applied(f, ctx["arm"], "\n".join(r["region_old"]),
                            "\n".join(r["region_new"]))
                for f, r in zip(ctx["findings"], rows))
            gates["row_check"] = all(RW.check_rewrite_row(r)[0] for r in rows)
            summary = gate_summary(gates, beh)
            if not summary:
                gate_snap = {k: v for k, v in gates.items()
                             if not k.startswith("_") and k != "loc_delta"}
                for r in rows:
                    r["gates"] = gate_snap
                return dict(kind="accepted", spec=spec, ctx=ctx, rows=rows,
                            gates=gates, attempts=attempt)
            feedback = summary
            if attempt == 2:
                for k in GATE_ORDER:
                    if gates.get(k) is False:
                        with stats_lock:
                            outs_entry["stats"]["gate_rejects"][k] += 1
                if beh == "failed":
                    bump(outs_entry, "behavior_failed")
                return dict(kind="rejected", spec=spec, gates=gates,
                            reason=summary, attempts=2, kind2=last_raw_kind)
            bump(outs_entry, "retried")
        return dict(kind="rejected", spec=spec, gates={}, reason="unparseable",
                    attempts=2, kind2=last_raw_kind)

    def flush_stats(final: bool = False):
        for out, entry in outs.items():
            if not entry["stats"]["attempted"] and not final:
                continue
            rep = dict(stats)
            rep.update(dict(ts=_now(), out=str(out), final=final,
                            rows_total=entry["rows"],
                            elapsed_s=round(time.time() - t0, 1),
                            backend_stats=backend.stats_summary(),
                            breaker=dict(rate_errors=breaker.rate_errors,
                                         rounds=breaker.rounds,
                                         stopped=breaker.stop.is_set()),
                            counts={k: (dict(v) if isinstance(v, Counter) else v)
                                    for k, v in entry["stats"].items()},
                            done_keys=len(entry["done"])))
            _write_json(Path(str(out) + ".stats.json"), rep)

    ex = ThreadPoolExecutor(max_workers=args.workers)
    outstanding: set[Future] = set()
    it = iter(pending)
    n_done = 0
    accepted_total = 0
    try:
        while True:
            if time.time() - t0 > args.time_budget or breaker.stop.is_set() \
                    or stop_hard.is_set():
                break
            while len(outstanding) < args.workers * 2:
                try:
                    s = next(it)
                except StopIteration:
                    break
                outstanding.add(ex.submit(process, s))
            if not outstanding:
                break
            done_set, _ = wait(outstanding, timeout=30,
                               return_when=FIRST_EXCEPTION)
            for fut in done_set:
                outstanding.discard(fut)
                try:
                    res = fut.result()
                except Exception as e:                       # driver bug guard
                    print(f"  [worker-exception] {e!r}", flush=True)
                    continue
                n_done += 1
                spec, arm = res["spec"], res["spec"]["arm"]
                entry = outs[FAMILY_OUT[arm][1]]
                if res["kind"] == "accepted":
                    new_rows = []
                    for row in res["rows"]:
                        if row["content_hash"] in entry["hashes"]:
                            entry["stats"]["dups"] += 1
                            continue
                        entry["hashes"].add(row["content_hash"])
                        new_rows.append(row)
                    for row in new_rows:
                        _append_line(entry["out"], row)
                        entry["rows"] += 1
                        accepted_total += 1
                        entry["stats"]["accepted"] += 1
                        entry["stats"]["per_arm"][arm] += 1
                    rec = dict(key=spec["id"], ok=True,
                               rows=len(new_rows), attempts=res["attempts"],
                               rules=[f["rule"] for f in spec["findings"]],
                               ts=_now())
                elif res["kind"] in ("rejected",):
                    rec = dict(key=spec["id"], ok=False,
                               reason=res["reason"][:200],
                               gates={k: v for k, v in res["gates"].items()
                                      if v is False},
                               rules=[f["rule"] for f in spec["findings"]],
                               ts=_now())
                else:                       # backend_error / aborted: retryable
                    rec = None
                if rec is not None:
                    _append_line(entry["done_path"], rec)
                    entry["done"][spec["id"]] = rec
            if time.time() - last_log > 60:
                last_log = time.time()
                acc = sum(e["stats"]["accepted"] for e in outs.values())
                print(f"  [progress] done={n_done}/{len(pending)} "
                      f"accepted={acc} "
                      f"rejects=" +
                      "/".join(f"{k}:{v}" for e in outs.values()
                               for k, v in e["stats"]["gate_rejects"].items()
                               if v) +
                      f" backend_ok={backend.stats_summary()['ok']} "
                      f"429={backend.stats_summary()['err_429']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                flush_stats()
    finally:
        for fut in outstanding:
            fut.cancel()
        ex.shutdown(wait=True)
        flush_stats(final=True)
    acc = sum(e["stats"]["accepted"] for e in outs.values())
    print(f"[author] FINISHED done={n_done}/{len(pending)} accepted={acc} "
          f"rows in {time.time()-t0:.0f}s; "
          f"stopped={'breaker' if breaker.stop.is_set() else 'no'}"
          f"{' hard' if stop_hard.is_set() else ''}")
    return 0


def cmd_stats(_args) -> int:
    for out in {DATASETS / "rewrite_lint_fix.jsonl",
                DATASETS / "fix_issue_inject.jsonl"}:
        sp = Path(str(out) + ".stats.json")
        print(f"== {out.name}")
        if sp.exists():
            print(sp.read_text())
        else:
            print("  (no stats yet)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 rewrite_author_zai.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mine")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--tidy-packages", type=int, default=250)
    p.add_argument("--random-packages", type=int, default=2200)
    p.add_argument("--mine-specs", type=int, default=1800)
    p.add_argument("--inject-specs", type=int, default=2400)
    p.add_argument("--buinject-specs", type=int, default=3000)
    p.add_argument("--bug-per-fn", type=int, default=2)
    p.add_argument("--time-budget", type=float, default=1800)
    p.add_argument("--fresh", action="store_true")
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("author")
    p.add_argument("--arms", default="mine,inject",
                   help="mine,inject | buinject | mine,inject,buinject")
    p.add_argument("--max", type=int, default=0, help="0 = all pending")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--time-budget", type=float, default=21600)
    p.set_defaults(fn=cmd_author)

    p = sub.add_parser("stats")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
