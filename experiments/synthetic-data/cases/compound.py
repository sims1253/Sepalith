#!/usr/bin/env python3
"""compound.py — compounding-samples prototype (base sample -> scenario matrix).

ONE base sample (one real corpus function, /mnt/h/sepalith/normalized, highest
version per package) yields MANY next-edit-suggestion scenarios via
DETERMINISTIC transformations. Zero LLM calls, zero quota, no GPU: every
ground truth is either the verbatim corpus text or an exact-by-construction
rewrite, and every row is checked with the EXISTING cases/ validators
(validators.check_row row_checks, validators.REGISTRY layer-3 gates,
scenarios.validate_example for the event-driven rows).

Transform registry (each tagged: family fed, determinism class, reuse):
  mbe_arg_edit / mbe_na_rm_insert / mbe_rename_once / mbe_insert_line
      -> mid_body_edit (cases.corpus._arg_edit_cands/_na_rm_cands/
         _rename_cands/_insert_cands + _mid_body_item; rc mid_body_edit_site)
  rename_propagation / na_rm_propagation
      -> event-driven scenario rows (scenarios.extract_rename /
         extract_na_rm filtered to the base function; scenarios.validate_example)
  removed_block -> removed_block_comment geometry with a DETERMINISTIC site
      comment (rc removed_block_site)
  comment_to_code -> an interior comment above an interior block: comment
      shown, block cut (rc ends_with_comment_line — the comment_to_code shape)
  comment_drafting -> the SAME site reversed: block shown, the AUTHOR's
      comment is the target (layer-3 gate r_comment; GT = corpus comment)
  astfim_partial -> retyped-partial of the removed block (rc astfim_partial_
      site, PSM cursor zone carries the partial)
  namespace_qualify -> per-function pkg:: reverse-strip (rc
      qualified_call_cursor)
  examples_completion -> roxygen @examples stanza of the base function: cut
      after k typed example lines, GT = the author's next example lines
      (proposals_v1 roxygen_examples direction; layer-3 r_fragment on the
      de-prefixed target)
  loop_to_apply -> NEW: for (i in seq_along(x)) { res[[i]] <- f(x[[i]]) }
      -> res <- lapply(x, function(i) ...) and the purrr::map variant
      (validator: splice re-parse + loop-var hygiene; exact by construction)
  trycatch_wrap -> NEW: I/O-doing function body wrapped in tryCatch(...)
      (validator: splice re-parse + the existing handler_clauses gate)

Usage (from experiments/synthetic-data, system python — NOT .venv-sft):
  python3 cases/compound.py --base-samples 20 --probe-packages 200 \
      --out results/compound_proto/scenarios.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # synthetic-data/ for scenarios, corpus

import scenarios as S                       # noqa: E402
from scenarios import node_text            # noqa: E402
import cases.corpus as C                   # noqa: E402
import cases.validators as V               # noqa: E402

OUT_DEFAULT = HERE.parent / "results" / "compound_proto" / "scenarios.jsonl"
STATS_DEFAULT = HERE.parent / "results" / "compound_proto" / "stats.json"
GENERATED_AT = "2026-08-20T00:00:00"       # fixed: fully deterministic output

TRANSFORMS: dict = {}


def register(name: str, family: str, determinism: str, reuses: str):
    def deco(fn):
        TRANSFORMS[name] = dict(fn=fn, family=family, determinism=determinism,
                                reuses=reuses)
        return fn
    return deco


# ---------------------------------------------------------------------------
# base samples
# ---------------------------------------------------------------------------

class BaseSample:
    """One function_definition in one Bundle, with its geometry pinned."""

    __slots__ = ("b", "fn", "body", "head_row", "r0", "r1", "nbody", "top_row",
                 "site_id")

    def __init__(self, b, fn, site_id: int):
        self.b, self.fn, self.site_id = b, fn, site_id
        geom = C._fn_body(b, fn)
        if geom is None:
            raise ValueError("no braced body")
        self.body, self.head_row, self.r0, self.r1, nb = geom
        self.nbody = len(nb)
        parent = fn.parent
        self.top_row = min(fn.start_point[0],
                           parent.children[0].start_point[0]) \
            if (parent is not None and parent.type == "binary_operator"
                and parent.children) else fn.start_point[0]

    def lines(self, r0: int, r1: int) -> list[str]:
        return [self.b.line_str(r) for r in range(r0, r1)]


def _plain_comment_rows(bs: BaseSample) -> list[int]:
    out = []
    for r in range(bs.r0 + 1, bs.r1):
        s = bs.b.line_str(r).strip()
        if s.startswith("#") and not s.startswith("#'") and len(s) > 2:
            out.append(r)
    return out


def collect_base_samples(rng: random.Random, want: int, params: dict):
    """Scan tidy + random packages (highest version, cached versions map) for
    functions with 6..40 non-blank body lines and >= 1 plain interior comment;
    then (pass 2) top up with @examples-bearing comment functions so the
    examples family fires on real samples. Returns (samples, funnel,
    scanned_files) — the funnel records why functions were skipped (for
    honest yield reporting)."""
    min_body = int(params.get("min_body_lines", 6))
    max_body = int(params.get("max_body_lines", 40))
    n_tidy = int(params.get("tidy_packages", 40))
    n_rand = int(params.get("random_packages", 40))
    time_budget = float(params.get("time_budget_s", 300))
    ex_quota = int(params.get("examples_quota", 5))

    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in S.list_packages() if p not in set(tidy)]
    rng.shuffle(rest)
    pool = tidy[:n_tidy] + rest[:n_rand]
    rng.shuffle(pool)

    funnel = dict(functions_seen=0, size_ok=0, comment_ok=0,
                  roxygen_examples=0, files=0, packages=set())
    samples: list[BaseSample] = []
    ex_seen = 0
    pkg_counts: dict[str, int] = {}
    per_pkg_cap = int(params.get("per_package_samples", 3))

    def consider(b, fn) -> tuple[bool, bool]:
        """(eligible, has_examples) — size + plain-comment criteria."""
        funnel["functions_seen"] += 1
        geom = C._fn_body(b, fn)
        if geom is None:
            return False, False
        _body, _head, _r0, _r1, nb = geom
        if not min_body <= len(nb) <= max_body:
            return False, False
        funnel["size_ok"] += 1
        try:
            bs = BaseSample(b, fn, 0)
        except ValueError:
            return False, False
        if not _plain_comment_rows(bs):
            return False, False      # comment families must fire on every sample
        funnel["comment_ok"] += 1
        has_ex = bool(_examples_block(bs))
        if has_ex:
            funnel["roxygen_examples"] += 1
        return True, has_ex

    t0 = time.time()
    site_id = 0

    def take(b, fn) -> None:
        nonlocal site_id
        samples.append(BaseSample(b, fn, site_id))
        site_id += 1

    for b in C.iter_bundles_highest(pool, rng):
        funnel["files"] += 1
        funnel["packages"].add(b.package)
        if time.time() - t0 > time_budget or len(samples) >= want:
            break
        if pkg_counts.get(b.package, 0) >= per_pkg_cap:
            continue
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            if len(samples) >= want:
                break
            eligible, has_ex = consider(b, fn)
            if not eligible:
                continue
            if has_ex:
                if ex_seen >= ex_quota:
                    continue          # quota met: prefer plain functions
                ex_seen += 1
            take(b, fn)
            pkg_counts[b.package] = pkg_counts.get(b.package, 0) + 1

    # pass 2: if the quota of @examples-bearing samples was not met, scan the
    # remaining pool for comment+examples functions only
    if ex_seen < ex_quota and time.time() - t0 <= time_budget:
        for b in C.iter_bundles_highest(pool, rng):
            if ex_seen >= ex_quota or time.time() - t0 > time_budget:
                break
            if pkg_counts.get(b.package, 0) >= per_pkg_cap:
                continue
            for fn in (n for n in V._walk(b.tree.root_node)
                       if n.type == "function_definition"):
                if ex_seen >= ex_quota:
                    break
                eligible, has_ex = consider(b, fn)
                if not eligible or not has_ex:
                    continue
                take(b, fn)
                ex_seen += 1
                pkg_counts[b.package] = pkg_counts.get(b.package, 0) + 1
    funnel["packages"] = len(funnel["packages"])
    for i, bs in enumerate(samples):      # stable ids after collection
        bs.site_id = i
    return samples, funnel


# ---------------------------------------------------------------------------
# row assembly + validation
# ---------------------------------------------------------------------------

def item_to_row(family: str, transform: str, item: dict, carry=()) -> dict:
    """case-item (prefix/block/suffix) -> suffix-convention scenario row with
    the mock-draw convention: model_target == corpus_target (tier-1 exact by
    construction; the model draw never gates a corpus-side family)."""
    row = dict(family=family, transform=transform,
               package=item.get("package", "?"), path=item.get("path", "?"),
               prefix=list(item.get("prefix") or []),
               region_old=[""], region_new=[str(l) for l in item.get("block")],
               cursor_idx=0, event_diff="", note=item.get("note", ""),
               suffix=list(item.get("suffix") or []),
               corpus_target=item.get("corpus_target"))
    for k in carry:
        if item.get(k) is not None:
            row[k] = item[k]
    if row["corpus_target"] is not None:
        row["model_target"] = row["corpus_target"]
    row.update(case="compound_proto", backend="deterministic",
               model="static-transform", full_prompt="",
               generated_at=GENERATED_AT,
               base_sample=f"bs:{item.get('_site', -1)}")
    return row


def _validate_case_row(row: dict, rc: dict | None, validator: tuple | None):
    """Base row-structure gate + optional registered row_check + optional
    layer-3 validator (name, params, target-text). Returns (ok, reason)."""
    ok, reason = V.check_row(row, rc)
    if not ok:
        return False, f"rowcheck: {reason}"
    if validator is not None:
        name, params, target = validator
        ok, reason = V.REGISTRY[name](target, params)
        if not ok:
            return False, f"layer3[{name}]: {reason}"
    return True, ""


def _strict_mid(bs: BaseSample, row: int) -> bool:
    b = bs.b
    if not bs.r0 < row < bs.r1:
        return False
    return (any(b.line_str(r).strip() for r in range(bs.r0 + 1, row))
            and any(b.line_str(r).strip() for r in range(row + 1, bs.r1)))


def _splice_reparse(bs: BaseSample, start_byte: int, end_byte: int,
                    replacement: str) -> bool:
    """Replace [start_byte, end_byte) inside the WHOLE defining statement of
    the base function with `replacement` and re-parse it as a fragment — the
    strongest deterministic check that the rewrite keeps the function valid."""
    src = bs.b.src
    parent = bs.fn.parent
    base_start = parent.start_byte if (parent is not None
                                       and parent.type == "binary_operator") \
        else bs.fn.start_byte
    text = src[base_start:bs.fn.end_byte]
    new = text[:start_byte - base_start] + replacement.encode() \
        + text[end_byte - base_start:]
    return V.fragment_clean(new.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# mid_body_edit kinds (full reuse of the corpus extractors)
# ---------------------------------------------------------------------------

def _mbe_transform(cand_fn, kind: str):
    def run(bs: BaseSample, rng: random.Random, params: dict):
        out = []
        cap = int(params.get("cap", 2))
        if kind == "na_rm_insert" and C._STAT_SHADOW_RE.search(bs.b.src):
            return out, ["stat fn shadowed in file"]
        cands = cand_fn(bs)
        rejects = []
        for _k, row, payload in cands[:cap * 3]:
            if len(out) >= cap:
                break
            if not _strict_mid(bs, row):
                rejects.append(f"{kind}: site not strictly mid-body")
                continue
            item = C._mid_body_item(bs.b, kind, row, payload,
                                    int(params.get("window_lines", 10)),
                                    bs.head_row, bs.r1)
            if item is None:
                rejects.append(f"{kind}: _mid_body_item rejected the site")
                continue
            item["_site"] = bs.site_id
            rowd = item_to_row(
                "mid_body_edit", f"mbe_{kind}", item,
                carry=("mutation_kind", "corpus_line", "fn_head", "old_tok",
                       "new_tok", "stat_fn", "insert_base"))
            ok, reason = _validate_case_row(
                rowd, {"name": "mid_body_edit_site",
                       "params": {"max_target_lines": 1}},
                ("mid_body_edit_line", {"max_lines": 3, "max_len": 220},
                 "\n".join(rowd["region_new"])))
            if not ok:
                rejects.append(reason)
                continue
            out.append(rowd)
        return out, rejects
    return run


# ---------------------------------------------------------------------------
# event-driven propagation rows (scenarios.py families, filtered to the fn)
# ---------------------------------------------------------------------------

def _scenario_rows(bs: BaseSample, examples: list[dict]) -> list[dict]:
    """Keep only the examples whose region line is inside the base function
    (matched by exact line text + the preceding prefix line)."""
    span = {bs.b.line_str(r): r for r in range(bs.r0, bs.r1)}
    out = []
    for ex in examples:
        if ex.get("path") != bs.b.rel:
            continue
        r = span.get((ex.get("region_old") or ["?"])[0])
        if r is None or not bs.r0 < r < bs.r1:
            continue
        if ex["prefix"] and ex["prefix"][-1] != bs.b.line_str(r - 1):
            continue
        out.append(ex)
    return out


# ---------------------------------------------------------------------------
# comment sites -> comment_to_code + comment_drafting (one site, two rows)
# ---------------------------------------------------------------------------

def _comment_block(bs: BaseSample, crow: int, max_lines: int = 6):
    """The statement block immediately below an interior comment (through the
    first blank/comment/brace line)."""
    block = []
    j = crow + 1
    while j < bs.r1 and len(block) < max_lines:
        s = bs.b.line_str(j).strip()
        if not s or s.startswith("#") or s == "}":
            break
        block.append(bs.b.line_str(j))
        j += 1
    return block


def _comment_pair(bs: BaseSample, rng: random.Random, params: dict):
    out = []
    rejects = []
    window = int(params.get("window_lines", 10))
    cap = int(params.get("cap", 2))
    for crow in _plain_comment_rows(bs)[:cap * 2]:
        if len(out) >= cap * 2:
            break
        block = _comment_block(bs, crow)
        if not block or not V.fragment_clean("\n".join(block)):
            rejects.append("comment site: block below does not parse clean")
            continue
        n_stmt = len([c for c in V.parse_fragment("\n".join(block)).root_node.children
                      if c.is_named])
        if not 1 <= n_stmt <= 4:
            rejects.append("comment site: block statements out of 1..4")
            continue
        prefix = bs.lines(max(0, bs.head_row - window), crow + 1)
        suffix = bs.lines(crow + len(block) + 1,
                          min(bs.b.nlines(), bs.r1 + 1 + window))
        if not any(l.strip() == "}" for l in suffix):
            rejects.append("comment site: closing brace not in suffix")
            continue
        # (a) comment_to_code: comment shown, block cut (GT = corpus block)
        c2c = dict(package=bs.b.package, path=bs.b.rel, row=crow,
                   prefix=prefix, block=block, suffix=suffix,
                   corpus_target="\n".join(block), _site=bs.site_id,
                   note="interior comment -> the block it describes "
                        "(comment_to_code geometry)")
        row_a = item_to_row("comment_to_code", "comment_to_code", c2c)
        ok, reason = _validate_case_row(row_a, {"name": "ends_with_comment_line",
                                                "params": {}}, None)
        if ok:
            out.append(row_a)
        else:
            rejects.append(reason)
        # (b) comment_drafting: block shown, the AUTHOR's comment is the GT
        comment_line = bs.b.line_str(crow)
        ok_c, reason_c = V.REGISTRY["r_comment_gate"](
            comment_line.strip().lstrip("#").strip(), {"max_len": 90})
        if not ok_c:
            rejects.append(f"comment_drafting: r_comment gate: {reason_c}")
            continue
        d_prefix = bs.lines(max(0, bs.head_row - window), crow)
        if not d_prefix or not d_prefix[-1].strip():
            rejects.append("comment_drafting: cursor line blank")
            continue
        d_item = dict(package=bs.b.package, path=bs.b.rel, row=crow,
                      prefix=d_prefix, block=[comment_line], suffix=suffix,
                      corpus_target=comment_line, _site=bs.site_id,
                      note="draft the author's comment for the block below "
                           "(corpus-attested GT)")
        row_b = item_to_row("comment_drafting", "comment_drafting", d_item)
        ok, reason = _validate_case_row(row_b, None,
                                        ("r_comment_gate", {"max_len": 90},
                                         comment_line.strip()))
        if ok:
            out.append(row_b)
        else:
            rejects.append(reason)
    return out, rejects


@register("comment_pair", family="comment_to_code+comment_drafting",
          determinism="pure-static (one corpus comment site -> both directions)",
          reuses="rc ends_with_comment_line; layer-3 r_comment (comment_drafting)")
def _t_comment_pair(bs: BaseSample, rng: random.Random, params: dict):
    return _comment_pair(bs, rng, params)


# ---------------------------------------------------------------------------
# removed_block + astfim_partial (one removed-block site, two rows)
# ---------------------------------------------------------------------------

def _removed_block_site(bs: BaseSample, rng: random.Random, params: dict):
    """extract_removed_blocks geometry (statement boundaries, 3..8 lines,
    strictly interior) bounded to the base function, with a DETERMINISTIC
    site comment (the author-LLM slot of the real family)."""
    block_min = int(params.get("block_lines_min", 3))
    block_max = int(params.get("block_lines_max", 8))
    window = int(params.get("window_lines", 10))
    max_len = int(params.get("max_block_chars", 700))
    stmts = C._body_statements(bs.body)
    cands = []
    for i in range(len(stmts)):
        first = stmts[i].start_point[0]
        for j in range(i, len(stmts) - 1):        # never the whole body
            last = stmts[j].end_point[0]
            n_lines = last - first + 1
            if n_lines > block_max:
                break
            if n_lines >= block_min and i > 0:
                cands.append((first, last))
    rng.shuffle(cands)
    for first, last in cands:
        block_lines = bs.lines(first, last + 1)
        if block_lines[0].lstrip().startswith("#"):
            continue
        block_text = "\n".join(block_lines)
        if not block_text.strip() or len(block_text) > max_len:
            continue
        if not C._block_clean(block_text):
            continue
        if not (any(bs.b.line_str(r).strip() for r in range(bs.r0 + 1, first))
                and any(bs.b.line_str(r).strip()
                        for r in range(last + 1, bs.r1))):
            continue          # strictly interior, never the last statement
        prefix = bs.lines(max(0, bs.head_row - window), first)
        if not prefix or not prefix[-1].strip():
            continue
        suffix = bs.lines(last + 1, min(bs.b.nlines(), bs.r1 + 1 + window))
        if not any(l.strip() for l in suffix):
            continue
        indent = block_lines[0][:len(block_lines[0])
                                - len(block_lines[0].lstrip())]
        return first, last, block_lines, \
            prefix + [f"{indent}# TODO restore this step"], suffix, indent
    return None


@register("removed_block", family="removed_block_comment",
          determinism="static+validator (site comment = the author-LLM slot)",
          reuses="cases.corpus._body_statements/_block_clean; rc removed_block_site")
def t_removed_block(bs: BaseSample, rng: random.Random, params: dict):
    site = _removed_block_site(bs, rng, params)
    if site is None:
        return [], ["no qualifying interior statement run (3..8 lines)"]
    first, last, block_lines, prefix, suffix, indent = site
    item = dict(package=bs.b.package, path=bs.b.rel, row=first, prefix=prefix,
                block=block_lines, suffix=suffix, _site=bs.site_id,
                fn_head=bs.b.line_str(bs.head_row),
                corpus_target="\n".join(block_lines),
                note=f"removed interior {len(block_lines)}-line sub-block; "
                     f"deterministic site comment marks it")
    row = item_to_row("removed_block_comment", "removed_block", item,
                      carry=("fn_head",))
    ok, reason = _validate_case_row(
        row, {"name": "removed_block_site",
              "params": {"block_lines_min": 3, "block_lines_max": 8}}, None)
    return ([row], []) if ok else ([], [reason])


@register("astfim_partial", family="astfim_partial", determinism="pure-static",
          reuses="cases.corpus.derive_astfim_partial geometry; rc astfim_partial_site")
def t_astfim_partial(bs: BaseSample, rng: random.Random, params: dict):
    site = _removed_block_site(bs, rng, params)
    if site is None:
        return [], ["no removed-block site to retype from"]
    first, last, block_lines, _prefix, suffix, _indent = site
    max_k = int(params.get("max_partial_lines", 3))
    if len(block_lines) < 2:
        return [], ["block too short to split"]
    k = rng.randint(1, min(max_k, len(block_lines) - 1))
    partial, remaining = block_lines[:k], block_lines[k:]
    if not any(l.strip() for l in remaining):
        return [], ["no remaining lines"]
    above = bs.lines(max(0, bs.head_row - 6), first)
    below = suffix
    psm = ("<|context|>" + f"{bs.b.package}/{bs.b.rel}\n"
           + "\n".join(above + partial) + "\n<|history|>\n\n<|cursor|>"
           + "\n".join(partial) + "<|suffix|>\n" + "\n".join(below)
           + "\n<|end|>\n")
    item = dict(package=bs.b.package, path=bs.b.rel, row=first,
                prefix=above + partial, block=remaining, suffix=below,
                _site=bs.site_id, k_partial=k, partial_lines=partial,
                psm_prompt=psm, corpus_target="\n".join(remaining),
                note=f"retyped-partial: user removed the block and retyped "
                     f"its first {k} line(s); finish the remaining "
                     f"{len(remaining)}")
    row = item_to_row("astfim_partial", "astfim_partial", item,
                      carry=("k_partial", "partial_lines", "psm_prompt"))
    ok, reason = _validate_case_row(
        row, {"name": "astfim_partial_site",
              "params": {"max_partial_lines": max_k}}, None)
    return ([row], []) if ok else ([], [reason])


# ---------------------------------------------------------------------------
# namespace_qualify (per-function reverse-strip)
# ---------------------------------------------------------------------------

@register("namespace_qualify", family="namespace_qualify_propagation",
          determinism="pure-static",
          reuses="cases.corpus._NS_RE/_QUAL_HEAD_RE/_strip_ns/_mk_item; rc qualified_call_cursor")
def t_namespace(bs: BaseSample, rng: random.Random, params: dict):
    b, out, rejects = bs.b, [], []
    window = int(params.get("window_lines", 8))
    occ: dict[str, list] = {}
    for r in range(bs.top_row, bs.r1 + 1):
        line = b.line_str(r)
        if not line or line.lstrip().startswith("#"):
            continue
        for m in C._NS_RE.finditer(line):
            occ.setdefault(m.group(1), []).append((r, m.start()))
    for p, positions in sorted(occ.items()):
        if len(positions) < 2:
            continue
        (ev_row, _evc), (row, col) = positions[0], positions[1]
        line = b.line_str(row)
        if any(pr == row and pc < col for pr, pc in positions):
            rejects.append("namespace: earlier same-line occurrence")
            continue
        target_text = line[col:].rstrip()
        if not C._QUAL_HEAD_RE.match(target_text) or len(target_text) > 240:
            rejects.append("namespace: target not a qualified call head")
            continue
        prefix = [(b.line_str(r) if r == ev_row else C._strip_ns(b.line_str(r), p))
                  for r in range(max(0, bs.top_row - window), row)]
        prefix.append(line[:col])
        suffix = [C._strip_ns(b.line_str(r), p)
                  for r in range(row + 1, min(b.nlines(), bs.r1 + 1 + window))]
        item = C._mk_item(b.package, b.rel, row, prefix, [target_text], suffix,
                          qualify_package=p, corpus_line=line, _site=bs.site_id,
                          note=f"re-qualify with {p}:: (event: earlier "
                               f"occurrence restored above)")
        rowd = item_to_row("namespace_qualify_propagation", "namespace_qualify",
                           item, carry=("qualify_package", "corpus_line"))
        ok, reason = _validate_case_row(
            rowd, {"name": "qualified_call_cursor", "params": {}}, None)
        if ok:
            out.append(rowd)
        else:
            rejects.append(reason)
        break                     # one package per base sample
    if not out and not rejects:
        rejects.append("namespace: no pkg:: used >= 2x inside the function")
    return out, rejects


# ---------------------------------------------------------------------------
# roxygen @examples -> examples_completion (deterministic usage-code source)
# ---------------------------------------------------------------------------

ROXY_RE = re.compile(r"^\s*#'(?:\s|$)")
ROXY_TAG_RE = re.compile(r"^\s*#'\s*@([A-Za-z]+)")
DONT_RE = re.compile(r"\\dont(run|show|test)")


def _examples_block(bs: BaseSample):
    """(example_code_lines, roxy_start_row, ex_tag_row) or None. The #'
    prefix + @tag grammar; \\dont* blocks skipped (not deterministic)."""
    b = bs.b
    r, block = bs.top_row - 1, []
    while r >= 0 and ROXY_RE.match(b.line_str(r)):
        block.append(r)
        r -= 1
    if not block:
        return None
    block.reverse()
    for i, rr in enumerate(block):
        m = ROXY_TAG_RE.match(b.line_str(rr))
        if not m or m.group(1) != "examples":
            continue
        code = []
        for r2 in block[i + 1:]:
            line = b.line_str(r2)
            if ROXY_TAG_RE.match(line):
                break
            body = re.sub(r"^\s*#'\s?", "", line)
            if body.strip():
                code.append((r2, body))
        if not code:
            return None
        text_lines = [t for _r, t in code]
        if DONT_RE.search("\n".join(text_lines)):
            return None
        return text_lines, block[0], rr
    return None


@register("examples_completion", family="roxygen_examples",
          determinism="pure-static (extraction) + r_fragment floor",
          reuses="proposals_v1 roxygen_examples corpus direction; validators.r_fragment")
def t_examples(bs: BaseSample, rng: random.Random, params: dict):
    got = _examples_block(bs)
    if got is None:
        return [], ["no clean @examples stanza above the function"]
    code_lines, roxy_start, ex_tag = got
    if len(code_lines) < 3:
        return [], [f"@examples too short ({len(code_lines)} code lines)"]
    b = bs.b
    k = rng.randint(1, max(1, len(code_lines) - 2))
    m_target = rng.randint(1, min(3, len(code_lines) - k))
    # the code lines directly follow the @examples tag line, one per roxygen
    # line (blank roxygen lines were dropped by the block walk — walk the
    # physical rows back off the tag line to keep prefix/target contiguous)
    typed_rows, target_rows = [], []
    r = ex_tag
    while len(typed_rows) + len(target_rows) < k + m_target:
        r += 1
        if r >= bs.top_row:
            return [], ["@examples stanza hit the function head"]
        line = b.line_str(r)
        if ROXY_TAG_RE.match(line):
            return [], ["@examples stanza hit the next tag"]
        typed_rows.append(r) if len(typed_rows) < k else target_rows.append(r)
    prefix = bs.lines(roxy_start, typed_rows[-1] + 1)
    target_lines = [b.line_str(r) for r in target_rows]
    suffix = [b.line_str(r) for r in
              range(target_rows[-1] + 1, min(b.nlines(), bs.r1 + 1 + 6))]
    deprefixed = "\n".join(re.sub(r"^\s*#'\s?", "", l) for l in target_lines)
    item = dict(package=b.package, path=b.rel, row=target_rows[0],
                prefix=prefix, block=target_lines, suffix=suffix,
                _site=bs.site_id, corpus_target="\n".join(target_lines),
                k_typed=k,
                note=f"@examples continuation: {k} example line(s) typed, "
                     f"complete the author's next {m_target}")
    row = item_to_row("roxygen_examples", "examples_completion", item,
                      carry=("k_typed",))
    ok, reason = _validate_case_row(
        row, None,
        ("r_fragment", {"min_statements": 1, "max_statements": 3,
                        "max_lines": 6}, deprefixed))
    return ([row], []) if ok else ([], [reason])


# ---------------------------------------------------------------------------
# NEW: loop_to_apply (for-loop -> lapply / purrr::map)
# ---------------------------------------------------------------------------

IO_CALLEES = {"readLines", "read.csv", "read.csv2", "read.table", "read.delim",
              "read.delim2", "scan", "download.file", "url", "file",
              "install.packages", "source", "readRDS", "readBin"}
GUARD_CALLEES = {"tryCatch", "withCallingHandlers", "suppressWarnings",
                 "try", "on.exit", "suppressMessages"}
_LOOP_VAR_USE_RE = r"(?<![\w.]){}(?![\w.])"


def _subset2_parts(src: bytes, n):
    """(sequence_text, index_text) of an x[[i]] node, else None. tree-sitter-r
    shapes subset2/subset as SEQ + arguments([[i]]) — the index is the value
    of the LAST argument inside the arguments node."""
    named = [c for c in n.children if c.is_named]
    if len(named) != 2:
        return None
    seq_node, args_node = named
    if args_node.type != "arguments":
        return None
    argv = [V._argument_value(a) for a in args_node.children
            if a.type == "argument"]
    if not argv or argv[-1] is None:
        return None
    return (node_text(src, seq_node).decode("utf-8", "replace"),
            node_text(src, argv[-1]).decode("utf-8", "replace"))


def _same_node(a, b) -> bool:
    """tree-sitter hands out fresh Node wrappers per .children access, so
    Python `is` is unreliable — byte span + type is the stable identity."""
    return a is not None and b is not None and a.type == b.type \
        and a.start_byte == b.start_byte and a.end_byte == b.end_byte


def _var_only_as_index(src: bytes, expr, var: str) -> bool:
    """Every `var` identifier inside expr is the index of an [[ ]] (or [ ]),
    i.e. sits in the argument list of a subset2/subset node (the direct-child
    shape is accepted too, for robustness across grammar versions)."""
    for n in V._walk(expr):
        if n.type != "identifier":
            continue
        if node_text(src, n).decode("utf-8", "replace") != var:
            continue
        p = n.parent
        if p is None:
            return False
        sub = None
        if p.type in ("subset2", "subset"):
            named = [c for c in p.children if c.is_named]
            if named and _same_node(named[-1], n):
                sub = p                       # direct index child
        elif p.type == "argument" and p.parent is not None \
                and p.parent.type == "arguments":
            sub = p.parent.parent             # argument -> arguments -> subset
            if sub is not None and sub.type not in ("subset2", "subset"):
                sub = None
            else:
                named = [c for c in sub.children if c.is_named] if sub else []
                if not named or not _same_node(named[-1], p.parent):
                    sub = None                # not the index position
        if sub is None:
            return False
    return True


def _loop_rewrite_site(bs: BaseSample):
    """Conservative deterministic detector:
    for (VAR in seq_along(SEQ)) { RES[[VAR]] <- EXPR }   (+ optional
    names(RES)[VAR] <- names(SEQ)) with VAR used ONLY as an [[ index]] and
    not referenced after the loop. Returns the rewrite ingredients."""
    src = bs.b.src
    for n in V._walk(bs.body):
        if n.type != "for_statement":
            continue
        anc = n.parent
        nested = False
        while anc is not None and anc is not bs.body:
            if anc.type == "for_statement":
                nested = True
                break
            anc = anc.parent
        if nested:
            continue          # nested loops: not deterministic, try the next
        kids = n.children
        var_node = next((c for c in kids if c.type == "identifier"), None)
        seq_call = next((c for c in kids if c.type == "call"), None)
        br = next((c for c in kids if c.type == "braced_expression"), None)
        if var_node is None or seq_call is None or br is None:
            continue
        var = node_text(src, var_node).decode("utf-8", "replace")
        callee = S.callee_name(src, seq_call)
        args = next((c for c in seq_call.children if c.type == "arguments"),
                    None)
        vals = [V._argument_value(a) for a in
                (args.children if args is not None else [])
                if a.type == "argument"]
        seq_sym = None
        if callee == "seq_along" and vals and vals[0].type == "identifier":
            seq_sym = node_text(src, vals[0]).decode("utf-8", "replace")
        elif callee == "seq_len" and vals and vals[0].type == "call" \
                and S.callee_name(src, vals[0]) == "length":
            inner = [c for c in vals[0].children if c.type == "arguments"]
            iv = [V._argument_value(a) for a in
                  (inner[0].children if inner else []) if a.type == "argument"]
            if iv and iv[0].type == "identifier":
                seq_sym = node_text(src, iv[0]).decode("utf-8", "replace")
        if not seq_sym:
            continue
        stmts = [c for c in br.children if c.is_named and c.type != "comment"]
        if not 1 <= len(stmts) <= 2:
            continue
        targets, names_stmt = [], None
        ok = True
        for st in stmts:
            if st.type != "binary_operator" or len(st.children) < 3 \
                    or node_text(src, st.children[1]) != b"<-":
                ok = False
                break
            lhs, rhs = st.children[0], st.children[2]
            if lhs.type != "subset2":
                ok = False
                break
            parts = _subset2_parts(src, lhs)
            if parts is None or parts[1] != var:
                ok = False
                break
            # names(RES)[[VAR]] <- names(SEQ): the names companion, not an
            # accumulation target
            lhs_seq = parts[0].strip()
            if lhs_seq.startswith("names(") and lhs_seq.endswith(")") \
                    and re.fullmatch(r"names\(\s*[A-Za-z.][\w.]*\s*\)", lhs_seq):
                # names(RES)[[VAR]] <- names(SEQ): the names companion
                if node_text(src, rhs).decode("utf-8", "replace") \
                        != f"names({seq_sym})" or names_stmt is not None:
                    ok = False
                    break
                names_stmt = lhs_seq[6:-1].strip()
                continue
            if not _var_only_as_index(src, rhs, var):
                ok = False
                break
            targets.append((parts[0],
                            node_text(src, rhs).decode("utf-8", "replace")))
        if not ok or not targets:
            continue
        # VAR must not appear after the loop (scoping safety)
        after = bs.lines(n.end_point[0] + 1, bs.r1 + 1)
        blob = S.strip_strings("\n".join(after).encode())
        if re.search(_LOOP_VAR_USE_RE.format(re.escape(var)).encode(), blob):
            continue
        return dict(var=var, seq=seq_sym, targets=targets,
                    names_res=names_stmt, node=n)
    return None


@register("loop_to_apply", family="loop_rewrite",
          determinism="static+validator (shape-gated: accumulation-only loops)",
          reuses="scenarios.strip_strings/callee_name; splice re-parse validator (new)")
def t_loop_rewrite(bs: BaseSample, rng: random.Random, params: dict):
    site = _loop_rewrite_site(bs)
    if site is None:
        return [], ["no accumulation-shaped for-loop (seq_along/seq_len + "
                    "[[i]]<- only; side-effect/nested/break loops rejected)"]
    n = site["node"]
    var, seq, targets = site["var"], site["seq"], site["targets"]
    indent = bs.b.line_str(n.start_point[0])
    ind = indent[:len(indent) - len(indent.lstrip())]
    out, rejects = [], []
    rewrites = ("lapply", "purrr_map")
    for vi, variant in enumerate(rewrites):
        lines = []
        for res, rhs in targets:
            lines.append(ind + (
                f"{res} <- lapply({seq}, function({var}) {rhs})"
                if vi == 0 else
                f"{res} <- purrr::map({seq}, \\({var}) {rhs})"))
        if site["names_res"]:
            lines.append(ind + f"names({site['names_res']}) <- names({seq})")
        prefix = bs.lines(max(0, bs.head_row - 6), n.start_point[0])
        suffix = bs.lines(n.end_point[0] + 1, min(bs.b.nlines(), bs.r1 + 1 + 6))
        new_target = "\n".join(lines)
        if not V.fragment_clean(new_target):
            rejects.append("loop_rewrite: replacement does not parse")
            continue
        if not _splice_reparse(bs, n.start_byte, n.end_byte, new_target):
            rejects.append("loop_rewrite: spliced function does not re-parse")
            continue
        item = dict(package=bs.b.package, path=bs.b.rel, row=n.start_point[0],
                    prefix=prefix, block=lines, suffix=suffix,
                    _site=bs.site_id, corpus_target=new_target,
                    variant="lapply" if vi == 0 else "purrr_map",
                    loop_var=var, seq_sym=seq,
                    note=f"rewrite the accumulation for-loop over {seq} as "
                         f"{'lapply' if vi == 0 else 'purrr::map'} "
                         f"(exact by construction)")
        row = item_to_row("loop_rewrite", "loop_to_apply", item,
                          carry=("variant", "loop_var", "seq_sym"))
        ok, reason = _validate_case_row(row, None, None)
        if ok:
            out.append(row)
        else:
            rejects.append(reason)
    return out, rejects


# ---------------------------------------------------------------------------
# NEW: trycatch_wrap (I/O-doing body -> tryCatch, exact by construction)
# ---------------------------------------------------------------------------

@register("trycatch_wrap", family="trycatch_wrap",
          determinism="static+validator (I/O-signal gate)",
          reuses="validators.handler_clauses as the structural floor (new wiring)")
def t_trycatch(bs: BaseSample, rng: random.Random, params: dict):
    src = bs.b.src
    callees = {S.callee_name(src, n) for n in V._walk(bs.body)
               if n.type == "call"}
    if not (callees & IO_CALLEES):
        return [], ["no I/O-ish call in the body"]
    if callees & GUARD_CALLEES:
        return [], ["body already guards conditions"]
    stmts = C._body_statements(bs.body)
    if not stmts:
        return [], ["empty body"]
    first, last = stmts[0].start_point[0], stmts[-1].end_point[0]
    if last >= bs.r1:
        return [], ["statement run reaches the closing brace"]
    inner = bs.lines(first, last + 1)
    ind0 = inner[0][:len(inner[0]) - len(inner[0].lstrip())]
    deeper = [ind0 + "  " + l[len(ind0):] if l.startswith(ind0) else "  " + l
              for l in inner]
    fn_name = ""
    parent = bs.fn.parent
    if parent is not None and parent.type == "binary_operator" and parent.children:
        fn_name = node_text(src, parent.children[0]).decode("utf-8", "replace")
    where = f" in {fn_name}" if fn_name else ""
    new_lines = [f"{ind0}tryCatch({{"] + deeper + \
        [f"{ind0}}}, error = function(e) {{",
         f'{ind0}  stop("failed{where}: ", conditionMessage(e))',
         f"{ind0}}})"]
    prefix = bs.lines(max(0, bs.head_row - 6), first)
    suffix = bs.lines(last + 1, min(bs.b.nlines(), bs.r1 + 1 + 6))
    target = "\n".join(new_lines)
    if not V.fragment_clean(target):
        return [], ["trycatch_wrap: target does not parse"]
    if not _splice_reparse(bs, stmts[0].start_byte, stmts[-1].end_byte,
                           target):
        return [], ["trycatch_wrap: spliced function does not re-parse"]
    handler_part = ('error = function(e) {\n  stop("failed: ", '
                    'conditionMessage(e))\n})')
    ok_h, reason_h = V.REGISTRY["handler_clauses"](handler_part,
                                                   {"max_len": 500})
    if not ok_h:
        return [], [f"trycatch_wrap: handler floor: {reason_h}"]
    item = dict(package=bs.b.package, path=bs.b.rel, row=first,
                prefix=prefix, block=new_lines, suffix=suffix,
                _site=bs.site_id, corpus_target=target, io_signal=sorted(
                    callees & IO_CALLEES),
                note=f"wrap the I/O-doing body of {fn_name or '<anon>'} in "
                     f"tryCatch (exact by construction)")
    row = item_to_row("trycatch_wrap", "trycatch_wrap", item,
                      carry=("io_signal",))
    ok, reason = _validate_case_row(row, None, None)
    return ([row], []) if ok else ([], [reason])


# ---------------------------------------------------------------------------
# transform wiring
# ---------------------------------------------------------------------------

@register("mbe_arg_edit", family="mid_body_edit", determinism="pure-static",
          reuses="cases.corpus._arg_edit_cands + _mid_body_item; rc mid_body_edit_site")
def _t_arg(bs, rng, p):
    return _mbe_transform(lambda bs: C._arg_edit_cands(bs.b, bs.body),
                          "arg_edit")(bs, rng, p)


@register("mbe_na_rm_insert", family="mid_body_edit", determinism="pure-static",
          reuses="cases.corpus._na_rm_cands + _mid_body_item; rc mid_body_edit_site")
def _t_narm(bs, rng, p):
    return _mbe_transform(lambda bs: C._na_rm_cands(bs.b, bs.body),
                          "na_rm_insert")(bs, rng, p)


@register("mbe_rename_once", family="mid_body_edit", determinism="pure-static",
          reuses="cases.corpus._rename_cands + _mid_body_item; rc mid_body_edit_site")
def _t_ren(bs, rng, p):
    return _mbe_transform(lambda bs: C._rename_cands(bs.b, bs.fn, bs.body),
                          "rename_once")(bs, rng, p)


@register("mbe_insert_line", family="mid_body_edit", determinism="pure-static",
          reuses="cases.corpus._insert_cands + _mid_body_item; rc mid_body_edit_site")
def _t_ins(bs, rng, p):
    return _mbe_transform(lambda bs: C._insert_cands(bs.b, bs.r0, bs.r1),
                          "insert_line")(bs, rng, p)


def run_matrix(samples, params) -> tuple[list[dict], dict]:
    rng = random.Random(int(params.get("seed", 13)))
    rows_out: list[dict] = []
    stats = dict(per_transform={}, per_sample=[], failures=[])
    # event-driven families are per-bundle: draw once, filter per sample
    per_bundle: dict[int, dict] = {}
    for bs in samples:
        if id(bs.b) not in per_bundle:
            per_bundle[id(bs.b)] = dict(
                rename_propagation=S.extract_rename(bs.b, rng),
                na_rm_propagation=S.extract_na_rm(bs.b, rng))
        for tname, meta in TRANSFORMS.items():
            t0 = time.time()
            try:
                rows, rejects = meta["fn"](bs, rng, params)
            except Exception as e:              # noqa: BLE001 — prototype
                rows, rejects = [], [f"EXC {type(e).__name__}: {e}"]
            dt = time.time() - t0
            slot = stats["per_transform"].setdefault(
                tname, dict(family=meta["family"],
                            determinism=meta["determinism"],
                            attempted=0, derived=0, rejected=0))
            slot["attempted"] += 1
            slot["derived"] += len(rows)
            slot["rejected"] += 1 if not rows else 0
            for r in rejects:
                stats["failures"].append(
                    dict(transform=tname, base_sample=bs.site_id,
                         package=bs.b.package, path=bs.b.rel,
                         reason=r[:140], seconds=round(dt, 3)))
            for row in rows:
                row["determinism"] = meta["determinism"]
                rows_out.append(row)
        # event-driven scenario rows (validated by scenarios.validate_example)
        for fam, exs in per_bundle[id(bs.b)].items():
            kept = _scenario_rows(bs, exs)
            for ex in kept:
                try:
                    S.validate_example(ex)
                except AssertionError as e:
                    stats["failures"].append(
                        dict(transform=fam, base_sample=bs.site_id,
                             package=bs.b.package, path=bs.b.rel,
                             reason=f"validate_example: {e}"[:140], seconds=0))
                    continue
                ex = dict(ex, transform=fam, base_sample=f"bs:{bs.site_id}",
                          determinism="pure-static",
                          validator="scenarios.validate_example")
                rows_out.append(ex)
                slot = stats["per_transform"].setdefault(
                    fam, dict(family=fam, determinism="pure-static",
                              attempted=0, derived=0, rejected=0))
                slot["attempted"] += 1
                slot["derived"] += 1
        fams = {r.get("family") for r in rows_out
                if r.get("base_sample") == f"bs:{bs.site_id}"}
        stats["per_sample"].append(
            dict(site=bs.site_id, package=bs.b.package, path=bs.b.rel,
                 body_lines=bs.nbody, families=sorted(fams),
                 n_families=len(fams),
                 n_rows=sum(1 for r in rows_out
                            if r.get("base_sample") == f"bs:{bs.site_id}")))
    return rows_out, stats


# ---------------------------------------------------------------------------
# wide probe: hit rates of the shape-gated NEW transforms + @examples coverage
# ---------------------------------------------------------------------------

def probe(rng: random.Random, params: dict) -> dict:
    n_pkgs = int(params.get("probe_packages", 200))
    time_budget = float(params.get("probe_time_budget_s", 420))
    tidy = S.tidy_packages()
    rng.shuffle(tidy)
    rest = [p for p in S.list_packages() if p not in set(tidy)]
    rng.shuffle(rest)
    pool = tidy[:n_pkgs // 2] + rest[:n_pkgs // 2]
    st = dict(packages_scanned=0, files=0, functions=0, loop_sites=0,
              trycatch_sites=0, comment_fns=0, roxygen_fns=0,
              namespace_fns=0, examples_blocks=0, examples_code_lines=0,
              examples_parse_ok=0, examples_len_hist={}, elapsed_s=0)
    seen_pkgs = set()
    t0 = time.time()
    for b in C.iter_bundles_highest(pool, rng):
        if time.time() - t0 > time_budget:
            break
        st["files"] += 1
        if b.package not in seen_pkgs:
            seen_pkgs.add(b.package)
            st["packages_scanned"] += 1
        for fn in (n for n in V._walk(b.tree.root_node)
                   if n.type == "function_definition"):
            st["functions"] += 1
            geom = C._fn_body(b, fn)
            if geom is None:
                continue
            try:
                bs = BaseSample(b, fn, -1)
            except ValueError:
                continue
            if _loop_rewrite_site(bs):
                st["loop_sites"] += 1
            src = b.src
            callees = {S.callee_name(src, n) for n in V._walk(bs.body)
                       if n.type == "call"}
            if callees & IO_CALLEES and not (callees & GUARD_CALLEES):
                st["trycatch_sites"] += 1
            if _plain_comment_rows(bs):
                st["comment_fns"] += 1
            # per-function namespace-qualify signal (>= 2 same-pkg :: uses)
            occ: dict[str, int] = {}
            for r in range(bs.top_row, bs.r1 + 1):
                line = b.line_str(r)
                if not line or line.lstrip().startswith("#"):
                    continue
                for m in C._NS_RE.finditer(line):
                    occ[m.group(1)] = occ.get(m.group(1), 0) + 1
            if any(v >= 2 for v in occ.values()):
                st["namespace_fns"] += 1
            got = _examples_block(bs)
            if got:
                code_lines, _rs, _et = got
                st["roxygen_fns"] += 1
                if len(code_lines) >= 2:
                    st["examples_blocks"] += 1
                    st["examples_code_lines"] += len(code_lines)
                    bucket = min(len(code_lines), 30)
                    st["examples_len_hist"][str(bucket)] = \
                        st["examples_len_hist"].get(str(bucket), 0) + 1
                    dep = "\n".join(code_lines)
                    if V.fragment_clean(dep):
                        st["examples_parse_ok"] += 1
    st["elapsed_s"] = round(time.time() - t0, 1)
    return st


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python cases/compound.py")
    ap.add_argument("--base-samples", type=int, default=20)
    ap.add_argument("--probe-packages", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stats", type=Path, default=STATS_DEFAULT)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args(argv)

    params = dict(seed=args.seed, cap=2, window_lines=10)
    t0 = time.time()
    rng = random.Random(args.seed)
    samples, funnel = collect_base_samples(rng, args.base_samples, params)
    print(f"[base] {len(samples)} base samples "
          f"(funnel: {json.dumps(funnel)}) "
          f"in {time.time()-t0:.0f}s")
    if not samples:
        sys.exit("no base samples collected")

    rows, stats = run_matrix(samples, params)
    per_sample = stats["per_sample"]
    fam_counts = [p["n_families"] for p in per_sample]
    print(f"[matrix] {len(rows)} validated scenario rows from "
          f"{len(samples)} base samples "
          f"(mean families/sample {sum(fam_counts)/len(fam_counts):.1f}, "
          f"min {min(fam_counts)}, max {max(fam_counts)}); "
          f"failures logged: {len(stats['failures'])}")
    for tname, slot in sorted(stats["per_transform"].items()):
        print(f"  {tname:22s} fam={slot['family']:32s} "
              f"derived={slot['derived']:3d} over {slot['attempted']} samples")

    probe_st = {}
    if not args.skip_probe:
        tp = time.time()
        probe_st = probe(random.Random(args.seed + 1),
                         dict(probe_packages=args.probe_packages))
        est = 14202 / max(1, probe_st.get("packages_scanned", 1))
        print(f"[probe] packages={probe_st['packages_scanned']} "
              f"files={probe_st['files']} functions={probe_st['functions']} "
              f"in {time.time()-tp:.0f}s")
        print(f"  loop_rewrite sites: {probe_st['loop_sites']} "
              f"({100*probe_st['loop_sites']/max(1,probe_st['functions']):.2f}% "
              f"of functions) | trycatch_wrap sites: "
              f"{probe_st['trycatch_sites']} "
              f"({100*probe_st['trycatch_sites']/max(1,probe_st['functions']):.2f}%) "
              f"| namespace-qualify fns: {probe_st['namespace_fns']} "
              f"({100*probe_st['namespace_fns']/max(1,probe_st['functions']):.1f}%) "
              f"| comment fns: {probe_st['comment_fns']} "
              f"({100*probe_st['comment_fns']/max(1,probe_st['functions']):.1f}%)")
        print(f"  @examples blocks(>=2 code lines): {probe_st['examples_blocks']} "
              f"| parse-clean: {probe_st['examples_parse_ok']} "
              f"({100*probe_st['examples_parse_ok']/max(1,probe_st['examples_blocks']):.0f}%) "
              f"| corpus estimate ~{int(probe_st['examples_blocks']*est):,} blocks, "
              f"~{int(probe_st['examples_parse_ok']*est):,} parse-clean "
              f"(x{est:.0f} extrapolation)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(base_samples=len(samples), funnel=funnel, rows=len(rows),
                  mean_families_per_sample=round(sum(fam_counts) /
                                                  len(fam_counts), 2),
                  min_families=min(fam_counts), per_transform=stats[
                      "per_transform"], per_sample=per_sample,
                  failures=stats["failures"][:400],
                  failure_count=len(stats["failures"]), probe=probe_st,
                  llm_calls=0, seed=args.seed,
                  elapsed_s=round(time.time() - t0, 1))
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(report, indent=1))
    print(f"[out] {args.out} + {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
