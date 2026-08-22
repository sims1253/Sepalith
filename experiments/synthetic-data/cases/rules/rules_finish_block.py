"""rules_finish_block.py — compound finish_block cut-point rules (registry).

The finish_block family is the mixture's biggest (100,715 source records)
but its corpus rows come from ONE cut geometry per function
(after-signature, or the fixed first-third mid cut) on CRAN functions that
ALMOST ALWAYS carry roxygen. glm-5.3 beat v7 on the landscape intent layer
here: general body-filling knowledge is the gap, and the deterministic
levers are CUT DIVERSITY + DOMAIN DIVERSITY
(docs/research/landscape-v7-vs-glm53.md; compounding doc 2.1).

This module applies the modular/compound machinery to the family: ONE base
sample (a full corpus function, or an authored one) -> MANY finish_block
records at DIFFERENT truncation depths and DIFFERENT context packagings.
Every ground truth is the verbatim corpus remainder (D1 pure-static); the
prompt side is the only thing that varies.

Rules (all family=finish_block, extends the existing mixture family):

  fb_cut_signature     cut after the signature (`{` only typed); target =
                       the whole body — the HARDEST completion
  fb_cut_after_first   cut after the first top-level statement; target = the
                       rest of the body
  fb_cut_mid_nested    cut BEFORE a nested close: the typed prefix ends
                       inside a nested braced block, the target STARTS with
                       the block's closing line (the model must emit the
                       close + the remaining statements)
  fb_cut_before_return cut before the final return-shaped statement — the
                       EASY end of the difficulty axis
  fb_cut_random        RANDOMIZED cut depth (the positional-realism fix):
                       the four cut rules above train the model on FOUR
                       discrete, patterned depths; fb_cut_random picks a
                       random STATEMENT BOUNDARY — start row of any
                       statement at any nesting depth, top-level or inside
                       a nested braced block — so the model must complete
                       from ANYWHERE. Uniform over candidate boundary rows
                       (seeded per-base_sample_id shuffle, first boundary
                       passing the family floors; document choice: uniform
                       over BOUNDARIES, not line-count-weighted — every cut
                       position equally likely, the maximally
                       position-agnostic choice). Same corpus-exact target,
                       same gates + validator manifest as the fixed cuts;
                       nested-CLOSE rows stay with fb_cut_mid_nested (their
                       geometry differs: the target LEADS with `}`).
  fb_docstring_strip   the docstring-variant axis (user addition): every cut
                       also emitted with the roxygen block STRIPPED from the
                       prompt. Corpus rows are docstring-almost-always; real
                       users type bare signatures and still expect
                       completion. Negative: a base sample with NO docstring
                       yields NO site (strip is a no-op — never double-emit).
  fb_ctx_outline       context-variant: same cut + a deterministic outline
                       header (statement counts + param names; ZERO
                       target-text leakage)
  fb_ctx_diag          context-variant: same cut + a static diagnostics slot
                       (typed-code facts only: statement counts, unused
                       args, unclosed-brace count)

Row schema: the family's own record shape (kind/prefix/target/fn/gated —
drop-in for assemble_sft_v5.load_finish_block, which dedups on target and
renders via render_finish_block) EXTENDED with the cases conventions
(case/backend/model/transform/corpus_target) and the registry parent-link
contract (transform-rule-registry.md 3.2): base_sample_id + rule_id@version
+ params, plus pair_key grouping the packaging variants of the SAME cut.

Selftest note: the SHARED registry gate (check_rewrite_row) enforces a
3-line/1-statement floor that is lint-rewrite shaped — finish_block targets
are body completions. SELFTEST therefore carries (a) restraint negatives
and (b) short-remainder positives that DO pass the shared floor; the FULL
positive suite (long targets, brace-leading mid_nested targets, docstring
pairing, render compatibility) runs under the family gate in
`finish_block_compound.py --selftest` (FAMILY_SELFTEST below).
"""
from __future__ import annotations

import hashlib
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))   # synthetic-data/

import scenarios as S                         # noqa: E402
from scenarios import node_text               # noqa: E402
import cases.corpus as C                      # noqa: E402
import cases.validators as V                  # noqa: E402
from cases.compound import BaseSample         # noqa: E402
from cases.rules import (Rewrite, Site, base_sample_id,   # noqa: E402
                         derivation_key, rule)

# family floors (assemble_sft_v5 / finish_block.py conventions)
MIN_TARGET_CHARS = 30           # MIN_TARGET_CHARS in the mixture
MAX_PROMPT_TARGET_CHARS = 5800  # MAX_CHARS_V1 (6000) minus render markers
MAX_TARGET_NB = 40              # finish_block.py signature ceiling
MAX_HEAD_NB = 25                # finish_block.py mid-body head ceiling
MIN_BODY_NB = 1                 # the family emits 1-statement bodies; the
                                # 30-char target floor handles triviality
MAX_RETURN_NB = 8               # a "final return" longer than this is a block

GENERATED_AT = "2026-08-20T00:00:00"           # fixed: deterministic output

ROXY_RE = re.compile(r"^\s*#'(?:\s|$)")   # compound.py convention (incl. bare #')
CUT_ORDER = ("before_return", "after_first", "mid_nested", "signature")


# ---------------------------------------------------------------------------
# validator manifest + environment stamp (registry convention, this module
# is the reference pattern): every row names WHICH validator accepted it
# (tool@version), and the run's .stats sidecar carries the environment
# stamp — if a tool changes semantics later, affected rows are selectively
# re-gated by tool+version instead of regenerating whole families.
# ---------------------------------------------------------------------------

_MANIFEST: dict | None = None
_ENV_STAMP: dict | None = None


def _tool_version(cmd: list[str]) -> str:
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = (r.stdout or "").strip() or (r.stderr or "").strip()
        return out.splitlines()[0][:80] if out else "?"
    except Exception as e:                       # noqa: BLE001 — stamp is best-effort
        return f"unavailable ({type(e).__name__})"


def gates_manifest() -> dict:
    """The validators that accepted the row (family_gate's checks)."""
    global _MANIFEST
    if _MANIFEST is None:
        import importlib.metadata as md

        def _pkg(name: str) -> str:
            try:
                return f"{name}@{md.version(name)}"
            except Exception:                    # noqa: BLE001
                return f"{name}@?"

        _MANIFEST = dict(
            parse=f"{_pkg('tree-sitter-r')} + {_pkg('tree-sitter')} "
                  f"(validators.fragment_clean)",
            splice_exact="internal@rules_finish_block@1 (verbatim corpus "
                         "byte-range assert)",
            render_compat="assemble_sft_v5.render_finish_block@v5 "
                          "('{'+'\\n' split, 6000-char cap, 30-char floor)",
        )
    return dict(_MANIFEST)


def environment_stamp() -> dict:
    """Run-level tool versions for the .stats sidecar (captured once)."""
    global _ENV_STAMP
    if _ENV_STAMP is None:
        import platform

        _ENV_STAMP = dict(
            captured_at=GENERATED_AT,
            python=platform.python_version(),
            tree_sitter_r=gates_manifest()["parse"],
            r=_tool_version(["Rscript", "--version"]),
            jarl=_tool_version(["jarl", "--version"]),
            ry=_tool_version(["ry", "--version"]),
            air=_tool_version(["air", "--version"]),
            jarl_used="no (finish_block rules are tree-sitter + splice-exact "
                      "only; jarl/ry/air versions stamped for cross-family "
                      "re-gating bookkeeping)",
        )
    return dict(_ENV_STAMP)


# ---------------------------------------------------------------------------
# base-sample statics shared by every finish_block rule
# ---------------------------------------------------------------------------

def _txt(src, n) -> str:
    return node_text(src, n).decode("utf-8", "replace")


def _lhs_name(bs: BaseSample) -> str | None:
    parent = bs.fn.parent
    if parent is None or parent.type != "binary_operator" or not parent.children:
        return None
    return _txt(bs.b.src, parent.children[0]).strip()


def roxygen_rows(bs: BaseSample) -> list[int]:
    """Contiguous roxygen (#') rows immediately above the defining statement
    (finish_block.py collect_functions convention)."""
    b = bs.b
    rows = []
    r = bs.top_row - 1
    while r >= 0 and ROXY_RE.match(b.line_str(r)):
        rows.append(r)
        r -= 1
    rows.reverse()
    return rows


def _gated(roxy_lines: list[str]) -> bool:
    """finish_block.py intent-gate: rich roxygen (>=12 words or @param/@return)."""
    text = "\n".join(roxy_lines)
    if not text:
        return False
    plain = re.sub(r"#'\s*@", "@", text)
    return len(plain.split()) >= 12 or ("@param" in text or "@return" in text)


def _nb(b, r0: int, r1: int) -> int:
    return sum(1 for r in range(r0, r1) if b.line_str(r).strip())


def _stmts(bs: BaseSample) -> list:
    return C._body_statements(bs.body)


_ASSIGN_OPS = (b"<-", b"=", b"<<-")
_LOOP_TYPES = ("for_statement", "while_statement", "repeat_statement")


def _return_shaped(stmt, src=None) -> bool:
    """The final statement reads as the function's value: anything EXCEPT an
    assignment (binary `<-`/`=`/`<<-`) or a bare loop — `env / max(env)` is
    a value; `a <- a * 2` is not."""
    if stmt.type in _LOOP_TYPES:
        return False
    if stmt.type == "binary_operator" and len(stmt.children) >= 2:
        op = stmt.children[1]
        if src is None:
            return False
        if node_text(src, op) in _ASSIGN_OPS:
            return False
    return True


def _nested_close_rows(bs: BaseSample) -> list[int]:
    """Rows that CLOSE a nested braced block (if/for/while/repeat/anonymous
    fn/tryCatch body) strictly inside the function body: the row's first
    non-ws char is `}` and the block spans more than one row."""
    b, out = bs.b, []
    for st in _stmts(bs):
        for n in V._walk(st):
            if n.type != "braced_expression":
                continue
            cr = n.end_point[0]
            if not (bs.r0 < cr < bs.r1) or n.start_point[0] >= cr:
                continue                  # one-line block: nothing to close
            if not b.line_str(cr).lstrip().startswith("}"):
                continue                  # close shares its row with code
            if cr not in out:
                out.append(cr)
    return out


def _absorb_blanks(b, sb: int, floor_row: int) -> int:
    """Move the cut byte UP over blank rows so the typed prefix ends on a
    non-blank line (the family's own heads end at statement ends); the blank
    lines move to the FRONT of the target — byte-contiguous and still
    corpus-exact (signature-kind targets already start with the newline
    after `{`)."""
    row, _col = b.rowcol(sb)
    while row - 1 > floor_row and not b.line_str(row - 1).strip():
        sb = b.starts[row - 1]
        row -= 1
    return sb


# ---------------------------------------------------------------------------
# cut finding — deterministic, restraint-first
# ---------------------------------------------------------------------------

def _finalize_cut(bs: BaseSample, sb: int, kind: str, cut: str,
                  note: str) -> dict | None:
    """Shared tail of every cut geometry: absorb blanks above the cut byte,
    slice head/target, apply the family floors + restraints. Returns the
    info dict or None (restraint)."""
    b = bs.b
    sb = _absorb_blanks(b, sb, bs.r0)
    cut_row = b.rowcol(sb)[0]
    body_sb, body_eb = bs.body.start_byte, bs.body.end_byte
    tgt_eb = body_eb - 1               # target ends just before the body `}`
    head = b.src[body_sb + 1:sb].decode("utf-8", "replace")
    target = b.src[sb:tgt_eb].decode("utf-8", "replace")

    # family floors + restraints
    if len(target.strip()) < MIN_TARGET_CHARS:
        return None                    # trivial remainder: skip (the mixture
                                       # forbids empty targets outside no_op)
    tgt_nb = _nb(b, cut_row, bs.r1)
    if tgt_nb > MAX_TARGET_NB:
        return None
    if cut != "signature":
        head_nb = _nb(b, bs.r0 + 1, cut_row)
        if head_nb < 1 or head_nb > MAX_HEAD_NB:
            return None
    # site row = the target's first NON-BLANK line (the corpus_line the
    # verify contract re-derives; the cut byte can sit at end-of-row — the
    # signature cut — or on an absorbed blank row)
    crow, ccol = b.rowcol(sb)
    trow = crow
    while trow < bs.r1:
        seg = b.line_str(trow)[ccol:] if trow == crow else b.line_str(trow)
        if seg.strip():
            break
        trow += 1
    return dict(kind=kind, cut=cut, row=trow, sb=sb, eb=tgt_eb,
                row_end=b.rowcol(tgt_eb - 1)[0], head=head, target=target,
                note=note)


def find_cut(bs: BaseSample, cut: str) -> dict | None:
    """One cut geometry on the base sample, or None when a restraint fires
    (trivial remainder, one-line statement collisions, family floors).
    Returns dict(kind, cut, row, sb, eb, row_end, head, target, note)."""
    b = bs.b
    if _lhs_name(bs) is None:
        return None                    # the family prefix needs `name <- fn`
    if _nb(b, bs.r0 + 1, bs.r1) < MIN_BODY_NB:
        return None
    body_sb, body_eb = bs.body.start_byte, bs.body.end_byte
    tgt_eb = body_eb - 1               # target ends just before the body `}`

    if cut == "signature":
        sb = body_sb + 1
        kind, note = "signature", (
            "after-signature cut: only the brace typed, complete the body")
    elif cut == "after_first":
        stmts = _stmts(bs)
        if len(stmts) < 2:
            return None
        if stmts[1].start_point[0] <= stmts[0].end_point[0]:
            return None                # statements share a row: not cuttable
        sb = b.starts[stmts[1].start_point[0]]
        kind, note = "mid_body", (
            "after-first-statement cut: first statement typed, complete the "
            "remainder")
    elif cut == "mid_nested":
        cand = None
        for cr in _nested_close_rows(bs):
            if _nb(b, bs.r0 + 1, cr) < 1:
                continue               # nothing typed: that is the signature cut
            if _nb(b, cr, bs.r1) < 2:
                continue               # remainder is just the close: trivial
            cand = cr
            break
        if cand is None:
            return None
        sb = b.starts[cand]
        kind, note = "mid_body", (
            "mid-block cut before a nested close: the completion starts "
            "with the closing line of the nested block")
    elif cut == "before_return":
        stmts = _stmts(bs)
        if len(stmts) < 2 or not _return_shaped(stmts[-1], b.src):
            return None
        if stmts[-1].start_point[0] <= stmts[-2].end_point[0]:
            return None
        if stmts[-1].end_point[0] - stmts[-1].start_point[0] + 1 > MAX_RETURN_NB:
            return None
        sb = b.starts[stmts[-1].start_point[0]]
        kind, note = "mid_body", (
            "before-final-return cut: body typed through the second-to-last "
            "statement, produce the return value")
    else:
        raise ValueError(cut)

    return _finalize_cut(
        bs, sb, kind, cut, note)


def cut_sites(bs: BaseSample, cuts: tuple[str, ...]) -> list[Site]:
    """Detector body shared by the rules: one Site per surviving cut,
    ordered so sites[0] carries the SHORTEST remainder (the shared registry
    gate's 3-line/1-statement floor can then exercise it in SELFTEST)."""
    out: list[Site] = []
    for cut in CUT_ORDER:
        if cut not in cuts:
            continue
        info = find_cut(bs, cut)
        if info is None:
            continue
        out.append(Site(row=info["row"], sb=info["sb"], eb=info["eb"],
                        row_end=info["row_end"], payload=info,
                        note=info["note"]))
    return out


# ---------------------------------------------------------------------------
# randomized cut depth (fb_cut_random): a statement boundary at ANY nesting
# depth — the positional-realism complement to the four fixed geometries
# ---------------------------------------------------------------------------

def statement_boundary_rows(bs: BaseSample) -> list[int]:
    """Rows that BEGIN a statement (top-level OR nested): every named
    non-comment direct child of a braced_expression inside the function
    body whose start point is the row's first non-ws character (a clean
    line cut — statements sharing a row with previous code are not
    cuttable, the after_first convention), strictly inside the body, with
    at least one non-blank body row above (something typed). Nested CLOSE
    rows are deliberately NOT candidates: a close row does not begin a
    statement, and that geometry (target leads with `}`) is owned by
    fb_cut_mid_nested."""
    b, rows, seen = bs.b, [], set()
    for st in _stmts(bs):
        for holder in (st, *[
                n for n in V._walk(st) if n.type == "braced_expression"]):
            for child in holder.children:
                if not child.is_named or child.type == "comment":
                    continue
                r, c = child.start_point
                if not (bs.r0 < r < bs.r1) or r in seen:
                    continue
                if b.line_str(r)[:c].strip():
                    continue     # statement starts mid-row: not a boundary
                if not any(b.line_str(rr).strip()
                           for rr in range(bs.r0 + 1, r)):
                    continue     # nothing typed above: that is the
                                  # signature cut's geometry
                seen.add(r)
                rows.append(r)
    rows.sort()
    return rows


def random_cut_site(bs: BaseSample) -> Site | None:
    """The fb_cut_random detector: ONE cut per base sample at a random
    statement boundary. CHOICE DOCUMENTED: uniform over candidate
    BOUNDARY rows (not line-count-weighted) — every admissible cut
    position is equally likely, the maximally position-agnostic prior;
    line-weighting would skew cuts toward long statements. Reproducible:
    the RNG is seeded per base_sample_id (the stable content-hash id), so
    the same function always yields the same random cut regardless of
    scan order, reruns or cache resets. The seeded shuffle orders the
    candidates; the FIRST boundary passing the family floors wins —
    uniform over admissible boundaries (floors are geometry facts, not
    position preferences)."""
    if _lhs_name(bs) is None or _nb(bs.b, bs.r0 + 1, bs.r1) < MIN_BODY_NB:
        return None
    rng = random.Random(f"fb_cut_random@1:{base_sample_id(bs)}")
    cands = statement_boundary_rows(bs)
    rng.shuffle(cands)
    for r in cands:
        info = _finalize_cut(
            bs, bs.b.starts[r], "mid_body", "random",
            "randomized-depth cut: cursor at a uniformly drawn statement "
            "boundary (any nesting depth) — complete the remainder")
        if info is None:
            continue
        return Site(row=info["row"], sb=info["sb"], eb=info["eb"],
                    row_end=info["row_end"], payload=info,
                    note=info["note"])
    return None


def rederive_cut(bs: BaseSample, cut: str) -> dict | None:
    """family_gate's re-derivation hook: the fixed geometries re-derive
    through find_cut; the random cut re-derives through its own SEEDED
    detector (deterministic per base_sample_id, so the gate assert is
    stable)."""
    if cut == "random":
        site = random_cut_site(bs)
        return None if site is None else site.payload
    return find_cut(bs, cut)


# ---------------------------------------------------------------------------
# packaging: prefix assembly + the variant axes
# ---------------------------------------------------------------------------

def _param_names(bs: BaseSample) -> list[str]:
    for c in bs.fn.children:
        if c.type == "parameters":
            names = []
            for p in c.children:
                if p.type != "parameter":
                    continue
                nm = next((k for k in p.children if k.type == "identifier"),
                          None)
                if nm is not None:
                    names.append(_txt(bs.b.src, nm))
            return names
    return []


def _typed_usage(bs: BaseSample, sb: int) -> set[str]:
    """Identifiers referenced in the TYPED part of the body only (above the
    cut byte) — feeds the diagnostics slot without touching the remainder."""
    used = set()
    for n in V._walk(bs.body):
        if n.start_byte >= sb or n.type != "identifier":
            continue
        used.add(_txt(bs.b.src, n))
    return used


def _outline_block(bs: BaseSample, sb: int) -> list[str]:
    """Deterministic outline header: structure facts ONLY (statement counts,
    param names, docstring presence). Zero remainder-text leakage."""
    n_stmts = len(_stmts(bs))
    k = len([st for st in _stmts(bs) if st.end_byte <= sb])
    params = ", ".join(_param_names(bs)[:8]) or "none"
    docs = "yes" if roxygen_rows(bs) else "no"
    return [
        "# ---- outline " + "-" * 32,
        f"# fn: {_lhs_name(bs) or '<anon>'}({params})",
        f"# plan: {n_stmts} top-level statements; {k} typed above the cursor",
        f"# docs: {docs}",
        "# " + "-" * 46,
    ]


def _diag_block(bs: BaseSample, sb: int) -> list[str]:
    """Deterministic diagnostics slot: static facts about the TYPED code."""
    n_stmts = len(_stmts(bs))
    k = len([st for st in _stmts(bs) if st.end_byte <= sb])
    unused = [p for p in _param_names(bs)
              if p not in _typed_usage(bs, sb)]
    typed = S.strip_strings(bs.b.src[bs.body.start_byte:sb])
    open_b = typed.count(b"{") - typed.count(b"}")
    return [
        "# ---- diagnostics (static) " + "-" * 20,
        f"# typed statements: {k} of {n_stmts}",
        f"# args not yet referenced: {', '.join(unused[:8]) or 'none'}",
        f"# unclosed braces in typed code: {open_b}",
        "# " + "-" * 46,
    ]


def build_prefix(bs: BaseSample, info: dict, docstring: str = "keep",
                 variant: str = "plain") -> str:
    """The family's exact prefix construction (finish_block.py semantics:
    roxy + name + ' <- ' + sig + '{' [+ '\\n' + head]) with the packaging
    axes applied. Variant blocks sit ABOVE the roxygen — never between
    roxygen and the function, which would detach the docstring."""
    b = bs.b
    name = _lhs_name(bs)
    sig = b.src[bs.fn.start_byte:bs.body.start_byte].decode("utf-8", "replace")
    roxy_text = "\n".join(b.line_str(r) for r in roxygen_rows(bs))
    parts = []
    if variant == "outline":
        parts.append("\n".join(_outline_block(bs, info["sb"])))
    elif variant == "diag":
        parts.append("\n".join(_diag_block(bs, info["sb"])))
    if roxy_text and docstring == "keep":
        parts.append(roxy_text)
    parts.append(f"{name} <- {sig}{{")
    prefix = "\n".join(parts)
    if info["kind"] == "mid_body":
        prefix = prefix + "\n" + info["head"]
    return prefix


def _author_base_id(bs: BaseSample) -> str:
    """Author samples hash the code itself (no corpus provenance)."""
    h = hashlib.sha1()
    h.update(b"author\x00")
    h.update(bs.b.src[bs.fn.start_byte:bs.fn.end_byte])
    return "bs:" + h.hexdigest()[:12]


def finish_block_row(bs: BaseSample, r, site: Site, docstring: str = "keep",
                     variant: str = "plain", prov: dict | None = None,
                     origin: str = "corpus") -> dict:
    """Family-schema record (drop-in for load_finish_block) extended with the
    cases provenance conventions and the registry parent-link block."""
    info = site.payload
    b = bs.b
    prefix = build_prefix(bs, info, docstring=docstring, variant=variant)
    target = info["target"]
    bsid = base_sample_id(bs) if origin == "corpus" else _author_base_id(bs)
    cut_id = f"{info['cut']}@{info['row']}"
    pair_key = f"{bsid}:{cut_id}"
    params = dict(cut=info["cut"], docstring=docstring, variant=variant,
                  site_row=info["row"],
                  site_col=info["sb"] - b.starts[info["row"]],
                  pair_key=pair_key)
    row = dict(
        kind=info["kind"], cut=info["cut"],
        package=(prov or {}).get("package", b.package),
        path=(prov or {}).get("path", b.rel),
        fn=_lhs_name(bs) or "",
        gated=_gated([b.line_str(rr) for rr in roxygen_rows(bs)]),
        prefix=prefix, target=target,
        # cases conventions (provenance + mock-draw; corpus-exact GT)
        family="finish_block", transform=r.id,
        case="finish_block_compound", backend="deterministic",
        model="static-transform", full_prompt="",
        determinism=r.determinism, rl_ready=r.is_rl_ready,
        corpus_target=target, model_target=target,
        generated_at=GENERATED_AT, base_sample=bsid,
        # family provenance (production finish_block_sample.jsonl keys)
        source_url=(prov or {}).get("source_url", ""),
        license=(prov or {}).get("license", ""),
        version=(prov or {}).get("version", ""),
        upstream=(prov or {}).get("upstream", ""),
        seed_domain=(prov or {}).get("seed_domain", ""),
        # registry parent-link contract (transform-rule-registry.md 3.2)
        rule=f"{r.id}@{r.version}",
        derivation=dict(base_sample_id=bsid, rule_id=r.id,
                        rule_version=r.version, params=params,
                        pair_key=pair_key, origin_kind=origin),
        # validator manifest: WHICH validator accepted this row (tool@version)
        gates=gates_manifest(),
        note=info["note"],
    )
    if origin == "author":
        row["origin_kind"] = "author"
    row["derivation_key"] = derivation_key(bsid, r, params)
    row["content_hash"] = hashlib.sha1(
        (f"{r.id}\x00{pair_key}\x00{docstring}\x00{variant}\x00"
         f"{target}").encode("utf-8", "replace")).hexdigest()
    return row


# ---------------------------------------------------------------------------
# the family gate: splice-exact + re-parse + render compatibility
# ---------------------------------------------------------------------------

def family_gate(bs: BaseSample, row: dict) -> tuple[bool, str]:
    """The finish_block-specific row gate (the analogue of check_rewrite_row
    for body-completion targets):
      1. splice-exact — the target is the verbatim corpus remainder and the
         prefix's code tail + target reconstruct the function;
      2. re-parse — the reconstruction parses as clean R;
      3. render compatibility — render_finish_block's requirements
         ('{'+'\\n' split point for mid rows, length caps, target floor)."""
    b = bs.b
    target = row["target"]
    params = row["derivation"]["params"]
    info = rederive_cut(bs, params["cut"])
    if info is None:
        return False, f"cut {params['cut']} no longer derivable (unstable)"
    if info["target"] != target:
        return False, "target is not the verbatim corpus remainder"
    prefix = row["prefix"]
    name = _lhs_name(bs) or ""
    anchor = f"{name} <- "
    if anchor not in prefix:
        return False, "prefix lost the defining assignment"
    code_start = prefix.rindex(anchor)
    recon = prefix[code_start:] + target + "}"
    if not V.fragment_clean(recon):
        return False, "rendered reconstruction does not re-parse"
    if row["kind"] == "mid_body" and "{\n" not in prefix:
        return False, "mid rows need the '{\\n' split point for the head strip"
    if len(target.strip()) < MIN_TARGET_CHARS:
        return False, f"target under the {MIN_TARGET_CHARS}-char mixture floor"
    if not target.strip():
        return False, "empty target (no_op twins are not a finish_block kind)"
    if len(prefix) + len(target) > MAX_PROMPT_TARGET_CHARS:
        return False, "prompt+target over the 6000-char mixture cap"
    return True, ""


def derive_all(bs: BaseSample, prov: dict | None = None,
               origin: str = "corpus") -> tuple[list[dict], dict]:
    """The full derivation matrix on one base sample: every surviving cut x
    {as-is, docstring-stripped} + the outline/diag context variants on two
    cuts each. Deterministic; every row passes family_gate (asserted here —
    a gate failure is a bug, surfaced as a restraint count)."""
    from cases.rules import REGISTRY
    rows: list[dict] = []
    restraints: dict[str, int] = {}    # cut restraints (expected, funnel)
    gate_failures: dict[str, int] = {}  # family_gate rejects (bugs: rows are
                                        # exact-by-construction)
    has_doc = bool(roxygen_rows(bs))

    def emit(rule_id, site, docstring, variant):
        r = REGISTRY[rule_id]
        row = finish_block_row(bs, r, site, docstring=docstring,
                               variant=variant, prov=prov, origin=origin)
        ok, reason = family_gate(bs, row)
        if not ok:
            gate_failures[reason] = gate_failures.get(reason, 0) + 1
            return None
        rows.append(row)
        return row

    sites = {s.payload["cut"]: s for s in cut_sites(bs, CUT_ORDER)}
    for cut in ("signature", "after_first", "mid_nested", "before_return"):
        site = sites.get(cut)
        if site is None:
            restraints[f"cut {cut}: restraint"] = \
                restraints.get(f"cut {cut}: restraint", 0) + 1
            continue
        emit("fb_cut_" + cut, site, "keep", "plain")
        if has_doc:
            emit("fb_docstring_strip", site, "strip", "plain")
    for cut in ("after_first", "before_return"):
        if cut in sites:
            emit("fb_ctx_outline", sites[cut], "keep", "outline")
    for cut in ("after_first", "mid_nested"):
        if cut in sites:
            emit("fb_ctx_diag", sites[cut], "keep", "diag")
    # randomized-depth cut: ONE per base sample (+ docstring twin), the
    # positional-realism axis — completes the matrix with arbitrary depths
    rsite = random_cut_site(bs)
    if rsite is None:
        restraints["cut random: restraint"] = \
            restraints.get("cut random: restraint", 0) + 1
    else:
        emit("fb_cut_random", rsite, "keep", "plain")
        if has_doc:
            emit("fb_docstring_strip", rsite, "strip", "plain")
    return rows, dict(cuts=sorted(sites), random_cut=rsite is not None,
                      has_docstring=has_doc,
                      rows=len(rows), restraints=restraints,
                      gate_failures=gate_failures)


# ---------------------------------------------------------------------------
# rule bodies shared by all seven (detector/rewrite/verify are cut-agnostic;
# each rule declares WHICH cuts it owns)
# ---------------------------------------------------------------------------

def _verify_fb(old_text: str, new_text: str) -> tuple[bool, str]:
    """D1 spot check: the target's first non-blank line is the corpus line at
    the cut row (or its tail, when the cut byte sits mid-row); full
    splice-exactness is proven by family_gate's byte-range assert)."""
    if not new_text.strip():
        return False, "empty completion"
    lines = [l for l in new_text.split("\n") if l.strip()]
    if not old_text.strip() or not lines:
        return True, ""      # cut row is an absorbed blank line: nothing to cmp
    first = lines[0].rstrip("\r")
    old = old_text.rstrip("\r")
    if first != old and not old.rstrip().endswith(first.strip()):
        return False, "first target line != corpus line at the cut"
    return True, ""


def _rewrite_fb(bs: BaseSample, site: Site) -> Rewrite:
    info = site.payload
    lines = [l for l in info["target"].strip("\n").split("\n")]
    return Rewrite(lines=lines, span_text=info["target"],
                   meta=dict(kind=info["kind"], cut=info["cut"]))


def _mk_rule(rid, cuts, signal, restraint, precondition=None):
    @rule(id=rid, family="finish_block", determinism="D1", kind="rewrite",
          requires=["fn_body"], signal=signal, restraint=restraint,
          extends="finish_block", status="extends")
    class _CutRule:
        CUTS = tuple(cuts)

        def detector(self, bs):
            if precondition is not None and not precondition(bs):
                return []
            return cut_sites(bs, self.CUTS)

        def rewrite(self, bs, site):
            return _rewrite_fb(bs, site)

        verify = staticmethod(_verify_fb)
    _CutRule.__name__ = rid
    return _CutRule


_fb_cut_signature = _mk_rule(
    "fb_cut_signature", ("signature",),
    "braced function body: cut after `{` — target = whole body (verbatim)",
    "skip when the body is trivial (<30 chars) or >40 non-blank lines")
_fb_cut_after_first = _mk_rule(
    "fb_cut_after_first", ("after_first",),
    ">=2 top-level statements: cut at the 2nd statement's first row — "
    "target = the remainder",
    "skip when statements share a row (not line-cuttable), head >25 lines, "
    "or the remainder is trivial")
_fb_cut_mid_nested = _mk_rule(
    "fb_cut_mid_nested", ("mid_nested",),
    "nested braced block closing on its own row: cut AT the close row — "
    "the target STARTS with the closing line",
    "skip when nothing is typed above the close, the remainder is just the "
    "close, or the family floors fail")
_fb_cut_before_return = _mk_rule(
    "fb_cut_before_return", ("before_return",),
    "final top-level statement is return-shaped (symbol/call/string): cut "
    "at its first row — target = the value statement",
    "skip when the tail is an assignment, shares a row with the previous "
    "statement, or exceeds 8 lines")


@rule(id="fb_cut_random", family="finish_block", determinism="D1",
      kind="rewrite", requires=["fn_body"],
      signal="any statement boundary (top-level or nested, any depth): "
             "ONE uniformly drawn cut per base sample — RNG seeded per "
             "base_sample_id, so the depth is random across the corpus but "
             "reproducible per function",
      restraint="skip when no statement-boundary row survives the family "
                "floors (trivial remainder, head/target over the family "
                "ceilings, prompt+target over the 6000-char cap)",
      extends="finish_block", status="extends")
class _FbCutRandom:
    def detector(self, bs):
        site = random_cut_site(bs)
        return [site] if site is not None else []

    def rewrite(self, bs, site):
        return _rewrite_fb(bs, site)

    verify = staticmethod(_verify_fb)


_fb_docstring_strip = _mk_rule(
    "fb_docstring_strip", CUT_ORDER,
    "any surviving cut, roxygen present: same cut with the docstring "
    "STRIPPED from the prompt (docstring-presence robustness axis)",
    "NO site when the base sample has no roxygen — strip is a no-op, never "
    "double-emit (the pairing rides derivation.pair_key)",
    precondition=lambda bs: bool(roxygen_rows(bs)))
_fb_ctx_outline = _mk_rule(
    "fb_ctx_outline", ("before_return", "after_first"),
    "context-variant: same cut + deterministic outline header (statement "
    "counts, param names — no target-text leakage)",
    "only on short-head cuts; the outline sits ABOVE the roxygen so it "
    "never detaches the docstring")
_fb_ctx_diag = _mk_rule(
    "fb_ctx_diag", ("before_return", "after_first", "mid_nested"),
    "context-variant: same cut + static diagnostics slot (typed-statement "
    "facts, unused args, unclosed-brace count)",
    "only on cuts whose head is non-empty; the slot sits ABOVE the roxygen")


# ---------------------------------------------------------------------------
# SELFTEST — shared-runner-safe cases (module docstring): restraint
# negatives + short-remainder positives that pass the 3-line/1-statement
# floor. FAMILY_SELFTEST below carries the rest under the family gate.
# ---------------------------------------------------------------------------

# 1-statement body >=30 chars: the ONLY shape whose signature-cut target
# passes the shared floor
_ONE_STMNT = b"""#' Sample variance of a vector
#'
#' @param x numeric vector
varr <- function(x) {
  sum((x - mean(x))^2) / (length(x) - 1)
}
# tail marker
"""

# 2 statements; the remainder after stmt 1 is a single call statement
_TWO_STMNT = b"""#' Summary of a scaled vector
#'
#' @param x numeric vector
#' @return list with mean and sd
summ <- function(x) {
  scaled <- scale(x)
  list(mean = mean(scaled), sd = sd(scaled))
}
# tail marker
"""

_NODOC = b"""tri <- function(x) {
  keep <- x > quantile(x, 0.25, na.rm = TRUE)
  mean(x[keep], na.rm = TRUE, trim = 0.05)
}
# tail marker
"""

_NOTAIL = b"""bad <- function(x) {
  a <- scale(x, center = TRUE, scale = TRUE)
  a <- sweep(a, 2, colMeans(a, na.rm = TRUE), "-")
}
# tail marker
"""

_TINY = b"""one <- function(x) {
  x
}
# tail marker
"""

_NONESTED = b"""flat <- function(x) {
  a <- x + 1
  b <- a * 2
  b
}
# tail marker
"""


def _selftest_cases(rid: str) -> list:
    per = {
        "fb_cut_signature": [
            (_ONE_STMNT, dict(
                expect_sites=1,
                first_new="  sum((x - mean(x))^2) / (length(x) - 1)")),
            (_TINY, dict(expect_sites=0, why="trivial body (restraint)")),
        ],
        "fb_cut_after_first": [
            (_TWO_STMNT, dict(
                expect_sites=1,
                first_new="  list(mean = mean(scaled), sd = sd(scaled))")),
            (b"f <- function(x) {\n  x\n}\n# t\n",
             dict(expect_sites=0, why="single statement: no second cut")),
        ],
        "fb_cut_before_return": [
            (_TWO_STMNT, dict(
                expect_sites=1,
                first_new="  list(mean = mean(scaled), sd = sd(scaled))")),
            (_NOTAIL, dict(expect_sites=0, why="final statement is an "
                           "assignment, not a value")),
        ],
        "fb_cut_random": [
            (_TWO_STMNT, dict(
                expect_sites=1,
                first_new="  list(mean = mean(scaled), sd = sd(scaled))",
                why="the ONLY statement boundary is stmt 2's row — the "
                    "seeded draw is forced there (deterministic per "
                    "base_sample_id)")),
            (_TINY, dict(expect_sites=0,
                         why="single statement: no interior boundary")),
        ],
        "fb_cut_mid_nested": [
            (_NONESTED, dict(expect_sites=0,
                             why="no nested braced block (the positive runs "
                                 "under the family gate: brace-leading "
                                 "targets cannot pass the shared "
                                 "3-line/1-statement floor)")),
        ],
        "fb_docstring_strip": [
            (_TWO_STMNT, dict(
                expect_sites=3,
                why="one site per surviving cut (after_first, "
                    "before_return, signature)")),
            (_NODOC, dict(expect_sites=0,
                          why="NO docstring: strip is a no-op — never "
                              "double-emit (required negative)")),
        ],
        "fb_ctx_outline": [
            (_TWO_STMNT, dict(expect_sites=2)),
            (_TINY, dict(expect_sites=0, why="no surviving cut")),
        ],
        "fb_ctx_diag": [
            (_TWO_STMNT, dict(expect_sites=2)),
            (_TINY, dict(expect_sites=0, why="no surviving cut")),
        ],
    }
    return per[rid]


def _attach_selftests():
    from cases.rules import REGISTRY
    for rid, r in REGISTRY.items():
        if rid.startswith("fb_"):
            r.selftest = _selftest_cases(rid)


_attach_selftests()


# ---------------------------------------------------------------------------
# FAMILY_SELFTEST — the full positive suite under the family gate (run by
# finish_block_compound.py --selftest; exercises long targets, brace-leading
# mid_nested targets, docstring pairing, render compatibility).
# ---------------------------------------------------------------------------

_NESTED = b"""#' Mean of squared integers
#'
#' @param n number of terms
sim <- function(n) {
  acc <- list()
  for (i in seq_len(n)) {
    acc[[i]] <- i * i
  }
  out <- sum(unlist(acc))
  c(n = length(acc), mean = out / n)
}
# tail marker
"""

FAMILY_SELFTEST = [
    (_TWO_STMNT, dict(name="with-docstring 2-statement function",
                      expect_cuts=["after_first", "before_return",
                                   "signature"],
                      expect_stripped=True, pairs_complete=True)),
    (_NESTED, dict(name="nested for-loop with close + tail",
                   expect_cuts=["after_first", "before_return",
                                "mid_nested", "signature"],
                   expect_stripped=True, mid_nested_leads_with_brace=True)),
    (_NODOC, dict(name="no docstring (strip no-op negative)",
                  expect_cuts=["after_first", "before_return", "signature"],
                  expect_stripped=False)),
    (_TINY, dict(name="trivial body", expect_cuts=[])),
    (_NOTAIL, dict(name="assignment tail",
                   expect_cuts=["after_first", "signature"])),
]
