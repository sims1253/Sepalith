#!/usr/bin/env python3
"""eval-v2 leg V1b: AST-equivalence + parse-validity re-scorer (CPU, free).

Re-scores STORED eval rows (docs/research/2026-09-02-eval-strategy-v2.md S3 V1b):
per row `parses`, `parse_tool`, `ast_equiv` under a pre-registered minimal
normalization, aggregated per family as the exact -> ast_equiv GAP (rows that
are right-but-not-exact under the normalizations below). Additive-only: reads
existing results jsonl, writes a new jsonl; never edits the inputs.

Parser tiers (R)
  fast  : tree-sitter + tree-sitter-r (structural tier; also the ONLY tier
          that yields trees for ast_equiv). Fragment-tolerant, see below.
  strict: `Rscript -e 'parse(text=...)'` subprocess (R 4.6). Used for
          `parses` when tree-sitter is not importable in the running
          interpreter, and as a confirming second opinion with --strict.
          NOTE: the strict tier rejects valid mid-call fragments such as
          `quiet2 = quiet,` (trailing comma, top-level) -- that divergence
          is expected and is why tree-sitter is the default parse tier for
          line-fragment rows. Python: the stdlib `ast` module (free tier).

Fragment tolerance (pre-registered; both pred and GT get it symmetrically)
  Predictions are single lines or short regions cut out of larger files, so
  bare `has_error` over-counts broken syntax. We tolerate exactly:
    - MISSING nodes (unclosed `(`/`{`/`[`)  -> note `unclosed_delim`
    - an ERROR node whose entire content is a trailing `,` -> `trailing_comma`
  Any other ERROR node => parses=0. Tolerated nodes are dropped from the
  canonical form, so ast_equiv on an unclosed region is a LOWER BOUND
  (structure beyond the cut is unknown); rows whose pred hit the stored
  400/600-char cap are additionally flagged `truncated: true` -- and when
  the truncated pred is a strict PREFIX of the GT, ast_equiv is null
  (note `truncated_prefix`) because the missing tail makes the comparison
  meaningless in both directions (an exact row must not show a fake -1 gap).

Fragment rows (documented fallback). Some GT regions are mid-expression
  (pipe continuations like `) |>`, regions starting with `}`, an unclosed
  `{` inside call args where tree-sitter-r recovers to a full ERROR): no
  parser can compare them structurally in isolation. When the GT itself
  does not parse, the row is flagged `fragment: true` and ast_equiv falls
  back to the battery's own text normalization (rstrip lines, pop trailing
  blanks) -- exact rows therefore stay ast_equiv=1 and the GAP metric
  counts only structurally comparable rows; `parses` is aggregated over
  complete-expression rows only (a fragment pred's parse-validity is not
  attributable to the model).

PRE-REGISTERED normalizations (ast_equiv == 1 iff all of these hold)
  N1 whitespace/comments: tree structure ignores whitespace; comment nodes
      are dropped. If BOTH trees are comment-only (doc_sync regions), the
      AST is vacuous, so we compare whitespace-collapsed comment TEXT
      (leading `#`-decoration stripped) instead -> note `comments_text_only`.
  N2 literal values: string -> STR, numeric (float/integer/complex) -> NUM;
      the class is kept (TRUE/FALSE/NULL/NA stay distinct), the value is not.
      Blind spot: `1` == `1.0`, any string == any string, and VALUE SWAPS of
      literals (`f(a=1, b=2)` vs `f(a=2, b=1)`) are invisible -- the
      corrupted-twin anchors therefore use identifier args, not literals.
  N3 identifier alpha-rename: ALL identifiers are renamed through a map that
      is a pure function of the tree's identifier SET (sorted names -> v0,
      v1, ...), applied identically to both trees. Consequences, by design:
      WHICH name is used WHERE is preserved, so GT `manual = manual2` vs pred
      `manual2 = manual` (wrong-side rename) is NOT equivalent, and swapped
      positional arguments are NOT equivalent. Tree equality under this map
      == structural equality modulo any order-preserving (by sorted name)
      identifier bijection; N4 then pins the names down, so the COMBINED
      rule is: same structure AND same identifier multiset, i.e. no renaming
      beyond placement-consistent identity -- the battery semantics (the GT
      names its targets), not lambda-calculus alpha-equivalence. Exceptions
      kept verbatim: named-argument NAMES (interface, matched by name) and
      `$`/`@` slots.
  N4 vars check (the N3 blind-spot mitigation): the identifier MULTISET of
      pred must equal that of GT (`x + y` vs `x + z` is tree-equal under N3
      -- both map to v0+v1 -- but scores 0 here, as does any renamed pred).
      If GT text is unavailable for a row, the check is skipped and the row
      is flagged `weak: true` (ast_equiv = tree-equality only, i.e. an
      upper bound). v2 TODO: derive the GT-defined rename bijection from
      scenario metadata (`note`/`event_diff`) so legit GT-pinned renames
      pass.
  N5 named-argument reorder: within one call/`[` argument list, NAMED
      arguments are sorted by name; POSITIONAL arguments keep their order.
      (R semantics: named args can appear anywhere in the list.)
  N6 assignment operator: `x <- v` == `x = v` (binary_operator only;
      `<<-` and `:=` stay distinct).
  N7 pure punctuation (commas, parens/braces as delimiters) does not appear
      in the canonical form; operator tokens do (`x[i]` vs `x[[i]]` differ).

  v2 TODOs (skipped, not faked): parenthesized-expression flattening;
  statement reordering; pipe `|>` vs nested-call equivalence; GT-defined
  identifier-target extraction from scenario metadata (N4 refinement);
  R `deparse(parse(x))` as a third cross-check tier.

Ground-truth re-derivation for stored rows (no serving, read-only)
  scenario rows (fields family+id): id is sha1(prompt)[:12] of the training
  render; we re-render every row of the five family files with the EXACT
  `assemble_sft_v2.edit_row()` and match ids -> GT = rstripped region_new
  (eval_scenarios.py conventions). Needs /mnt/h/sepalith/datasets/scenarios_v1.
  midtyping rows (fields repo+lang+sha): matched to examples.jsonl by
  (repo, path, sha); GT = the changed-lines region of the ORIGINAL example.
  run_eval.make_midtyping's random cut fraction is NOT reproducible (it
  seeded from str.__hash__, randomized per process), so line 0 is aligned by
  `align_midtyping` (whole lines only: as-is if pred line 0 equals GT line 0;
  full-line reconstruction when pred line 0 is a completion suffix with the
  cut inside the construction's [0.3, 0.6] bounds; else unaligned) ->
  notes `midtyping_full_line` / `midtyping_cut=N` / `midtyping_unaligned`.

Usage
  python experiments/eval/ast_equiv.py --results <results.jsonl> [--lang r]
      [--out <out.jsonl>] [--strict] [--scen-dir DIR] [--examples FILE]
Default --out sits next to --results as astequiv_<stem>.jsonl.
"""

import argparse
import ast as pyast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ tiers --
try:  # fast structural tier
    from tree_sitter import Language, Parser

    import tree_sitter_r

    R_PARSER = Parser(Language(tree_sitter_r.language()))
    HAVE_TS = True
except Exception:  # pragma: no cover - depends on interpreter
    R_PARSER, HAVE_TS = None, False

RSCRIPT = shutil.which("Rscript")

SCEN_FAMILIES = ("rename_propagation", "pipe_rewrite", "format_propagation",
                 "doc_sync", "na_rm_propagation")
SCEN_DIR_DEFAULT = Path("/mnt/h/sepalith/datasets/scenarios_v1")
EXAMPLES_DEFAULT = HERE / "examples.jsonl"

# punctuation hidden from the canonical form (operators are kept; N7)
_DELIMS = {"(", ")", "{", "}", "[", "]", ",", ";"}
_LITERAL_NUM = {"float", "integer", "complex"}
_LITERAL_KEPT = {"true", "false", "null", "na", "nan", "inf"}
_SKIP_TYPES = {"comment", "comma"}


def rscript_parses(text):
    """Strict tier: True/False from `Rscript -e parse(text=...)`; None if no Rscript."""
    if not RSCRIPT:
        return None
    esc = text.replace("\\", "\\\\").replace("'", "\\'")
    code = (f"invisible(tryCatch({{parse(text='{esc}'); cat('OK')}},"
            f"error=function(e) cat('ERR')))")
    try:
        out = subprocess.run([RSCRIPT, "-e", code], capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip().endswith("OK")
    except (subprocess.TimeoutExpired, OSError):
        return None


# ------------------------------------------------------------- R canonical --
def _classify_errors(root, src):
    """(parses, notes, tolerated_ids). Tolerated error nodes are dropped."""
    notes, tolerated = [], set()
    stack, genuine = [root], 0
    while stack:
        n = stack.pop()
        if n.is_missing:
            notes.append("unclosed_delim")
            tolerated.add(id(n))
        elif n.type == "ERROR":
            if src[n.start_byte:n.end_byte].decode("utf8", "replace").strip() == ",":
                notes.append("trailing_comma")
                tolerated.add(id(n))
            else:
                genuine += 1
        stack.extend(n.children)
    return genuine == 0, sorted(set(notes)), tolerated


def _identifier_names(root, src):
    """All identifier texts in the tree (arg names included; N3/N4)."""
    names, stack = [], [root]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            names.append(src[n.start_byte:n.end_byte].decode("utf8", "replace"))
        stack.extend(n.children)
    return names


def _alpha_map(names):
    """N3: rename map that is a pure function of the identifier set."""
    return {name: f"v{i}" for i, name in enumerate(sorted(set(names)))}


def _r_ser(node, src, amap, tolerated, comments):
    """Canonical form of one subtree (N1-N7). None => dropped from parent."""
    if node.is_missing or id(node) in tolerated:
        return None
    if node.type in _SKIP_TYPES:
        if node.type == "comment":  # N1: dropped here, kept as text fallback
            comments.append(src[node.start_byte:node.end_byte]
                            .decode("utf8", "replace"))
        return None
    if node.type == "identifier":
        return "ID:" + amap[src[node.start_byte:node.end_byte]
                            .decode("utf8", "replace")]
    if node.type in ("string", "raw_string"):
        return "STR"
    if node.type in _LITERAL_NUM:
        return "NUM"
    if node.type in _LITERAL_KEPT:
        return node.type.upper()

    kids = []
    for c in node.children:  # operators live in the op tag, not in kids
        if c.is_missing or not c.is_named:
            continue
        s = _r_ser(c, src, amap, tolerated, comments)
        if s is not None:
            kids.append(s)

    if node.type in ("arguments",):  # N5: named sorted, positional in order
        named, pos = [], []
        for c in node.children:
            if c.type != "argument" or c.is_missing:
                continue
            ch = c.children
            if (len(ch) >= 3 and not ch[1].is_named and ch[1].type == "="
                    and ch[0].is_named and ch[0].type in ("identifier", "string")):
                name = src[ch[0].start_byte:ch[0].end_byte].decode("utf8", "replace")
                val = _r_ser(ch[2], src, amap, tolerated, comments)
                named.append((name, val))
            else:
                pos.append(_r_ser(c, src, amap, tolerated, comments))
        named.sort(key=lambda t: t[0])
        inner = [f"{n}={v}" for n, v in named] + [p for p in pos if p]
        # keep the subset bracket class: x[i] vs x[[i]] (N7; `[[` is not a
        # plain delim, single `[` is and stays dropped)
        br = [src[c.start_byte:c.end_byte].decode("utf8", "replace")
              for c in node.children
              if not c.is_missing and not c.is_named and
              src[c.start_byte:c.end_byte].decode("utf8", "replace")
              not in _DELIMS]
        tag = "ARGS" + ("|" + "/".join(br) if br else "")
        return f"{tag}({','.join(inner)})"

    ops = []
    for c in node.children:  # N6/N7: operator tokens, delimiters dropped
        if c.is_missing or c.is_named:
            continue
        t = src[c.start_byte:c.end_byte].decode("utf8", "replace")
        if t not in _DELIMS:
            ops.append("ASSIGN" if (node.type == "binary_operator"
                                    and t in ("<-", "=")) else t)
    tag = node.type + ("|" + "/".join(ops) if ops else "")
    return f"{tag}({','.join(kids)})" if kids else tag


def r_canonical(text):
    """-> dict(parses, tool, notes, form, ids, comment_text) or None if empty."""
    if text is None or not text.strip():
        return None
    src = text.encode()
    if HAVE_TS:
        root = R_PARSER.parse(src).root_node
        parses, notes, tolerated = _classify_errors(root, src)
        names = _identifier_names(root, src)
        amap = _alpha_map(names)
        comments = []
        form = _r_ser(root, src, amap, tolerated, comments)
        return dict(parses=parses, tool="treesitter", notes=notes,
                    form=form, ids=Counter(names),
                    comment_text=_norm_comment_text(comments))
    strict = rscript_parses(text)
    return dict(parses=bool(strict), tool="rscript",
                notes=([] if strict else ["strict_parse_fail"]),
                form=None, ids=Counter(), comment_text=None)


def _norm_comment_text(comments):
    """N1 fallback for comment-only regions: strip # decorations, collapse ws."""
    out = []
    for c in comments:
        line = re.sub(r"^[\s]*#+'?\s*", "", c.rstrip())
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------- Python (stdlib) --
class _PyNorm(pyast.NodeTransformer):
    def __init__(self, amap):
        self.amap = amap

    def visit_Name(self, node):
        node.id = self.amap.get(node.id, node.id)
        node.ctx = pyast.Load()
        return node

    def visit_arg(self, node):
        node.arg = self.amap.get(node.arg, node.arg)
        node.ctx = pyast.Load()
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.amap.get(node.name, node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = self.amap.get(node.name, node.name)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):  # N2: value -> class tag
        v = node.value
        tag = ("BOOL" if isinstance(v, bool) else
               "STR" if isinstance(v, str) else
               "BYTES" if isinstance(v, bytes) else
               "NONE" if v is None else "NUM")
        return pyast.Constant(value=tag)

    def visit_Call(self, node):  # N5: keyword args sorted by name
        self.generic_visit(node)
        kw = [k for k in node.keywords if k.arg is not None]
        rest = [k for k in node.keywords if k.arg is None]
        node.keywords = sorted(kw, key=lambda k: k.arg) + rest
        return node

    def generic_visit(self, node):
        for f in ("ctx",):
            if hasattr(node, f):
                setattr(node, f, pyast.Load())
        return super().generic_visit(node)


def _py_names(tree):
    names = [n.id for n in pyast.walk(tree) if isinstance(n, pyast.Name)]
    names += [a.arg for a in pyast.walk(tree) if isinstance(a, pyast.arg)]
    names += [n.name for n in pyast.walk(tree)
              if isinstance(n, (pyast.FunctionDef, pyast.AsyncFunctionDef,
                                pyast.ClassDef))]
    return names


def py_canonical(text):
    """-> dict(parses, tool, notes, form, ids) via the stdlib ast module."""
    if text is None or not text.strip():
        return None
    try:
        tree = pyast.parse(text)
    except SyntaxError:
        return dict(parses=False, tool="pyast", notes=["syntax_error"],
                    form=None, ids=Counter(), comment_text=None)
    amap = _alpha_map(_py_names(tree))
    norm = _PyNorm(amap).visit(tree)
    pyast.fix_missing_locations(norm)
    return dict(parses=True, tool="pyast", notes=[], form=pyast.dump(norm),
                ids=Counter(_py_names(tree)), comment_text=None)


def canonical(text, lang):
    return py_canonical(text) if lang == "python" else r_canonical(text)


# ---------------------------------------------------------------- scoring --
def _norm_lines(lines):
    """The battery's own text normalization (run_eval.norm conventions)."""
    out = [l.rstrip() for l in lines]
    while out and not out[-1]:
        out.pop()
    return out


def score_pair(pred_text, gt_text, lang="r"):
    """ast_equiv of pred vs GT under N1-N7. weak when GT text is missing."""
    out = dict(ast_equiv=None, tree_equal=None, vars_equal=None, weak=False,
               notes=[])
    if gt_text is None:
        out["weak"] = True
        return out
    pc, gc = canonical(pred_text, lang), canonical(gt_text, lang)
    if pc is None:  # empty pred
        out.update(ast_equiv=0, tree_equal=False, notes=["empty_pred"])
        return out
    notes = list(pc["notes"])
    if not gc["parses"]:
        # GT itself is not a complete expression: mid-expression regions
        # (pipe continuations `) |>`, regions starting with `}`, unclosed
        # `{` inside call args where tree-sitter-r recovers to a full
        # ERROR). No structural tier applies; fall back to the battery's
        # own text normalization (rstrip lines, pop trailing blanks) so
        # exact rows stay ast_equiv=1 and the gap metric counts only
        # structurally comparable rows.
        eq_txt = _norm_lines(pred_text.splitlines()) == \
            _norm_lines(gt_text.splitlines())
        out.update(ast_equiv=int(eq_txt), tree_equal=None,
                   notes=notes + ["mid_expression_fragment", "text_fallback"])
        return out
    if not pc["parses"]:  # broken pred: parse-validity carries the verdict
        out.update(ast_equiv=0, tree_equal=False,
                   notes=notes + ["syntax_error"])
        return out
    if pc["form"] is None or gc["form"] is None:
        # no structural tier for this language (tree-sitter absent): only
        # parse-validity is meaningful; ast_equiv stays uncomputed
        out.update(ast_equiv=None, notes=notes + ["no_structural_tier"])
        return out
    # N1 fallback: comment-only regions compare normalized comment text
    if gc["form"] in ("", "program") and (gc["comment_text"] or "") != "":
        if pc["form"] not in ("", "program") or not (pc["comment_text"] or ""):
            out.update(ast_equiv=0, tree_equal=False, notes=["comment_vs_code"])
            return out
        eq = pc["comment_text"] == gc["comment_text"]
        out.update(ast_equiv=int(eq), tree_equal=eq,
                   notes=notes + ["comments_text_only"])
        return out
    tree_eq = bool(pc["parses"]) and pc["form"] == gc["form"]
    vars_eq = pc["ids"] == gc["ids"]  # N4
    out.update(tree_equal=tree_eq, vars_equal=vars_eq,
               ast_equiv=int(tree_eq and vars_eq),
               notes=notes + (["vars_checked"] if vars_eq else ["vars_differ"]))
    return out


def score_row(row, gt_text, pred_cap=400):
    """One stored row -> output row with parses/ast_equiv + pass-through ids."""
    lang = row.get("lang", "r")
    pred = row.get("pred")
    recovered_raw = False
    if row.get("raw") and (pred is None or len(pred) >= pred_cap):
        try:
            sys.path.insert(0, str(HERE))
            from run_eval import parse_pred
            pl = parse_pred("zeta2", row["raw"])
            full = "\n".join(pl) if pl is not None else None
            # Saved previews must agree with the raw rendering before recovery.
            if full is not None and (pred is None or full.startswith(pred)):
                pred = full
                recovered_raw = True
        except Exception:
            pass  # retain the preview and its truncation limitation
    out = {k: row[k] for k in ("id", "family", "i", "package", "path", "note",
                               "model", "lang", "repo", "sha", "exact",
                               "valid_pass") if k in row}
    out["lang"] = lang
    out["truncated"] = bool(not recovered_raw and pred is not None
                            and len(pred) >= pred_cap)
    out["prediction_source"] = "raw_recovered" if recovered_raw else "stored_preview"
    c = canonical(pred, lang)
    if c is None:
        out.update(parses=0, parse_tool="none", ast_equiv=0, weak=False,
                   norm_notes=["empty_pred"])
        return out
    out.update(parses=int(c["parses"]), parse_tool=c["tool"])
    sc = score_pair(pred, gt_text, lang)
    extra_notes = (["truncated_lower_bound"] if out["truncated"] else [])
    out.update(ast_equiv=sc["ast_equiv"], weak=sc["weak"],
               norm_notes=sc["notes"] + extra_notes)
    # fragment row: the GT region itself is not a complete expression, so
    # parse-validity of the pred in isolation is not attributable to the
    # model (counted separately in the aggregate)
    out["fragment"] = "mid_expression_fragment" in sc["notes"]
    # stored pred hit the 400/600-char cap and is a strict PREFIX of the GT:
    # the missing tail makes the comparison meaningless in both directions
    # (exact rows would otherwise show a fake -1 gap) -> uncomputable
    if out["truncated"] and gt_text and pred is not None \
            and gt_text.startswith(pred) and len(gt_text) > len(pred):
        out["ast_equiv"] = None
        out["norm_notes"] = out["norm_notes"] + ["truncated_prefix"]
    if gt_text is not None:
        pc, gc = canonical(pred, lang), canonical(gt_text, lang)
        if pc and gc:
            d = pc["ids"] - gc["ids"]
            e = gc["ids"] - pc["ids"]
            if d or e:
                out["vars_pred_only"] = sorted(d.elements())
                out["vars_gt_only"] = sorted(e.elements())
    return out


# ------------------------------------------------------- GT re-derivation --
def _rstripped(lines):
    return "\n".join(_norm_lines(lines))


def build_scenario_gt(rows, scen_dir):
    """{(family,id): gt_text} by re-rendering the family files with the exact
    training edit_row(); id = sha1(prompt)[:12] exactly as eval_scenarios.py."""
    want = {(r["family"], r["id"]) for r in rows if r.get("family") and r.get("id")}
    if not want:
        return {}
    scen_dir = Path(scen_dir)
    sys.path.insert(0, str(HERE.parent / "post-processing"))
    sys.path.insert(0, str(HERE.parent / "synthetic-data"))
    try:
        from assemble_sft_v2 import edit_row
    except Exception as e:
        print(f"warning: cannot import edit_row ({e}); scenario GT unavailable",
              file=sys.stderr)
        return {}
    got = {}
    for fam in sorted({f for f, _ in want}):
        p = scen_dir / f"{fam}.jsonl"
        if not p.exists():
            print(f"warning: missing {p}", file=sys.stderr)
            continue
        for line in open(p):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            try:
                rr = edit_row(dict(row), fam, row["package"])
            except Exception:
                continue
            if rr is None:
                continue
            rid = hashlib.sha1(rr["prompt"].encode()).hexdigest()[:12]
            if (fam, rid) in want and (fam, rid) not in got:
                got[(fam, rid)] = _rstripped(row["region_new"])
    return got


def _midtyping_region(ex):
    """Deterministic prefix of run_eval.make_midtyping: the changed-lines
    region [first changed line .. last changed line] of the original example."""
    ro, rn = ex["region_old"], ex["region_new"]
    i = 0
    while i < min(len(ro), len(rn)) and ro[i] == rn[i]:
        i += 1
    if i >= len(rn):
        i = max(0, len(rn) - 1)
    last = i
    for j in range(i, len(rn)):
        if rn[j] not in ro:
            last = j
    return rn[i:last + 1], i, last


def align_midtyping(pred_lines, gt_lines):
    """Line-0 alignment for midtyping rows (run_eval.make_midtyping's rng seed
    used str.__hash__, not reproducible). Keeps lines WHOLE (an AST needs full
    lines; cutting mid-token would produce unparseable fragments):
      - pred line 0 == GT line 0 or either empty -> compare as-is
        (`midtyping_full_line`; the raw-aligned pred echoes the typed partial)
      - pred line 0 is a completion SUFFIX of GT line 0 and the implied cut
        falls in make_midtyping's bounds [0.3, 0.6] (slack 0.25-0.75) ->
        reconstruct the full line from GT (`midtyping_cut=N`)
      - otherwise -> `midtyping_unaligned`, compare as-is (honest mismatch).
    Returns (pred_lines, gt_lines, note)."""
    if not gt_lines or not pred_lines:
        return pred_lines, gt_lines, "midtyping_unaligned"
    first, p0 = gt_lines[0], pred_lines[0]
    n = len(first)
    if not first or not p0 or p0 == first:
        return pred_lines, gt_lines, "midtyping_full_line"
    if first.endswith(p0):
        cut = n - len(p0)
        frac = cut / n if n else 1.0
        if 0.25 <= frac <= 0.75:
            return [first] + pred_lines[1:], gt_lines, f"midtyping_cut={cut}"
    return pred_lines, gt_lines, "midtyping_unaligned"


def build_midtyping_gt(rows, examples_path):
    """{(repo,path,sha): (gt_text, align_note)} from examples.jsonl."""
    exs = {}
    for line in open(examples_path):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        exs[(e.get("repo"), e.get("path"), e.get("sha"))] = e
    out = {}
    for r in rows:
        key = (r.get("repo"), r.get("path"), r.get("sha"))
        ex = exs.get(key)
        if ex is None:
            continue
        region, i0, last = _midtyping_region(ex)
        gt_lines = [l.rstrip() for l in region]
        pred_lines = (r.get("pred") or "").splitlines()
        p, g, note = align_midtyping(pred_lines, gt_lines)
        out[key] = ("\n".join(g), note)
    return out


# --------------------------------------------------------------- pipeline --
def detect_pred_cap(rows):
    """Stored truncation caps: 400 (scenario schema), 600 (run_eval schema)."""
    for r in rows:
        if "family" in r:
            return 400
    return 600


def load_rows(path):
    """Stored rows only (skip pretty-printed aggregate blocks)."""
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and ("family" in r or "lang" in r):
            out.append(r)
    return out


def score_file(args):
    rows = load_rows(args.results)
    if args.lang:
        rows = [r for r in rows if r.get("lang", "r") == args.lang]
    cap = detect_pred_cap(rows)
    scenario_schema = any("family" in r for r in rows)

    gt = {}
    if scenario_schema:
        gt = build_scenario_gt(rows, args.scen_dir)
    elif any("repo" in r for r in rows):
        gt = build_midtyping_gt(rows, args.examples)

    out_rows = []
    for r in rows:
        if scenario_schema:
            g = gt.get((r.get("family"), r.get("id")))
        else:
            g, note = gt.get((r.get("repo"), r.get("path"), r.get("sha")),
                             (None, None))
        o = score_row(r, g, cap)
        if not scenario_schema and note:
            o["norm_notes"] = o["norm_notes"] + [note]
        if args.strict and o["lang"] != "python" and o["parse_tool"] != "none":
            strict = rscript_parses(r.get("pred") or "")
            if strict is not None:
                o["parse_strict"] = int(strict)
        out_rows.append(o)

    # ---- aggregate: per family (or lang) counts of exact vs ast_equiv ----
    groups = {}
    for o in out_rows:
        key = o.get("family", f"lang:{o.get('lang', 'r')}")
        groups.setdefault(key, []).append(o)
    agg = {}
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        comp = [o for o in g if o["ast_equiv"] is not None]

        def cnt(pred):
            return sum(1 for o in g if pred(o))
        exact = cnt(lambda o: o.get("exact"))
        aeq = cnt(lambda o: o["ast_equiv"] == 1)
        nfrag = cnt(lambda o: o.get("fragment"))
        ncomplete = n - nfrag
        pok = cnt(lambda o: o["parses"] and not o.get("fragment"))
        agg[key] = dict(
            n=n,
            exact=exact, exact_rate=round(exact / n, 4),
            ast_equiv=aeq, ast_equiv_rate=round(aeq / n, 4),
            gap=round((aeq - exact) / n, 4),
            gap_rows=cnt(lambda o: o["ast_equiv"] == 1 and not o.get("exact")),
            valid_pass=cnt(lambda o: o.get("valid_pass")) or None,
            parses=pok, fragments=nfrag,
            parse_rate_complete=(round(pok / ncomplete, 4) if ncomplete else None),
            ast_computable=len(comp),
            uncomputable=n - len(comp),
            gt_missing=cnt(lambda o: o.get("weak")),
            truncated=cnt(lambda o: o.get("truncated")),
            weak=cnt(lambda o: o.get("weak")),
        )

    out_path = Path(args.out) if args.out else \
        args.results.parent / f"astequiv_{args.results.stem}.jsonl"
    with open(out_path, "w") as f:
        for o in out_rows:
            f.write(json.dumps(o) + "\n")
        f.write(json.dumps(dict(aggregate=dict(
            results=str(args.results), lang=args.lang, pred_cap=cap,
            tiers=dict(treesitter=HAVE_TS, rscript=bool(RSCRIPT),
                       strict_run=bool(args.strict)),
            families=agg, total=len(out_rows)))) + "\n")

    print(f"\n{'group':<24}{'n':>4}{'exact':>10}{'ast_equiv':>12}"
          f"{'gap':>8}{'gap_rows':>9}{'parse*':>9}{'frag':>6}{'trunc':>6}"
          f"{'weak':>6}{'gt?':>6}")
    for key, a in agg.items():
        ncomp = a["n"] - a["fragments"]
        print(f"{key:<24}{a['n']:>4}"
              f"{str(a['exact']) + '/' + str(a['n']):>10}"
              f"{str(a['ast_equiv']) + '/' + str(a['n']):>12}"
              f"{a['gap']:>8.1%}{a['gap_rows']:>9}"
              f"{str(a['parses']) + '/' + str(ncomp):>9}"
              f"{a['fragments']:>6}{a['truncated']:>6}{a['weak']:>6}"
              f"{a['uncomputable']:>6}")
    print("\nparse* = parses=1 among complete-expression rows (frag = rows "
          "whose GT region itself is mid-expression; those use the\ntext "
          "fallback for ast_equiv and their parse-validity is not "
          "attributable to the model)")
    print(f"\ntiers on this box: tree-sitter-r={HAVE_TS}  rscript={bool(RSCRIPT)}"
          f"  python=ast-module\nwrote {out_path}")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--lang", default="r",
                    help="filter by row lang (r|python|'' for all)")
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="also record Rscript strict-tier parse per R row")
    ap.add_argument("--scen-dir", type=Path, default=SCEN_DIR_DEFAULT)
    ap.add_argument("--examples", type=Path, default=EXAMPLES_DEFAULT)
    args = ap.parse_args()
    if args.lang == "''" or args.lang.lower() in ("all", "none"):
        args.lang = ""
    if not args.results.exists():
        sys.exit(f"missing: {args.results}")
    score_file(args)


if __name__ == "__main__":
    main()
