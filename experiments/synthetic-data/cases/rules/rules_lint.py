"""rules_lint.py — lintr-catalog rewrite rules (registry wave 1).

Six D2 rewrites, each (detector, rewrite, verify) over BaseSample, each
backed by SELFTEST snippets (run `python3 cases/rules/run_rules.py
--selftest`). Sources: lintr's default rule catalog (T_and_F_symbol,
seq_linters, paste, single_quotes) + R modernization lore (class()==
inherits, if(!x) stop() -> stopifnot named form, R >= 4.0).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

import scenarios as S                       # noqa: E402
from scenarios import node_text             # noqa: E402
import cases.validators as V                # noqa: E402
from cases.rules import Rewrite, Site, rule  # noqa: E402

_MAX_LINE = 220


def _txt(src, n) -> str:
    return node_text(src, n).decode("utf-8", "replace")


def _one_row(n) -> bool:
    return n.start_point[0] == n.end_point[0]


def _geom_ok(bs, row: int) -> bool:
    """Row-shape gate for rewrite sites: strictly inside the braced body,
    a non-blank non-comment site line, and at least one non-blank statement
    BELOW inside the body (the suffix must carry the function remainder).
    Unlike mid_body_edit, the FIRST body statement qualifies — a guard
    clause / input check is canonically the first statement."""
    b = bs.b
    if not bs.r0 < row < bs.r1:
        return False
    if not any(b.line_str(r).strip() for r in range(row + 1, bs.r1)):
        return False
    line = b.line_str(row)
    return bool(line.strip()) and not line.lstrip().startswith("#")


def _balanced_arg(text: str, open_idx: int) -> str | None:
    """Argument text of the paren at open_idx, paren-balanced and string-
    aware; None when unbalanced."""
    depth, q, i = 0, None, open_idx
    while i < len(text):
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
        i += 1
    return None


# ---------------------------------------------------------------------------
# 1. tf_true_false — T/F used as logical constants -> TRUE/FALSE
# ---------------------------------------------------------------------------

_TF_WORD = r"(?<![\w.$@])(T|F)(?![\w.$@])"
_TF_RE = re.compile(_TF_WORD.encode())
_TF_ASSIGNED = re.compile(rb"(?<![\w.$@])(T|F)\s*(?:=|<-|<<-)")


def _tf_ok_position(src, n) -> bool:
    """T/F in genuine value position: not callee, not an argument name, not
    part of :: / $ chains, not an assignment target, not a for-loop variable."""
    p = n.parent
    if p is None:
        return False
    if p.type in ("namespace_operator", "extract_operator"):
        return False
    if p.type == "argument":
        kids = list(p.children)
        if any(c.type == "=" for c in kids) and kids and \
                _txt(src, kids[0]) == _txt(src, n):
            return False                       # named-argument slot
    if S.parent_is_caller(n):
        return False                           # callee position
    if p.type == "binary_operator" and p.children and \
            _txt(src, p.children[1]) in ("<-", "=", "<<-") and \
            _same_span(p.children[0], n):
        return False                           # assignment target
    if p.type == "for_statement":
        return False                           # loop variable
    return True


def _same_span(a, b) -> bool:
    return a.type == b.type and a.start_byte == b.start_byte \
        and a.end_byte == b.end_byte


@rule(id="tf_true_false", family="lint_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="identifier 'T'/'F' in value position (tree-sitter walk; "
             "lintr T_and_F_symbol)",
      restraint="suppressed when T/F is assigned anywhere in the file "
                "(shadowed symbol) or sits in callee/arg-name/assign/for-var "
                "position — the naive rewrite would change program meaning",
      status="new")
class TfTrueFalse:
    PRESCREEN = [rb"(?<![\w.$@])[TF](?![\w.$@])"]
    SELFTEST = [
        (b'f <- function(x) {\n  a <- T\n  b <- x || F\n  a && b\n}\n',
         dict(expect_sites=2)),
        (b'f <- function(x) {\n  T <- TRUE\n  a <- T\n}\n',
         dict(expect_sites=0, why="T shadowed by local assignment")),
        (b'f <- function(df) {\n  z <- df$T\n  y <- T(df)\n}\n',
         dict(expect_sites=0, why="$-slot and callee positions only")),
    ]

    def detector(self, bs):
        src = bs.b.src
        if _TF_ASSIGNED.search(S.strip_strings(src)):
            return []                          # file shadows T/F: restraint
        out = []
        for n in V._walk(bs.body):
            if n.type != "identifier" or _txt(src, n) not in ("T", "F"):
                continue
            if not _one_row(n) or not _tf_ok_position(src, n):
                continue
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            new = "TRUE" if _txt(src, n) == "T" else "FALSE"
            out.append(Site(row=row, sb=n.start_byte, eb=n.end_byte,
                            payload=dict(old=_txt(src, n), new=new),
                            note=f"{_txt(src, n)} -> {new} "
                                 f"(T/F are shadowable variables, not "
                                 f"constants)"))
        return out

    def rewrite(self, bs, site):
        b = bs.b
        lb = b.line_bytes(site.row)
        col = site.sb - b.starts[site.row]
        new_lb = lb[:col] + site.payload["new"].encode() \
            + lb[col + len(site.payload["old"].encode()):]
        new_line = new_lb.decode("utf-8", "replace").rstrip("\r")
        if not new_line.strip() or len(new_line) > _MAX_LINE:
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_tok=site.payload["old"],
                                 new_tok=site.payload["new"]),
                       span_text=site.payload["new"])

    def verify(self, old_text, new_text):
        # multi-site safe: exactly ONE T/F token disappears, one
        # TRUE/FALSE constant appears, nothing else changes
        old_tf = len(_TF_RE.findall(old_text.encode()))
        new_tf = len(_TF_RE.findall(new_text.encode()))
        if new_tf != old_tf - 1:
            return False, f"expected exactly one T/F token swapped " \
                          f"({old_tf} -> {new_tf})"
        grew = (new_text.count("TRUE") + new_text.count("FALSE")
                - old_text.count("TRUE") - old_text.count("FALSE"))
        if grew < 1:
            return False, "no TRUE/FALSE constant appeared"
        return True, ""


# ---------------------------------------------------------------------------
# 2. seq_along_replace — 1:length(x) -> seq_along(x); 1:nrow(df) -> seq_len()
# ---------------------------------------------------------------------------

_SEQ_CALLEES = {"length": "seq_along", "nrow": "seq_len", "ncol": "seq_len",
                "NROW": "seq_len", "NCOL": "seq_len"}


@rule(id="seq_along_replace", family="lint_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="binary `:` with integer-literal 1 LHS and length/nrow/ncol RHS "
             "(tree-sitter; lintr seq_linters)",
      restraint="only literal-1 LHS: 2:length(x) and 1:compute() have no "
                "exact seq_* equivalent; the rewrite also FIXES the length-0 "
                "bug (1:0 -> c(1,0)) — that is the point of the lint",
      status="new")
class SeqAlongReplace:
    PRESCREEN = [rb"(?<![\w.\d])1\s*:\s*(?:length|nrow|ncol|NROW|NCOL)\s*\("]
    SELFTEST = [
        (b'f <- function(x) {\n  for (i in 1:length(x)) y[i] <- x[i]\n  y\n}\n',
         dict(expect_sites=1, first_new="  for (i in seq_along(x)) y[i] <- x[i]")),
        (b'f <- function(df) {\n  for (j in 1:nrow(df)) s <- s + df$v[j]\n  s\n}\n',
         dict(expect_sites=1,
              first_new="  for (j in seq_len(nrow(df))) s <- s + df$v[j]")),
        (b'f <- function(x) {\n  k <- 2:length(x)\n  z <- 1:10\n  z\n}\n',
         dict(expect_sites=0, why="2: and literal rhs are not seq sites")),
    ]

    def detector(self, bs):
        src = bs.b.src
        out = []
        for n in V._walk(bs.body):
            if n.type != "binary_operator" or len(n.children) < 3 or \
                    _txt(src, n.children[1]) != ":":
                continue
            lhs, rhs = n.children[0], n.children[2]
            if lhs.type != "float" or _txt(src, lhs) != "1" or rhs.type != "call":
                continue
            callee = S.callee_name(src, rhs)
            if callee not in _SEQ_CALLEES or not _one_row(n):
                continue
            args = next((c for c in rhs.children if c.type == "arguments"),
                        None)
            argv = [V._argument_value(a) for a in
                    (args.children if args is not None else [])
                    if a.type == "argument"]
            if len(argv) != 1 or argv[0] is None:
                continue
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            out.append(Site(row=row, sb=n.start_byte, eb=n.end_byte,
                            payload=dict(callee=callee,
                                         arg=_txt(src, argv[0]),
                                         repl=_SEQ_CALLEES[callee]),
                            note=f"1:{callee}(...) -> "
                                 f"{_SEQ_CALLEES[callee]} "
                                 f"(empty-sequence safe)"))
        return out

    def rewrite(self, bs, site):
        p = site.payload
        if p["callee"] == "length":
            new_expr = f"seq_along({p['arg']})"
        else:
            new_expr = f"seq_len({p['callee']}({p['arg']}))"
        b = bs.b
        lb = b.line_bytes(site.row)
        col0 = site.sb - b.starts[site.row]
        col1 = site.eb - b.starts[site.row]
        new_line = (lb[:col0] + new_expr.encode() + lb[col1:]) \
            .decode("utf-8", "replace").rstrip("\r")
        if not new_line.strip() or len(new_line) > _MAX_LINE:
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_expr=f"1:{p['callee']}(...)",
                                 new_expr=new_expr.strip()),
                       span_text=new_expr)

    def verify(self, old_text, new_text):
        m = re.search(r"(?<![\w.\d])1\s*:\s*(length|nrow|ncol|NROW|NCOL)\s*\(",
                      old_text)
        if not m:
            return False, "old text has no 1:<len-fn>( site"
        arg = _balanced_arg(old_text, m.end() - 1)
        if arg is None:
            return False, "unbalanced call in old text"
        arg = arg.strip()
        if m.group(1) == "length":
            expect = f"seq_along({arg})"
        else:
            expect = f"seq_len({m.group(1)}({arg}))"
        if expect not in new_text:
            return False, f"expected {expect} in the rewrite"
        if re.search(r"(?<![\w.\d])1\s*:\s*(length|nrow|ncol)\s*\(", new_text):
            return False, "rewrite left a 1:length site behind"
        return True, ""


# ---------------------------------------------------------------------------
# 3. paste_sep_empty — paste(x, sep="") -> paste0(x); paste0(x, sep="") drop
# ---------------------------------------------------------------------------

def _drop_sep_arg(call_text: str) -> str | None:
    """Remove a `sep = \"\"` argument from a call's text (both comma sides)."""
    t = re.sub(r",\s*sep\s*=\s*\"\"", "", call_text, count=1)
    if t == call_text:
        t = re.sub(r"sep\s*=\s*\"\"\s*,", "", call_text, count=1)
    if t == call_text:
        return None
    return re.sub(r",\s*,", ",", t)


@rule(id="paste_sep_empty", family="lint_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="paste/paste0 call carrying a literal `sep = \"\"` argument "
             "(tree-sitter argument nodes; lintr paste linter)",
      restraint="literal sep=\"\" only — any other sep expression (variable, "
                "\"\\\\t\") is not provably paste0-equivalent; collapse= is "
                "preserved verbatim (paste0 honours it identically)",
      status="new")
class PasteSepEmpty:
    PRESCREEN = [rb"(?<![\w.])paste0?\s*\(", rb"sep\s*=\s*\"\""]
    SELFTEST = [
        (b'f <- function(x) {\n  s <- paste(x, sep = "")\n  s\n}\n',
         dict(expect_sites=1, first_new='  s <- paste0(x)')),
        (b'f <- function(x) {\n  s <- paste(x, collapse = ", ", sep = "")\n'
         b'  s\n}\n',
         dict(expect_sites=1,
              first_new='  s <- paste0(x, collapse = ", ")')),
        (b'f <- function(x) {\n  s <- paste0("a", b, sep = "")\n  s\n}\n',
         dict(expect_sites=1, first_new='  s <- paste0("a", b)')),
        (b'f <- function(x) {\n  s <- paste(x, y, sep = "-")\n  s\n}\n',
         dict(expect_sites=0, why='sep is a real separator, not the "" case')),
    ]

    def detector(self, bs):
        src = bs.b.src
        out = []
        for n in V._walk(bs.body):
            if n.type != "call" or not _one_row(n):
                continue
            callee = S.callee_name(src, n)
            if callee not in ("paste", "paste0"):
                continue
            args = next((c for c in n.children if c.type == "arguments"), None)
            hit = False
            for a in (args.children if args is not None else []):
                if a.type != "argument":
                    continue
                kids = list(a.children)
                if not any(c.type == "=" for c in kids):
                    continue
                name = kids[0] if kids and kids[0].type == "identifier" else None
                val = V._argument_value(a)
                if name is not None and _txt(src, name) == "sep" and \
                        val is not None and val.type == "string" and \
                        _txt(src, val) == '""':
                    hit = True
            if not hit:
                continue
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            if len(bs.b.line_str(row)) > _MAX_LINE:
                continue
            out.append(Site(row=row, sb=n.start_byte, eb=n.end_byte,
                            payload=dict(callee=callee),
                            note=f'{callee} with sep = "" is paste0'))
        return out

    def rewrite(self, bs, site):
        b = bs.b
        line = b.line_str(site.row)
        col0, col1 = site.sb - b.starts[site.row], site.eb - b.starts[site.row]
        call_text = line[col0:col1]
        dropped = _drop_sep_arg(call_text)
        if dropped is None:
            return None
        if site.payload["callee"] == "paste":
            if not dropped.startswith("paste("):
                return None
            dropped = "paste0" + dropped[len("paste"):]
        new_line = line[:col0] + dropped + line[col1:]
        new_line = new_line.rstrip("\r")
        if not new_line.strip() or len(new_line) > _MAX_LINE:
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_call=call_text, new_call=dropped.strip()),
                       span_text=dropped)

    def verify(self, old_text, new_text):
        if not re.search(r"(?<![\w.])paste0?\s*\(", old_text):
            return False, "old text has no paste/paste0 call"
        if not re.search(r"sep\s*=\s*\"\"", old_text):
            return False, "old call carries no literal sep = \"\""
        if re.search(r"sep\s*=\s*\"", new_text):
            return False, "rewrite kept a sep argument"
        if not re.search(r"(?<![\w.])paste0\s*\(", new_text):
            return False, "rewrite is not a paste0 call"
        if re.search(r"(?<![\w.])paste\s*\(", new_text):
            return False, "a bare paste( call survived"
        return True, ""


# ---------------------------------------------------------------------------
# 4. class_inherits — class(x) == "foo"  ->  inherits(x, "foo")
# ---------------------------------------------------------------------------

_CLASS_EQ_RE = re.compile(
    r"class\s*\((?P<arg>[^()]*(?:\([^()]*\)[^()]*)*)\)\s*(?P<op>==|!=)\s*"
    r"(?P<str>\"(?:[^\"\\]|\\.)*\")")


@rule(id="class_inherits", family="lint_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="binary ==/!= whose LHS is a 1-arg class() call and RHS a single "
             "string literal (tree-sitter; lintr class_equals)",
      restraint="single string RHS only: class(x) == c(\"a\",\"b\") compares "
                "the whole class vector elementwise and has no inherits "
                "twin; documented semantic caveat — class(x)==\"foo\" is "
                "exact-class, inherits() is chain-membership (the lintr-"
                "endorsed idiom switch)",
      status="new")
class ClassInherits:
    PRESCREEN = [rb"class\s*\([^)\n]{0,80}\)\s*(?:==|!=)"]
    SELFTEST = [
        (b'f <- function(x) {\n  if (class(x) == "lm") summary(x)\n  NULL\n}\n',
         dict(expect_sites=1,
              first_new='  if (inherits(x, "lm")) summary(x)')),
        (b'f <- function(x) {\n  ok <- class(x) != "data.frame"\n  ok\n}\n',
         dict(expect_sites=1,
              first_new='  ok <- !inherits(x, "data.frame")')),
        (b'f <- function(x) {\n  ok <- class(x) == c("a", "b")\n  ok\n}\n',
         dict(expect_sites=0, why="vector RHS: no exact inherits twin")),
    ]

    def detector(self, bs):
        src = bs.b.src
        out = []
        for n in V._walk(bs.body):
            if n.type != "binary_operator" or len(n.children) < 3:
                continue
            lhs, op, rhs = n.children[0], n.children[1], n.children[2]
            if _txt(src, op) not in ("==", "!="):
                continue
            if lhs.type != "call" or S.callee_name(src, lhs) != "class":
                continue
            if rhs.type != "string" or not _one_row(n):
                continue
            args = next((c for c in lhs.children if c.type == "arguments"),
                        None)
            argv = [V._argument_value(a) for a in
                    (args.children if args is not None else [])
                    if a.type == "argument"]
            if len(argv) != 1 or argv[0] is None:
                continue
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            out.append(Site(row=row, sb=n.start_byte, eb=n.end_byte,
                            payload=dict(arg=_txt(src, argv[0]),
                                         s=_txt(src, rhs),
                                         neg=_txt(src, op) == "!="),
                            note='class(...) == "..." -> inherits(...)'))
        return out

    def rewrite(self, bs, site):
        p = site.payload
        expr = f'inherits({p["arg"]}, {p["s"]})'
        if p["neg"]:
            expr = "!" + expr
        b = bs.b
        line = b.line_str(site.row)
        col0, col1 = site.sb - b.starts[site.row], site.eb - b.starts[site.row]
        new_line = (line[:col0] + expr + line[col1:]).rstrip("\r")
        if not new_line.strip() or len(new_line) > _MAX_LINE:
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_expr=f'class({p["arg"]})',
                                 new_expr=expr),
                       span_text=expr)

    def verify(self, old_text, new_text):
        m = _CLASS_EQ_RE.search(old_text)
        if not m:
            return False, "old text has no class(...) == \"...\" site"
        expect = f'inherits({m.group("arg").strip()}, {m.group("str")})'
        if m.group("op") == "!=":
            expect = "!" + expect
        if expect not in new_text:
            return False, f"expected {expect} in the rewrite"
        if re.search(r"class\s*\([^()]*\)\s*(==|!=)", new_text):
            return False, "rewrite left a class() comparison behind"
        return True, ""


# ---------------------------------------------------------------------------
# 5. stopifnot_named — if (!cond) stop("msg") -> stopifnot("msg" = cond)
# ---------------------------------------------------------------------------

@rule(id="stopifnot_named", family="lint_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="if_statement whose condition is a unary ! and whose body is a "
             "single stop(\"literal\") call (tree-sitter; R >= 4.0 named-"
             "condition form)",
      restraint="single string-literal stop() message only — stop(..., "
                "call.=FALSE)/domain=/conditionMessage variants and stop() "
                "with no message keep their if-form (or rewrite to the "
                "UNNAMED stopifnot(cond), which the detector also allows)",
      status="new")
class StopifnotNamed:
    PRESCREEN = [rb"if\s*\(\s*!", rb"(?<![\w.])stop\s*\("]
    SELFTEST = [
        (b'f <- function(x) {\n  if (!is.numeric(x)) stop("x must be numeric")\n'
         b'  x + 1\n}\n',
         dict(expect_sites=1,
              first_new='  stopifnot("x must be numeric" = is.numeric(x))')),
        (b'f <- function(x) {\n  if (!is.numeric(x)) {\n'
         b'    stop("x must be numeric")\n  }\n  x + 1\n}\n',
         dict(expect_sites=1,
              first_new='  stopifnot("x must be numeric" = is.numeric(x))')),
        (b'f <- function(x) {\n  if (!is.numeric(x)) stop("bad", call. = FALSE)\n'
         b'  x\n}\n',
         dict(expect_sites=0, why="stop carries extra arguments")),
        (b'f <- function(x) {\n  if (is.numeric(x)) x else stop("no")\n  x\n}\n',
         dict(expect_sites=0, why="positive condition: no ! to strip")),
    ]

    def detector(self, bs):
        src = bs.b.src
        out = []
        for n in V._walk(bs.body):
            if n.type != "if_statement":
                continue
            if n.end_point[0] - n.start_point[0] > 2:
                continue                    # keep sites <= 3 rows
            kids = [c for c in n.children if c.is_named]
            if len(kids) < 2:
                continue
            cond = kids[0]
            if cond.type == "parenthesized_expression" and cond.children:
                inner = [c for c in cond.children if c.is_named]
                cond = inner[0] if len(inner) == 1 else cond
            if cond.type != "unary_operator":
                continue
            ukids = [c for c in cond.children if c.is_named]
            if not ukids or _txt(src, cond.children[0]) != "!":
                continue
            operand = ukids[0]
            body = kids[1]
            stmts = [c for c in body.children if c.is_named and
                     c.type != "comment"] if body.type == "braced_expression" \
                else [body]
            if len(stmts) != 1 or stmts[0].type != "call" or \
                    S.callee_name(src, stmts[0]) != "stop":
                continue
            args = next((c for c in stmts[0].children if c.type == "arguments"),
                        None)
            argv = [V._argument_value(a) for a in
                    (args.children if args is not None else [])
                    if a.type == "argument"]
            named = [a for a in (args.children if args is not None else [])
                     if a.type == "argument" and
                     any(c.type == "=" for c in a.children)]
            if named or len(argv) > 1:
                continue
            msg = None
            if argv and argv[0].type == "string":
                msg = _txt(src, argv[0])
            elif argv:
                continue                    # non-literal message: restraint
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            out.append(Site(row=row, sb=n.start_byte, eb=n.end_byte,
                            payload=dict(cond=_txt(src, operand), msg=msg,
                                         n_rows=n.end_point[0] - row + 1),
                            row_end=n.end_point[0],
                            note='if (!cond) stop("msg") -> '
                                 'stopifnot("msg" = cond) (R >= 4.0)'))
        return out

    def rewrite(self, bs, site):
        p = site.payload
        if p["msg"] is not None:
            new_line = f'stopifnot({p["msg"]} = {p["cond"]})'
        else:
            new_line = f'stopifnot({p["cond"]})'
        b = bs.b
        old_lines = bs.lines(site.row, site.row + p["n_rows"])
        indent = old_lines[0][:len(old_lines[0]) - len(old_lines[0].lstrip())]
        new_line = indent + new_line
        if len(new_line) > _MAX_LINE:
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_text="\n".join(old_lines),
                                 corpus_line="\n".join(old_lines)),
                       span_text=new_line[len(indent):])

    def verify(self, old_text, new_text):
        flat = " ".join(l.strip() for l in old_text.split("\n") if l.strip())
        m = re.match(r"^if\s*\(\s*!\s*(?P<cond>.+?)\s*\)\s*"
                     r"(?:stop\s*\(\s*(?P<msg>\"(?:[^\"\\]|\\.)*\")?\s*\)\s*;?"
                     r"|\{\s*stop\s*\(\s*(?P<msg2>\"(?:[^\"\\]|\\.)*\")?\s*\)\s*;?\s*\})$",
                     flat)
        if not m:
            return False, "old text is not an if (!cond) stop([msg]) guard"
        cond = m.group("cond").strip()
        if cond.startswith("(") and cond.endswith(")"):
            inner = _balanced_arg(cond, 0)
            if inner is not None and _balanced_arg("(" + inner + ")", 0) == inner:
                cond = inner                # drop one redundant paren layer
        msg = m.group("msg") or m.group("msg2")
        expect = f'stopifnot({msg} = {cond})' if msg else f'stopifnot({cond})'
        if new_text.strip() != expect:
            return False, f"expected {expect}"
        return True, ""


# ---------------------------------------------------------------------------
# 6. single_quote — 'lit' -> "lit" (the CONTRIBUTING worked example)
# ---------------------------------------------------------------------------

_SQ_RE = re.compile(r"'(?P<body>[^'\"\\]+)'")


@rule(id="single_quote", family="lint_rewrite", determinism="D2",
      kind="rewrite", requires=["fn_body"],
      signal="string literal nodes whose text starts with ' (tree-sitter; "
             "lintr single_quotes, a default linter)",
      restraint="plain bodies only: strings containing \", escapes, or "
                "embedded ' are skipped (quote conversion needs re-escaping; "
                "a follow-up rule can own that case)",
      status="new")
class SingleQuote:
    PRESCREEN = [rb"(?<![\w)])'[^'\n\"]{1,80}'"]
    SELFTEST = [
        (b'f <- function(x) {\n  msg <- \'hello world\'\n  message(msg)\n}\n',
         dict(expect_sites=1, first_new='  msg <- "hello world"')),
        (b'f <- function(x) {\n  p <- \'it\\\'s\'\n  q <- "plain"\n  p\n}\n',
         dict(expect_sites=0, why="escaped quote inside: restraint case")),
        (b'f <- function(x) {\n  a <- \'has \\"dq\\" inside\'\n  a\n}\n',
         dict(expect_sites=0, why="double quote inside: restraint case")),
    ]

    def detector(self, bs):
        src = bs.b.src
        out = []
        for n in V._walk(bs.body):
            if n.type != "string" or not _one_row(n):
                continue
            t = _txt(src, n)
            if not (t.startswith("'") and t.endswith("'") and len(t) >= 2):
                continue
            body = t[1:-1]
            if '"' in body or "\\" in body:
                continue                    # restraint: re-escaping needed
            row = n.start_point[0]
            if not _geom_ok(bs, row):
                continue
            out.append(Site(row=row, sb=n.start_byte, eb=n.end_byte,
                            payload=dict(old=t, new=f'"{body}"'),
                            note="'single-quoted' -> \"double-quoted\" "
                                 "(tidyverse style)"))
        return out

    def rewrite(self, bs, site):
        b = bs.b
        lb = b.line_bytes(site.row)
        col = site.sb - b.starts[site.row]
        new_lb = lb[:col] + site.payload["new"].encode() \
            + lb[col + len(site.payload["old"].encode()):]
        new_line = new_lb.decode("utf-8", "replace").rstrip("\r")
        if not new_line.strip() or len(new_line) > _MAX_LINE:
            return None
        return Rewrite(lines=[new_line],
                       meta=dict(old_tok=site.payload["old"],
                                 new_tok=site.payload["new"]),
                       span_text=site.payload["new"])

    def verify(self, old_text, new_text):
        # multi-site safe: converting ANY one plain '...' to "..." must
        # yield the target (the site may be the 2nd string on the line)
        for m in _SQ_RE.finditer(old_text):
            expect = old_text[:m.start()] + f'"{m.group("body")}"' \
                + old_text[m.end():]
            if new_text == expect:
                return True, ""
        return False, "not an exact single-quote -> double-quote conversion"
