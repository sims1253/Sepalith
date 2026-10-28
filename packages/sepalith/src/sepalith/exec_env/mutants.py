"""Mutant generator for exec-env test gating (R parse data, token-level edits).

Mutants change only the target function (the last top-level expression of a
unit file), one edit each. Operator classes:

- ``relational``: ``<`` <-> ``>=``, ``>`` <-> ``<=``, ``==`` <-> ``!=``
- ``arithmetic``: ``+`` <-> ``-``, ``*`` <-> ``/``, ``^`` -> ``*``, ``%%`` -> ``%/%``
- ``logical``: ``&&`` <-> ``||``, ``&`` <-> ``|``, drop a ``!``
- ``constant``: integers n -> n + 1, other numbers x -> 2x, ``TRUE`` <-> ``FALSE``, strings get a suffix
- ``na_handling``: ``is.na(`` / ``complete.cases(`` / ``anyNA(`` negated (``na.rm`` flips are constants)
- ``branch``: an ``if`` condition replaced by ``TRUE`` / ``FALSE``
- ``statement_delete``: one whole-line statement of any ``{`` block; ``statement_duplicate``: the same
  for self-updating assignments (``x <- f(x)``), since other duplicates are mostly equivalent
- ``swap_args``: the first two positional arguments of a call (single-line arguments)
- ``indexing``: a single-symbol or numeric subscript ``x[i]`` -> ``x[i + 1]``
- ``early_return``: ``return(NULL)`` (and ``return(<first formal>)``) inserted before a body statement

Each mutant records the 1-based line span it touches, so a per-cut gate can
select the mutants local to a cut. Mutants that are textually identical after
removing whitespace are dropped here; ones that do not parse or source are
dropped by the gate (they are invalid, not killed). ``sample`` picks a
deterministic, class-stratified subset.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile

from sepalith.data.synth.common import rng_for

HERE = Path(__file__).resolve().parent
R_TOKENS = HERE / "r_tokens.R"

REL = {"<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "=="}
ARITH = {"+": "-", "-": "+", "*": "/", "/": "*", "^": "*", "%%": "%/%", "%/%": "%%"}
LOGIC = {"&&": "||", "||": "&&", "&": "|", "|": "&"}
NA_FUNS = {"is.na", "complete.cases", "anyNA", "is.null", "is.finite", "is.nan"}
SYMMETRIC = {"max", "min", "sum", "prod", "pmax", "pmin", "all", "any", "identical", "range", "isTRUE", "xor",
             "all.equal", "setequal", "intersect", "union", "rbind", "cbind"}
# calls whose string and logical arguments are message text or message options (`call. = FALSE`): their
# mutants only change wording or printing (tests pin a distinctive word at most), so they are not generated
MESSAGE_CALLS = {"stop", "warning", "message", "cat", "print", "simpleError", "simpleWarning", "simpleMessage",
                 "warningCondition", "errorCondition", "packageStartupMessage", "signalCondition", "stopifnot",
                 ".handleSimpleError", "gettextf", "ngettext"}
STMT_TOKENS = {"expr", "equal_assign", "expr_or_assign_or_help"}


@dataclass(frozen=True)
class Token:
    id: int
    parent: int
    token: str
    terminal: bool
    line1: int
    col1: int
    line2: int
    col2: int


@dataclass(frozen=True)
class Mutant:
    id: str
    op: str
    line1: int  # 1-based span in the original unit touched by the edit
    line2: int
    description: str
    text: str
    original: tuple[str, ...] = ()
    replacement: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"id": self.id, "op": self.op, "lines": [self.line1, self.line2], "description": self.description}


def parse_tokens(texts: Sequence[str]) -> list[list[Token] | None]:
    """Parse data per text (``None`` if it does not parse)."""
    with tempfile.TemporaryDirectory(prefix="sepalith-mut-") as tmp:
        paths = []
        for index, text in enumerate(texts):
            path = Path(tmp) / f"{index}.R"
            path.write_text(text, encoding="utf-8")
            paths.append(str(path))
        out_path = Path(tmp) / "tokens.tsv"
        subprocess.run(["Rscript", "--vanilla", str(R_TOKENS), str(out_path), *paths], check=True,
                       capture_output=True, timeout=600)
        tables: list[list[Token] | None] = [[] for _ in texts]
        for line in out_path.read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            index = int(parts[0])
            if parts[1] == "ERROR":
                tables[index] = None
                continue
            table = tables[index]
            assert table is not None
            table.append(Token(int(parts[1]), int(parts[2]), parts[3], parts[4] == "1", int(parts[5]), int(parts[6]),
                               int(parts[7]), int(parts[8])))
    return tables


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


class _Builder:
    def __init__(self, lines: list[str], target_start: int) -> None:
        self.lines = lines
        self.target_start = target_start
        self.out: dict[str, Mutant] = {}
        self.seen = {_squash("\n".join(lines))}

    def add(self, op: str, line1: int, line2: int, new_lines: list[str], description: str) -> None:
        if line1 < self.target_start:
            return
        mutated = [*self.lines[:line1 - 1], *new_lines, *self.lines[line2:]]
        text = "\n".join(mutated)
        key = _squash(text)
        if key in self.seen:
            return
        self.seen.add(key)
        # the id binds the edit itself (span and replacement text), so a declared equivalent can never
        # silently point at a different mutant of the same line
        mid = hashlib.sha256(f"{op}\0{line1}\0{line2}\0{chr(10).join(new_lines)}".encode()).hexdigest()[:10]
        self.out[mid] = Mutant(mid, op, line1, line2, description, text if text.endswith("\n") else text + "\n",
                               tuple(self.lines[line1 - 1:line2]), tuple(new_lines))

    def replace_span(self, op: str, tok: Token, replacement: str, description: str) -> None:
        if tok.line1 != tok.line2:
            return
        line = self.lines[tok.line1 - 1]
        new = line[:tok.col1 - 1] + replacement + line[tok.col2:]
        self.add(op, tok.line1, tok.line1, [new], description)

    def text_of(self, tok: Token) -> str:
        if tok.line1 == tok.line2:
            return self.lines[tok.line1 - 1][tok.col1 - 1:tok.col2]
        parts = [self.lines[tok.line1 - 1][tok.col1 - 1:]]
        parts += self.lines[tok.line1:tok.line2 - 1]
        parts.append(self.lines[tok.line2 - 1][:tok.col2])
        return "\n".join(parts)


def generate(unit: str, tokens: list[Token]) -> list[Mutant]:
    """All first-order mutants of the target (last top-level) function of ``unit``."""
    lines = unit.rstrip("\n").split("\n")
    by_id = {t.id: t for t in tokens}
    children: dict[int, list[Token]] = defaultdict(list)
    for tok in tokens:
        children[tok.parent].append(tok)
    for kids in children.values():
        kids.sort(key=lambda t: (t.line1, t.col1, -t.line2, -t.col2))
    tops = [t for t in tokens if t.parent == 0 and not t.terminal] or [t for t in tokens if t.parent <= 0]
    target = max(tops, key=lambda t: (t.line1, t.col1))
    b = _Builder(lines, target.line1)
    inside = [t for t in tokens if t.line1 >= target.line1 and t.line2 <= target.line2]
    formals = [t for t in inside if t.token == "SYMBOL_FORMALS"]
    for tok in inside:
        if not tok.terminal:
            continue
        text = b.text_of(tok)
        if tok.token in ("LT", "GT", "LE", "GE", "EQ", "NE") and text in REL:
            b.replace_span("relational", tok, REL[text], f"{text} -> {REL[text]}")
        elif text in ARITH and tok.token in ("'+'", "'-'", "'*'", "'/'", "'^'", "SPECIAL"):
            b.replace_span("arithmetic", tok, ARITH[text], f"{text} -> {ARITH[text]}")
        elif tok.token in ("AND2", "OR2", "AND", "OR") and text in LOGIC:
            b.replace_span("logical", tok, LOGIC[text], f"{text} -> {LOGIC[text]}")
        elif tok.token == "'!'":
            b.replace_span("logical", tok, "", "drop !")
        elif tok.token == "NUM_CONST":
            if text in ("TRUE", "FALSE") and _in_message_call(tok, by_id, children, b):
                continue
            if text in ("TRUE", "FALSE"):
                flip = "FALSE" if text == "TRUE" else "TRUE"
                b.replace_span("constant", tok, flip, f"{text} -> {flip}")
            elif re.fullmatch(r"\d+L?", text):
                value = int(text.rstrip("L")) + 1
                b.replace_span("constant", tok, f"{value}{'L' if text.endswith('L') else ''}", f"{text} -> +1")
            elif re.fullmatch(r"(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
                b.replace_span("constant", tok, f"({text} * 2)", f"{text} -> x2")
        elif tok.token == "STR_CONST" and len(text) >= 2 and text[0] in "\"'" \
                and not _in_message_call(tok, by_id, children, b):
            q = text[0]
            b.replace_span("constant", tok, f"{text[:-1]}_m{q}" if len(text) > 2 else f"{q}m{q}", "string altered")
        elif tok.token == "SYMBOL_FUNCTION_CALL" and text in NA_FUNS:
            call = by_id.get(tok.parent)
            call_expr = by_id.get(call.parent) if call else None
            outer = by_id.get(call_expr.parent) if call_expr else None
            negated = outer is not None and any(k.token == "'!'" for k in children.get(outer.id, []))
            if call_expr is not None and call_expr.line1 == call_expr.line2 and not negated:
                b.replace_span("na_handling", call_expr, "!" + b.text_of(call_expr), f"negate {text}()")
    for node in inside:
        if node.terminal:
            continue
        kids = children.get(node.id, [])
        heads = [k.token for k in kids]
        # if conditions
        if heads[:2] == ["IF", "'('"] and len(kids) >= 4:
            cond = kids[2]
            if cond.line1 == cond.line2:
                b.replace_span("branch", cond, "TRUE", "if condition -> TRUE")
                b.replace_span("branch", cond, "FALSE", "if condition -> FALSE")
        # statements of a { block
        if heads and heads[0] == "'{'":
            stmts = [k for k in kids if k.token in STMT_TOKENS]
            for position, stmt in enumerate(stmts):
                first, last = lines[stmt.line1 - 1], lines[stmt.line2 - 1]
                whole = not first[:stmt.col1 - 1].strip() and not last[stmt.col2:].strip()
                if not whole:
                    continue
                block = lines[stmt.line1 - 1:stmt.line2]
                stmt_text = b.text_of(stmt).strip()
                returned = re.fullmatch(r"(?:return\()?\s*([A-Za-z.][A-Za-z0-9._]*)\s*\)?", stmt_text)
                previous = b.text_of(stmts[position - 1]) if position > 0 else ""
                trailing_echo = (position == len(stmts) - 1 and returned is not None and previous
                                 and re.match(r"^\s*" + re.escape(returned.group(1)) + r"\s*(<-|=)(?!=)", previous))
                if not trailing_echo:
                    b.add("statement_delete", stmt.line1, stmt.line2, [], f"delete statement at line {stmt.line1}")
                if stmt.line2 - stmt.line1 <= 2 and _self_update(b.text_of(stmt)):
                    b.add("statement_duplicate", stmt.line1, stmt.line2, [*block, *block],
                          f"duplicate statement at line {stmt.line1}")
                if node.parent and by_id.get(node.parent) and _is_function_body(node, by_id, children) and position > 0:
                    indent = re.match(r"\s*", first).group(0)  # type: ignore[union-attr]
                    if stmt_text not in ("NULL", "return(NULL)", "invisible(NULL)"):
                        b.add("early_return", stmt.line1, stmt.line1, [indent + "return(NULL)", first],
                              f"return(NULL) before line {stmt.line1}")
                    name = b.text_of(formals[0]) if formals else None
                    if name and stmt_text not in (name, f"return({name})", f"invisible({name})"):
                        b.add("early_return", stmt.line1, stmt.line1, [indent + f"return({name})", first],
                              f"return({name}) before line {stmt.line1}")
        # calls: swap the first two positional arguments; subscripts
        if len(kids) >= 4 and kids[1].token == "'('" and kids[0].token == "expr":
            args, named, depth = [], False, 0
            for k in kids[2:]:
                if k.token in ("SYMBOL_SUB", "STR_CONST_SUB", "EQ_SUB"):
                    named = True
                elif k.token == "','":
                    named = False
                elif k.token == "expr" and not named:
                    args.append(k)
            depth += 1
            head = children.get(kids[0].id, [])
            fname = b.text_of(head[0]) if head and head[0].token == "SYMBOL_FUNCTION_CALL" else ""
            if fname in SYMMETRIC:
                args = []
            if len(args) >= 2 and args[0].line1 == args[0].line2 == args[1].line1 == args[1].line2:
                a, c = args[0], args[1]
                line = lines[a.line1 - 1]
                new = line[:a.col1 - 1] + b.text_of(c) + line[a.col2:c.col1 - 1] + b.text_of(a) + line[c.col2:]
                b.add("swap_args", a.line1, a.line1, [new], f"swap arguments at line {a.line1}")
        if len(kids) == 4 and kids[1].token in ("'['", "LBB") and kids[2].token == "expr":
            sub = kids[2]
            inner = children.get(sub.id, [])
            if len(inner) == 1 and inner[0].token in ("SYMBOL", "NUM_CONST") and sub.line1 == sub.line2:
                b.replace_span("indexing", sub, f"{b.text_of(sub)} + 1", "subscript + 1")
    return list(b.out.values())


def _in_message_call(tok: Token, by_id: dict[int, Token], children: dict[int, list[Token]], b: _Builder) -> bool:
    """True when the token sits (within 4 levels) inside a stop()/warning()/message()-style call."""
    node = by_id.get(tok.parent)
    for _ in range(5):
        if node is None:
            return False
        kids = children.get(node.id, [])
        if len(kids) >= 2 and kids[0].token == "expr" and kids[1].token == "'('":
            head = children.get(kids[0].id, [])
            if head and head[0].token == "SYMBOL_FUNCTION_CALL" and b.text_of(head[0]) in MESSAGE_CALLS:
                return True
        node = by_id.get(node.parent)
    return False


def _self_update(text: str) -> bool:
    """``x <- f(x, ...)``-style updates: duplicating them changes behaviour (other duplicates are mostly no-ops)."""
    match = re.match(r"^\s*([A-Za-z.][A-Za-z0-9._]*)\s*(<-|=)(?!=)(.*)$", text, flags=re.S)
    return bool(match and re.search(r"(?<![A-Za-z0-9._$@])" + re.escape(match.group(1)) + r"(?![A-Za-z0-9._])",
                                    match.group(3)))


def _is_function_body(block: Token, by_id: dict[int, Token], children: dict[int, list[Token]]) -> bool:
    parent = by_id.get(block.parent)
    return parent is not None and any(k.token == "FUNCTION" for k in children.get(parent.id, []))


def sample(mutants: Sequence[Mutant], limit: int, seed: str) -> list[Mutant]:
    """Deterministic class-stratified subset (round robin over operator classes)."""
    if len(mutants) <= limit:
        return list(mutants)
    rng = rng_for("exec-env-mutants", seed)
    by_op: dict[str, list[Mutant]] = defaultdict(list)
    for mutant in sorted(mutants, key=lambda m: m.id):
        by_op[mutant.op].append(mutant)
    for group in by_op.values():
        rng.shuffle(group)
    out: list[Mutant] = []
    while len(out) < limit:
        for op in sorted(by_op):
            if by_op[op] and len(out) < limit:
                out.append(by_op[op].pop())
    return out
