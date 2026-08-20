"""Registered validators: layer 3 of the gate (layer 1 = JSON extraction in
backends.py, layer 2 = spec schema check in generate.py).

Every validator has the signature  fn(target: str, params: dict) ->
(ok: bool, reason: str)  and must be NON-EXECUTABLE unless it explicitly
opts into the validate.py gate (`validate_py` runs Rscript/jarl and is only
for whole-snippet code targets; the structural checks below are pure
tree-sitter and safe everywhere).

Also hosts the final row-structure checks (the scenario_block shape plus
optional per-case row_check registered here).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for scenarios/
import tree_sitter_r
from tree_sitter import Language, Parser

REGISTRY: dict = {}
ROW_CHECKS: dict = {}

_PARSER = Parser(Language(tree_sitter_r.language()))

# tidyselect helpers (idea #1 of docs/research/gemini-family-ideas.txt).
# first-argument shape per helper:
#   string  -> string literal required ("PREFIX")
#   vector  -> call to c(...) or a variable (all_of/any_of/num_range)
#   numeric -> numeric literal or expression (last_col offset)
#   predicate -> identifier / call (where(is.numeric))
TIDYSELECT_HELPERS = {
    "starts_with": "string",
    "ends_with": "string",
    "contains": "string",
    "matches": "string",
    "regex": "string",
    "num_range": "string",
    "last_col": "numeric",
    "all_of": "vector",
    "any_of": "vector",
    "where": "predicate",
}

MIN_ROW_KEYS = ("family", "package", "path", "prefix", "region_old",
                "region_new", "cursor_idx", "event_diff", "note",
                "case", "backend", "model", "full_prompt", "generated_at")


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def register_row_check(name: str):
    def deco(fn):
        ROW_CHECKS[name] = fn
        return fn
    return deco


def get_validator(cfg: dict):
    name = cfg["name"]
    if name not in REGISTRY:
        raise KeyError(f"unknown validator {name!r}; registered: {sorted(REGISTRY)}")
    params = dict(cfg.get("params") or {})

    def run(target: str):
        if not isinstance(target, str):
            return False, f"target is {type(target).__name__}, not str"
        return REGISTRY[name](target, params)
    return run


# ---------------------------------------------------------------------------
# tree-sitter fragment helpers (mirror comment_to_code.fragment_statements)
# ---------------------------------------------------------------------------

def parse_fragment(text: str):
    return _PARSER.parse(text.encode("utf-8", "surrogateescape"))


def _walk(n):
    stack = [n]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def fragment_clean(text: str) -> bool:
    """Fragment parses with no ERROR/missing nodes anywhere."""
    tree = parse_fragment(text)
    if tree.root_node.has_error:
        return False
    return not any(n.type == "ERROR" or n.is_missing
                   for n in _walk(tree.root_node))


def top_level_calls(text: str) -> list:
    """Named top-level children of the fragment parse that are calls."""
    tree = parse_fragment(text)
    return [c for c in tree.root_node.children
            if c.is_named and c.type == "call"]


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------

@register("r_comment_gate")
def v_comment(target: str, params: dict):
    """One concise single-line R comment: non-empty, length-capped, and free
    of code-ish punctuation (the comment_to_code._gate_ok convention)."""
    max_len = int(params.get("max_len", 90))
    forbidden = list(params.get("forbid") or [";", "<-", "(", "\n"])
    c = target.strip()
    if not c:
        return False, "empty comment"
    if len(c) > max_len:
        return False, f"comment longer than {max_len} chars ({len(c)})"
    for f in forbidden:
        if f in c:
            return False, f"forbidden {f!r} in comment"
    return True, ""


@register("r_fragment")
def v_fragment(target: str, params: dict):
    """Generic tree-sitter structural gate: the target must re-parse as
    clean R (no ERROR/missing nodes) and hold min..max top-level statements."""
    lo = int(params.get("min_statements", 1))
    hi = int(params.get("max_statements", 10))
    max_lines = int(params.get("max_lines", 25))
    if not target.strip():
        return False, "empty target"
    if len(target.splitlines()) > max_lines:
        return False, f"more than {max_lines} lines"
    if not fragment_clean(target):
        return False, "does not parse as clean R"
    n = len([c for c in parse_fragment(target).root_node.children if c.is_named])
    if not lo <= n <= hi:
        return False, f"{n} top-level statements, expected {lo}..{hi}"
    return True, ""


@register("tidyselect_call")
def v_tidyselect(target: str, params: dict):
    """tidyselect helper completion (NON-EXECUTABLE, pure tree-sitter): the
    target must be ONE clean call whose callee is a tidyselect helper with
    the correct first-argument shape (string literal / vector / numeric /
    predicate). No Rscript, no jarl."""
    helpers = params.get("helpers") or dict(TIDYSELECT_HELPERS)
    max_len = int(params.get("max_len", 120))
    t = target.strip()
    if not t:
        return False, "empty completion"
    if len(t) > max_len:
        return False, f"completion longer than {max_len} chars ({len(t)})"
    if "\n" in t:
        return False, "completion must be a single line"
    if not fragment_clean(t):
        return False, "does not parse as clean R"
    calls = top_level_calls(t)
    if len(calls) != 1:
        return False, f"expected exactly 1 call, fragment has {len(calls)}"
    call = calls[0]
    head = call.children[0] if call.children else None
    name = None
    if head is not None and head.type == "identifier":
        name = t.encode()[head.start_byte:head.end_byte].decode("utf-8", "replace")
    elif head is not None and head.type == "namespace_operator":
        name = t.encode()[head.start_byte:head.end_byte].decode("utf-8", "replace")
        name = name.split("::")[-1]
    if name not in helpers:
        return False, f"callee {name!r} is not a tidyselect helper"
    args_node = next((c for c in call.children if c.type == "arguments"), None)
    arg_nodes = [a for a in (args_node.children if args_node is not None else [])
                 if a.type == "argument"]
    if not arg_nodes:
        return False, "helper call has no arguments"
    src = t.encode()
    first = _argument_value(arg_nodes[0])
    if first is None:
        return False, "helper call has an empty first argument"
    kind = helpers[name]
    ftype = first.type
    ftxt = src[first.start_byte:first.end_byte].decode("utf-8", "replace")
    if kind == "string":
        if ftype != "string":
            return False, f"{name} needs a string literal first arg, got {ftype}"
    elif kind == "vector":
        if ftype not in ("call", "identifier"):
            return False, f"{name} needs a character-vector expression, got {ftype}"
        if ftype == "call" and ftxt.lstrip().startswith("list("):
            return False, f"{name} vector arg must be c(...) / a variable / a call, got {ftxt[:30]!r}"
    elif kind == "numeric":
        if ftype not in ("float", "integer", "identifier", "call"):
            return False, f"{name} needs a numeric/offset expression, got {ftype}"
    elif kind == "predicate":
        if ftype not in ("identifier", "call"):
            return False, f"where() needs a predicate (is.* call/name), got {ftype}"
    return True, ""


def _argument_value(arg_node):
    """Value node of a tree-sitter-r `argument`: the first named child,
    skipping a `name =` prefix when the argument is named."""
    kids = list(arg_node.children)
    named = [c for c in kids if c.is_named]
    if not named:
        return None
    if any(c.type == "=" for c in kids):  # named argument: value after '='
        i = next(i for i, c in enumerate(kids) if c.type == "=")
        after = [c for c in kids[i + 1:] if c.is_named]
        return after[0] if after else None
    return named[0]


@register("validate_py")
def v_validate_py(target: str, params: dict):
    """Bridge into the shared 3-layer gate of validate.py (jsonschema ->
    Rscript parse -> jarl lint). Only for whole-snippet code targets where
    execution-based validation is acceptable; needs Rscript (+jarl) on PATH."""
    import json as _json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import validate as V
    schemas = {"analyst": V.ANALYST_SCHEMA, "finish": V.FINISH_SCHEMA}
    schema = schemas[params.get("schema", "analyst")]
    obj = _json.loads(target) if target.strip().startswith("{") else {
        params.get("intent_key", "intent"): "generated snippet",
        params.get("code_key", "code"): target,
        "packages_used": [],
    }
    ok, layer, info, _w = V.validate(
        obj, schema, code_key=params.get("code_key", "code"),
        run_jarl=bool(params.get("run_jarl", False)))
    return ok, ("" if ok else f"validate.py layer {layer}: {info}")


# ---------------------------------------------------------------------------
# row-structure checks
# ---------------------------------------------------------------------------

def check_row(row: dict, extra: dict | None = None) -> tuple[bool, str]:
    """Final self-check on every assembled row: the scenario_block shape
    (single-line string lists, empty-cursor-line convention, GT non-empty)
    plus full provenance, then the optional registered case-specific check."""
    for k in MIN_ROW_KEYS:
        if k not in row or row[k] is None:
            return False, f"row missing field {k!r}"
    for f in ("prefix", "region_old", "region_new"):
        v = row.get(f)
        if not isinstance(v, list) or not v:
            return False, f"{f} must be a non-empty list"
        if any(not isinstance(l, str) or "\n" in l for l in v):
            return False, f"{f} must be single-line strings"
    if row["region_old"] != [""]:
        return False, "region_old must be the empty cursor line"
    if row["cursor_idx"] != 0:
        return False, "cursor_idx must be 0 (start of the empty line)"
    if not any(l.strip() for l in row["region_new"]):
        return False, "region_new must be non-blank"
    if row["event_diff"] != "":
        return False, "case rows carry no triggering event"
    if extra:
        name = extra.get("name", "")
        fn = ROW_CHECKS.get(name)
        if fn is None:
            return False, f"unknown row_check {name!r}"
        ok, reason = fn(row, dict(extra.get("params") or {}))
        if not ok:
            return False, reason
    return True, ""


@register_row_check("ends_with_comment_line")
def rc_comment(row: dict, params: dict):
    last = row["prefix"][-1].lstrip()
    if not last.startswith("#"):
        return False, "prefix must end with the comment line"
    if last.startswith("#'"):
        return False, "roxygen comments never trigger this family"
    return True, ""


@register_row_check("mid_line_cursor")
def rc_midline(row: dict, params: dict):
    """Completion families: the last prefix line must be a genuinely partial
    line (ends with an opener/comma/unary operator — never a complete
    statement). '-' matters: select(-starts_with(...)) elides to 'select(-'."""
    tail = row["prefix"][-1].rstrip()
    if not tail or tail[-1] not in "(,= +|>%$@-!&:":
        return False, f"prefix must end mid-expression, got ...{tail[-24:]!r}"
    return True, ""
