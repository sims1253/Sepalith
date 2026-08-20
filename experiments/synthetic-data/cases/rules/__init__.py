"""cases.rules — the CONTRIBUtable transform-rule registry.

A RULE is the unit contributors add: (detector, rewriter, validator) over
the compounding base-sample schema (docs/research/compounding-samples-design.md
§2, cases/compound.py:BaseSample). Writing a rule = writing one small Python
module under cases/rules/rules_*.py and decorating functions:

    from cases.rules import rule, Site, Rewrite

    @rule(id="my_rule", family="lint_rewrite", determinism="D2",
          kind="rewrite", requires=["fn_body"],
          signal="callee == 'my_pattern' (tree-sitter)",
          restraint="skip when the callee is locally shadowed")
    class MyRule:
        def detector(self, bs):        # -> list[Site]
            ...
        def rewrite(self, bs, site):   # -> Rewrite | None
            ...
        def verify(self, old_text, new_text):   # deterministic reward fn
            ...

Rules are auto-discovered: `load_rules()` imports every cases.rules.rules_*
module and collects decorated rules. Every rule must ship SELFTEST cases
(code snippets + expected detector/rewrite outcomes); `run_rules.py
--selftest` executes them against real tree-sitter parses, so a contributor
validates a rule with ONE command and no corpus access.

Determinism classes (compounding doc §3): D1 pure-static (GT = corpus text),
D2 static+validator (GT = our construction, equivalence provable by
re-parse/token-derivation), D3 author-LLM (wording), D4 judge. kind:
"rewrite" rules emit scenario rows; "metadata" rules only annotate the base
sample (smell metrics as difficulty knobs / restraint selectors) and emit
annotations, never rows.

The registry is deliberately a PYTHON PLUGIN layer, not declarative JSON:
detectors need tree-sitter walks and byte-spans that JSON cannot express.
The hybrid: a rule feeds an existing cases/specs family (declared via
`family` + `extends`) and its prompt/prompt-templates live in that spec; the
rule module owns only detection/rewrite/verify. See
docs/research/transform-rule-registry.md for the full contract.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:      # experiments/synthetic-data/
    sys.path.insert(0, str(HERE.parents[1]))

REGISTRY: dict[str, "Rule"] = {}


# ---------------------------------------------------------------------------
# the site / rewrite value types
# ---------------------------------------------------------------------------

@dataclass
class Site:
    """One detector hit inside a base sample.

    row        0-based file row of the anchor (first line the rewrite changes)
    sb, eb    byte span of the rewritten node within the WHOLE file source
    payload   rule-specific ingredients (node refs, tokens, message text...)
    note      one-line human reason (stamped on the derived row)
    row_end   LAST row of the site (defaults to row; multi-row sites like a
              3-line if-guard MUST set it so the suffix starts below them)
    """
    row: int
    sb: int
    eb: int
    payload: dict = field(default_factory=dict)
    note: str = ""
    row_end: int = 0


@dataclass
class Rewrite:
    """A constructed replacement (exact by construction for D2 rules).

    lines      the region_new lines (1..3, one clean statement)
    meta       carried onto the row (old_tok/new_tok/variant/...)
    span_text  the replacement text for the SITE's byte span (what the
               splice re-parse substitutes); defaults to lines joined
    """
    lines: list[str]
    meta: dict = field(default_factory=dict)
    span_text: str = ""


# ---------------------------------------------------------------------------
# the rule contract
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    id: str
    family: str                    # family the rule's rows feed (existing or new)
    determinism: str               # "D1" | "D2" | "D3" | "D4"
    kind: str                      # "rewrite" | "metadata"
    requires: list[str]            # base-sample requirements, e.g. ["fn_body"]
    signal: str                    # the static signal (for the catalog/docs)
    restraint: str = ""            # negative/suppression condition (docs)
    extends: str = ""              # existing family/spec this EXTENDS ("")
    version: int = 1               # bump when detector/rewriter semantics change
    rl_ready: bool = None          # None -> derived: D1/D2 + verify() present
    status: str = "new"            # "new" | "extends" | "exists-as-proposal"
    detector = None                # fn(bs) -> list[Site]
    rewrite = None                 # fn(bs, site) -> Rewrite | None
    verify = None                  # fn(old_text, new_text) -> (ok, reason)
    annotate = None                # metadata rules: fn(bs) -> dict
    selftest = None                # list of Selftest cases
    prescreen = None               # byte-regex list (ALL must match a file):
                                   # cheap signal-directed corpus pre-selection

    @property
    def is_rl_ready(self) -> bool:
        if self.rl_ready is not None:
            return self.rl_ready
        return self.kind == "rewrite" and self.determinism in ("D1", "D2") \
            and self.verify is not None

    def describe(self) -> str:
        return (f"{self.id}@{self.version} [{self.determinism}/{self.kind}] "
                f"family={self.family} status={self.status} "
                f"rl_ready={self.is_rl_ready} :: {self.signal}")


class Selftest:
    """One contributor-authored test case: R code + expected outcomes.

    expect_sites   number of detector sites (default: >= 1)
    first_new      expected region_new[0] of the first rewrite (optional)
    suppressed     when True, expect the rewrite to be REJECTED at the
                   registry gate or by the rule's own hygiene checks
    """

    def __init__(self, code: bytes, expect_sites: int | None = 1,
                 first_new: str | None = None, suppressed: bool = False,
                 why: str = ""):
        self.code = code
        self.expect_sites = expect_sites
        self.first_new = first_new
        self.suppressed = suppressed
        self.why = why


def rule(id: str, family: str, determinism: str, kind: str, requires: list,
         signal: str, restraint: str = "", extends: str = "", version: int = 1,
         rl_ready: bool | None = None, status: str = "new"):
    """Class decorator registering a rule. The class is instantiated once and
    its detector/rewrite/verify/annotate/selftest methods become the rule."""
    def deco(cls):
        inst = cls()
        r = Rule(id=id, family=family, determinism=determinism, kind=kind,
                 requires=list(requires), signal=signal, restraint=restraint,
                 extends=extends, version=version, rl_ready=rl_ready,
                 status=status)
        r.detector = getattr(inst, "detector", None)
        r.rewrite = getattr(inst, "rewrite", None)
        r.verify = getattr(inst, "verify", None)
        r.annotate = getattr(inst, "annotate", None)
        r.selftest = getattr(inst, "selftest", None) or getattr(cls, "SELFTEST", None)
        r.prescreen = getattr(cls, "PRESCREEN", None)
        if kind == "rewrite" and (r.detector is None or r.rewrite is None
                                  or r.verify is None):
            raise TypeError(f"rule {id!r}: rewrite rules need detector, "
                            f"rewrite AND verify methods")
        if kind == "metadata" and r.annotate is None:
            raise TypeError(f"rule {id!r}: metadata rules need annotate()")
        if id in REGISTRY:
            raise KeyError(f"duplicate rule id {id!r}")
        REGISTRY[id] = r
        return cls
    return deco


def load_rules() -> dict[str, Rule]:
    """Import every cases.rules.rules_* module (idempotent) and return the
    registry. A broken module raises at load time — a contributor's rule that
    does not import never silently disappears from the matrix."""
    for m in pkgutil.iter_modules([str(HERE)]):
        if m.name.startswith("rules_"):
            importlib.import_module(f"cases.rules.{m.name}")
    return REGISTRY


# ---------------------------------------------------------------------------
# shared gates every rewrite row passes (the registry's own layer on top of
# the existing cases gates; production promotion registers these into
# validators.ROW_CHECKS — this file deliberately does not modify it)
# ---------------------------------------------------------------------------

def check_rewrite_row(row: dict, rule_: Rule) -> tuple[bool, str]:
    """Registry row gate: (1) the existing structural row check, (2) the
    mid_body_edit_line floor (few short lines, exactly one clean statement),
    (3) the rule's own verify() re-derivation (old carried corpus_line ->
    region_new), and (4) rule id/version provenance on the row."""
    import cases.validators as V
    ok, reason = V.check_row(row)
    if not ok:
        return False, f"rowcheck: {reason}"
    target = "\n".join(row["region_new"])
    ok, reason = V.REGISTRY["mid_body_edit_line"](
        target, {"max_lines": 3, "max_len": 220})
    if not ok:
        return False, f"floor[mid_body_edit_line]: {reason}"
    old = row.get("corpus_line") or ""
    ok, reason = rule_.verify(old, target)
    if not ok:
        return False, f"verify[{rule_.id}]: {reason}"
    # geometry pins (the scenario-row conventions): typed prefix tail, the
    # function's closing brace + a non-blank remainder in the suffix
    if not row["prefix"][-1].strip():
        return False, "prefix must end with a typed (non-blank) line"
    sfx = row.get("suffix") or []
    if not any(l.strip() == "}" for l in sfx):
        return False, "closing brace of the function not visible in the suffix"
    if not any(l.strip() and l.strip() != "}" for l in sfx):
        return False, "site is the last statement of the function"
    return True, ""


def splice_reparse(bs, sb: int, eb: int, replacement: str) -> bool:
    """Re-parse the whole defining statement with [sb, eb) replaced — the
    compounding doc's D2 equivalence proof (cases.compound._splice_reparse,
    reused verbatim)."""
    from cases.compound import _splice_reparse
    return _splice_reparse(bs, sb, eb, replacement)


# ---------------------------------------------------------------------------
# parent-sample linking (dataset bookkeeping contract — see the registry doc
# §mechanics: stable IDs, split hygiene, dedup keys, per-base-sample yield)
# ---------------------------------------------------------------------------

def base_sample_id(bs) -> str:
    """STABLE content-hash id of a base sample: sha1 over
    (origin-kind, package, path, defining-statement source). NOT a counter —
    the id survives rebuilds, re-shuffles and backend switches because it is
    derived from content + provenance only (the same function re-collected
    next week yields the same id; a row's parent is always re-findable and
    the derivation is regenerable from (base_sample_id, rule_id, params)).
    Duplicate identical functions inside one file intentionally share the id
    (that is dedup working)."""
    import hashlib
    src = bs.b.src
    parent = bs.fn.parent
    base_start = parent.start_byte if (parent is not None
                                       and parent.type == "binary_operator") \
        else bs.fn.start_byte
    h = hashlib.sha1()
    h.update(b"corpus\x00")                     # origin kind (author: later)
    h.update(bs.b.package.encode("utf-8", "replace") + b"\x00")
    h.update(bs.b.rel.encode("utf-8", "replace") + b"\x00")
    h.update(src[base_start:bs.fn.end_byte])
    return "bs:" + h.hexdigest()[:12]


def derivation_key(base_id: str, rule_: Rule, params: dict) -> str:
    """Dedup key across rebuilds/backends: base_sample_id + rule_id@version +
    content-hash of the derivation params. Two rows with the same key are
    the SAME derivation (e.g. re-collected after a cache reset, or emitted
    by the zai fallback after the agy run) — keep one."""
    import hashlib
    blob = json.dumps(params or {}, sort_keys=True)
    ph = hashlib.sha1(blob.encode()).hexdigest()[:8]
    return f"{base_id}:{rule_.id}@{rule_.version}:{ph}"


def make_row(bs, rule_: Rule, site: Site, rw: Rewrite, window: int = 10,
             params: dict | None = None) -> dict:
    """Suffix-convention scenario row for one rewrite site (the
    compound.item_to_row shape, case='rules_proto'). The row EXTENDS the
    cases provenance convention (package, path, corpus_target) with the
    parent-link block: base_sample_id + rule_id + rule_version + derivation
    params, so any row traces to its parent and can be regenerated."""
    b = bs.b
    target_lines = [l for l in rw.lines]
    prefix = [b.line_str(r) for r in range(max(0, bs.head_row - window),
                                           site.row)]
    row_end = site.row_end or site.row
    suffix = [b.line_str(r) for r in
              range(row_end + 1, min(b.nlines(), bs.r1 + 1 + window))]
    bsid = base_sample_id(bs)
    row = dict(
        family=rule_.family, transform=rule_.id, package=b.package, path=b.rel,
        prefix=prefix or [""], region_old=[""], region_new=target_lines,
        cursor_idx=0, event_diff="", note=site.note or rule_.signal,
        suffix=suffix or [""],
        corpus_target="\n".join(target_lines),
        corpus_line=b.line_str(site.row),
        rule=f"{rule_.id}@{rule_.version}",
        determinism=rule_.determinism, rl_ready=rule_.is_rl_ready,
        case="rules_proto", backend="deterministic", model="static-transform",
        full_prompt="", generated_at="2026-08-20T00:00:00",
        base_sample=bsid, base_sample_id=bsid,
        derivation=dict(base_sample_id=bsid, rule_id=rule_.id,
                        rule_version=rule_.version,
                        params=dict(params or {}, window_lines=window)))
    row.update(rw.meta)
    row["model_target"] = row["corpus_target"]      # mock-draw convention
    return row
