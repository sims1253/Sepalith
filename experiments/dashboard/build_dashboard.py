#!/usr/bin/env python3
"""Build the Sepalith status dashboard (self-contained HTML) from state JSON.

Two tabs: "Experiments" (the classic tracker, rendered from
dashboard_state.json) and "Synthetic Data" (a landscape overview of the
synthetic-data program, computed LIVE at build time from the dataset
files, case specs, the transform-rule registry and the ideation-tournament
outputs — nothing on that tab is hardcoded, so a rebuild refreshes it).

Update workflow: edit dashboard_state.json as results land, then
  uv run python experiments/dashboard/build_dashboard.py
  npx postplan upload experiments/dashboard/index.html

Raw-HTML gotcha: any cell that starts with "<" is passed through raw by
table(); every dynamic string must go through esc() first (R code full of
"<-" especially).
"""
import html
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "dashboard_state.json"
OUT = HERE / "index.html"

# --- synthetic-data landscape sources (live-computed at build time) --------
DATASETS = Path("/mnt/h/sepalith/datasets")
CASES_V1 = DATASETS / "cases_v1"
SCEN_V1 = DATASETS / "scenarios_v1"
SFT_V7_STATS = DATASETS / "sft_v7" / "stats.json"
SYN = HERE.parent / "synthetic-data"
SPECS = SYN / "cases" / "specs"
RULES_DIR = SYN / "cases" / "rules"
TOUR = SYN / "results" / "ideation_tournament"


def esc(s):
    return html.escape(str(s))


def chip(status):
    color = {"done": "#2ea043", "running": "#d29922", "pending": "#8b949e",
             "blocked": "#f85149", "win": "#2ea043", "info": "#58a6ff"}.get(status, "#8b949e")
    return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;padding:1px 8px;border-radius:10px;font-size:12px;white-space:nowrap">{esc(status)}</span>'


def table(headers, rows):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td style="{"font-family:monospace" if i > 0 else ""}">{c if str(c).startswith("<") else esc(c)}</td>'
            for i, c in enumerate(r)) + "</tr>"
    return f'<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def pre(code, note=""):
    out = f'<pre style="background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px 10px;margin:6px 0 0;font-size:12.5px;overflow-x:auto;white-space:pre">{esc(code)}</pre>'
    if note:
        out += f'<div class="note" style="margin:4px 0 0">{esc(note)}</div>'
    return out


def fmt_int(n):
    return f"{n:,}" if isinstance(n, int) else str(n)


# ---------------------------------------------------------------------------
# (a) aggregates — live row counts per family + sft_v7 mixture composition
# ---------------------------------------------------------------------------

_VARIANT_RE = re.compile(
    r"_(zai|spark|gemini|xpreview(?:-free)?|orfree_[A-Za-z0-9.\-]+|free)$")
# files that are inputs/ledger, not case rows
_SKIP_FILES = {"base_samples_spark", "suffix_scenarios"}


def _count_lines(p: Path) -> int:
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            n += b.count(b"\n")
    return n


def _stats_backend(p: Path):
    """(backend, model) from a .jsonl.stats.json sidecar, if present."""
    sp = p.with_name(p.name + ".stats.json")
    if not sp.exists():
        return None, None
    try:
        d = json.loads(sp.read_text())
        if isinstance(d, dict):
            c = d.get("counts") if isinstance(d.get("counts"), dict) else d
            return c.get("backend"), c.get("model")
    except Exception:
        pass
    return None, None


def _family_of(stem: str) -> str:
    """Merge author-variant files into one family (rewrite_lint_fix_spark ->
    rewrite_lint_fix)."""
    base = stem
    m = _VARIANT_RE.search(base)
    while m:
        base = base[:m.start()]
        m = _VARIANT_RE.search(base)
    if base == "rewrite_fixissue":
        base = "fix_issue_inject"      # spark/zai-authored fix-issue waves
    return base


def gather_bank(mixture):
    """Live-count every *.jsonl under cases_v1 + scenarios_v1, group per
    family, attach authors/models from the .stats sidecars, and diff against
    the sft_v7 mixture (rows banked since the last cut)."""
    fams = {}          # family -> dict(rows, files, backends, eval_rows)
    for d in (CASES_V1, SCEN_V1):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jsonl")):
            if p.name.endswith(".done.jsonl") or p.name.endswith(".bak"):
                continue
            stem = p.stem
            if stem in _SKIP_FILES:
                continue
            is_eval = stem.endswith("_eval")
            base_stem = stem[:-5] if is_eval else stem
            fam = _family_of(base_stem)
            # keep mixture-native keys that carry their own variant name
            # (comment_to_code_gemini is its own sft_v7 family, while
            # rewrite_lint_fix_spark folds into rewrite_lint_fix)
            if fam not in mixture and base_stem in mixture:
                fam = base_stem
            be, model = _stats_backend(p)
            tag = "/".join(x for x in (be, model) if x) if be else None
            if not tag:
                for tok, label in (("zai", "zai"), ("spark", "opencode-spark"),
                                   ("gemini", "agy-gemini"),
                                   ("orfree", "openrouter-free"),
                                   ("xpreview", "zen-xpreview")):
                    if tok in stem:
                        tag = label
                        break
            ent = fams.setdefault(fam, dict(rows=0, eval_rows=0, files=[],
                                            tags=set()))
            n = _count_lines(p)
            if is_eval:
                ent["eval_rows"] += n
            else:
                ent["rows"] += n
            ent["files"].append(p.parent.name + "/" + p.name)
            if tag:
                ent["tags"].add(tag)
            elif fam in ("rename_propagation", "pipe_rewrite",
                         "format_propagation", "na_rm_propagation",
                         "no_op", "mid_roxygen", "doc_sync", "finish_block",
                         "mid_body_edit", "fix_issue_inject", "edit_pairs",
                         "hidden_r_instruction", "roxygen_drafting",
                         "astfim_partial", "namespace_qualify_propagation",
                         "pkg_metadata_sync", "removed_block_comment",
                         "expectation_completion", "pipe_chain_link"):
                ent["tags"].add("deterministic corpus-side" if fam != \
                                "hidden_r_instruction" else "harvested")
    rows = []
    for fam in sorted(fams, key=lambda f: -(fams[f]["rows"] + fams[f]["eval_rows"])):
        e = fams[fam]
        live = e["rows"] + e["eval_rows"]
        mix = mixture.get(fam, 0)
        new = live - mix
        rows.append([fam, fmt_int(e["rows"]) +
                     (f" +{fmt_int(e['eval_rows'])} eval" if e["eval_rows"] else ""),
                     fmt_int(mix) if mix else "0",
                     ("+" + fmt_int(new)) if new > 0 else "-",
                     ", ".join(sorted(e["tags"])) or "?",
                     len(e["files"])])
    return rows


def gather_mixture():
    """sft_v7 train/eval per family + share of the train mixture."""
    d = json.loads(SFT_V7_STATS.read_text())
    rep = d["report"]
    fams = rep["families"]
    total_t, total_e = rep["total_train"], rep["total_eval"]
    rows = []
    for fam, v in sorted(fams.items(), key=lambda kv: -kv[1]["train"]):
        t, e = v["train"], v["eval"]
        share = (100.0 * t / total_t) if total_t else 0.0
        rows.append([fam, fmt_int(t), fmt_int(e), f"{share:.1f}%"])
    return rows, total_t, total_e


# ---------------------------------------------------------------------------
# (b) scenario kinds — human-readable catalog of the family TYPES
# ---------------------------------------------------------------------------

# Curated one-liners for families without a spec file (spec'd families use
# the spec's own description, first sentence).
_CURATED = {
    "rename_propagation":
        "A rename performed at one site must be carried to the next "
        "occurrence of the symbol — deterministic corpus-side edits, "
        "exact validator.",
    "na_rm_propagation":
        "After the user adds na.rm = TRUE once, the next aggregation call "
        "needs it too.",
    "format_propagation":
        "A formatting/style edit made at one site propagates to the next "
        "site of the same pattern.",
    "doc_sync":
        "Roxygen docs must follow a code edit (signature change -> man "
        "page) — the maintenance skill, not fresh drafting.",
    "finish_block":
        "Function-body completion from the block start — the backbone "
        "family (90k rows).",
    "mid_roxygen":
        "Complete a roxygen doc block at a mid-line cursor inside the "
        "block (suffix convention).",
    "hidden_r_instruction":
        "R rows harvested from general code instruction datasets.",
    "edit_pairs":
        "Real git commit diffs mined into edit rows (eval-heavy; the "
        "human-modernization mine is thin, ~3% of pairs).",
    "rewrite_lint_fix":
        "Author-LLM fixes of real lint findings (T/F, seq, paste0, "
        "sapply, class==) on corpus code, behavior-preservation gated.",
    "fix_issue_inject":
        "Reverse-strip injection: the dirty twin is injected, the corpus "
        "original is the exact ground truth (char_swap, wrong_variable, "
        "boundary_operator).",
    "loop_rewrite":
        "Loop -> vectorized rewrites (verified-behavior track; corpus "
        "supply measured thin ~0.06% of functions).",
    "no_op":
        "Restraint: the RIGHT completion is changing nothing — plausible "
        "hooks are bait; teaches when NOT to edit (RL no_op reward went "
        "0.39 -> 0.97).",
    "roxygen_drafting":
        "Function -> full roxygen doc block, mined corpus ground truth.",
    "comment_drafting":
        "One-line comments drafted for real corpus code (5 style pool, "
        "comment gate).",
    "comment_insert":
        "Insert a comment at the right place mid-function (cursor "
        "positioning skill).",
    "synthetic_analyst":
        "Analyst-style scripts generated from a domain/construct grid.",
    "paper_to_r":
        "Reimplement a paper's reported analysis as R (grows from "
        "Bioconductor + pwc repos, not LaTeX).",
    "compound":
        "One base sample -> many cases: multi-family derivations per "
        "corpus function via the author-LLM grid (10 rows/sample "
        "measured, mock 40-3500x cheaper).",
    "registry":
        "Deterministic transform-rule registry: detector/rewrite/verify "
        "plugins over real corpus code — verify is pure re-derivation, "
        "usable as deterministic RL reward (table below).",
}

_KIND_GROUPS = [
    ("Propagation families",
     ["rename_propagation", "na_rm_propagation", "format_propagation",
      "doc_sync", "namespace_qualify_propagation", "pkg_metadata_sync"]),
    ("Completion families",
     ["finish_block", "mid_body_edit", "astfim_partial", "pipe_chain_link",
      "expectation_completion", "tidyselect_completion",
      "trycatch_handler_completion", "removed_block_comment",
      "mid_roxygen", "hidden_r_instruction"]),
    ("Rewrite + fix families",
     ["rewrite_lint_fix", "fix_issue_inject", "loop_rewrite", "edit_pairs"]),
    ("Restraint (no_op) families",
     ["no_op"]),
    ("Drafting families",
     ["roxygen_drafting", "comment_drafting", "comment_insert",
      "comment_to_code_styles", "synthetic_analyst", "paper_to_r"]),
    ("Compounding",
     ["compound", "registry"]),
]


def _first_sentence(text, cap=220):
    text = re.sub(r"\s+", " ", text or "").strip()
    m = re.match(r"(.{20,}?[.!?])(\s|$)", text)
    out = m.group(1) if m else text
    return out[:cap - 1] + ("…" if len(out) >= cap else "")


def gather_kinds(bank_rows, n_rules=None):
    """(group_title, rows) per group; rows = family, description, source,
    live rows, in v7."""
    specs = {}
    for p in SPECS.glob("*.json"):
        if p.name == "proposals_v1.json":
            continue
        try:
            d = json.loads(p.read_text())
            if d.get("family"):
                specs[d["family"]] = (p.stem, d.get("description", ""))
        except Exception:
            pass
    bank = {r[0]: r for r in bank_rows}
    out = []
    for gtitle, fams in _KIND_GROUPS:
        rows = []
        for fam in fams:
            desc, src = _CURATED.get(fam, ""), "curated"
            if fam in specs:
                stem, d = specs[fam]
                desc = _first_sentence(d)
                src = f"spec: cases/specs/{stem}.json"
            elif fam in ("compound", "registry"):
                src = ("cases/compound.py grid" if fam == "compound"
                       else "cases/rules/ registry")
            b = bank.get(fam)
            if fam == "registry":
                live = f"{n_rules} rules" if n_rules else "see table below"
            elif b:
                live = b[1]
            else:
                live = "0 (spec/proposal only)"
            mix = b[2] if b else "0"
            rows.append([fam, desc, src, live, mix])
        out.append((gtitle, rows))
    return out


# ---------------------------------------------------------------------------
# (c) modular rules with minimal examples (SELFTEST-derived, live-executed)
# ---------------------------------------------------------------------------

def _rule_examples_live():
    """Import the real registry and execute each rule's smallest positive
    SELFTEST snippet (detector -> rewrite) so the before->after example is
    the rule's own verified behavior, not a hand copy."""
    if str(SYN) not in sys.path:
        sys.path.insert(0, str(SYN))
    import scenarios as S                                    # noqa: F401
    from cases.compound import BaseSample
    from cases.rules import load_rules
    import cases.validators as V
    rules = load_rules()
    out = []
    for rid in sorted(rules):
        r = rules[rid]
        # smallest positive selftest snippet
        cands = []
        for case in (r.selftest or []):
            code, opts = case[0], (case[1] if len(case) > 1 else {})
            want = opts.get("expect_sites")
            want = 1 if want is None else want      # default: >= 1 site
            if r.kind == "metadata":
                if opts.get("expect_annotations"):
                    cands.append((len(code), code, opts))
            elif want >= 1 and not opts.get("suppressed"):
                cands.append((len(code), code, opts))
        if not cands:
            out.append(dict(id=rid, family=r.family, det=r.determinism,
                            kind=r.kind, signal=r.signal,
                            restraint=r.restraint, before=None, after=None,
                            note="no positive selftest snippet"))
            continue
        _, code, opts = min(cands)
        try:
            b = S.Bundle("selftest", "R/selftest.R", code)
            fn = next(n for n in V._walk(b.tree.root_node)
                      if n.type == "function_definition")
            bs = BaseSample(b, fn, 0)
            if r.kind == "metadata":
                ann = r.annotate(bs)
                before = code.decode().rstrip("\n")
                after = None
                note = "annotate() -> " + json.dumps(
                    {k: ann[k] for k in ("cyclo", "nesting", "body_lines",
                                         "flags") if k in ann})
            else:
                sites = r.detector(bs)
                rw = r.rewrite(bs, sites[0])
                lines = code.decode().rstrip("\n").split("\n")
                after_lines = list(lines)
                for i, nl in enumerate(rw.lines):
                    after_lines[sites[0].row + i] = nl
                before = "\n".join(lines)
                after = "\n".join(after_lines)
                note = sites[0].get("note", "") if isinstance(sites[0], dict) \
                    else getattr(sites[0], "note", "")
        except Exception as e:                              # noqa: BLE001
            before, after, note = code.decode().rstrip("\n"), None, \
                f"live execution failed ({type(e).__name__})"
        out.append(dict(id=rid, family=r.family, det=r.determinism,
                        kind=r.kind, signal=r.signal, restraint=r.restraint,
                        before=before, after=after, note=note))
    return out


def _rule_examples_static():
    """Fallback when the registry cannot be imported (no tree-sitter in the
    build env): AST-parse the rule modules for decorator metadata and the
    SELFTEST lists; the example is the smallest positive snippet plus its
    first_new line when the case carries one."""
    import ast
    out = []
    for p in sorted(RULES_DIR.glob("rules_*.py")):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            meta = {}
            for dec in cls.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") \
                        == "rule":
                    for kw in dec.keywords:
                        if isinstance(kw.value, ast.Constant):
                            meta[kw.arg] = kw.value.value
            for st in (n for n in cls.body
                       if isinstance(n, ast.Assign)
                       and any(getattr(t, "id", "") == "SELFTEST"
                               for t in n.targets)):
                cases = []
                for el in (st.value.elts
                           if isinstance(st.value, ast.Tuple) else []):
                    parts = [e for e in el.elts]
                    if not parts:
                        continue
                    try:
                        code = ast.literal_eval(parts[0])
                        opts = ast.literal_eval(parts[1]) if len(parts) > 1 \
                            else {}
                    except (ValueError, SyntaxError):
                        continue
                    want = opts.get("expect_sites")
                    want = 1 if want is None else want
                    if isinstance(code, bytes) and want >= 1 and \
                            not opts.get("suppressed"):
                        cases.append((len(code), code, opts))
                if cases and meta:
                    _, code, opts = min(cases)
                    before = code.decode().rstrip("\n")
                    after = None
                    if opts.get("first_new"):
                        ln = opts["first_new"].strip()
                        after = "first rewritten line: " + ln
                    out.append(dict(
                        id=meta.get("id", cls.name), family=meta.get("family"),
                        det=meta.get("determinism"), kind=meta.get("kind"),
                        signal=meta.get("signal", ""),
                        restraint=meta.get("restraint", ""),
                        before=before, after=after,
                        note="static extraction (registry import failed)"))
    return out


def gather_rules():
    try:
        rows = _rule_examples_live()
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] live rule import failed ({type(e).__name__}: {e}); "
              f"falling back to static extraction", file=sys.stderr)
        rows = _rule_examples_static()
    for r in rows:
        if r["signal"]:
            r["signal"] = _first_sentence(r["signal"], cap=200)
        if r["restraint"]:
            r["restraint"] = _first_sentence(r["restraint"], cap=200)
    return rows


def render_rules(rows):
    out = []
    for r in rows:
        cell = ""
        if r["before"] is not None:
            cell += pre(r["before"])
            if r["after"]:
                cell += ('<div class="note" style="margin:2px 0 0 2px">'
                         '&#8628; becomes</div>')
                cell += pre(r["after"])
        if r["note"]:
            cell += f'<div class="note" style="margin:4px 0 0">{esc(r["note"])}</div>'
        if r["restraint"]:
            cell += (f'<div class="note" style="margin:2px 0 0">restraint: '
                     f'{esc(r["restraint"])}</div>')
        out.append([f'<code>{esc(r["id"])}</code>',
                    f'{r["kind"]} / {r["det"]}',
                    r["signal"],
                    cell])
    return table(["rule", "kind/D", "detects", "minimal example (from the rule's own SELFTEST)"], out)


# ---------------------------------------------------------------------------
# (d) tournament output
# ---------------------------------------------------------------------------

def gather_tournament():
    out = {}
    specs = []
    p = TOUR / "build_specs.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                specs.append([f"r{d.get('round', '?')}",
                              d.get("title", "?"),
                              d.get("kind", "?"),
                              (f'{d["composite"]:.2f}' if isinstance(
                                  d.get("composite"), (int, float))
                               else "v1 winner"),
                              d.get("band", "")])
    out["build_specs"] = specs
    bank = []
    p = TOUR / "spec_bank.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                bank.append(json.loads(line))
    out["bank_size"] = len(bank)
    out["top_banked"] = [
        [f'{d.get("composite", 0):.2f}', d.get("title", "?"),
         _first_sentence((d.get("proposal") or {}).get("description", "")
                         or d.get("note", ""), cap=200)]
        for d in sorted(bank, key=lambda x: -(x.get("composite") or 0))[:5]]
    rounds_dir = TOUR / "rounds"
    done = in_flight = 0
    if rounds_dir.is_dir():
        for rd in rounds_dir.iterdir():
            if re.fullmatch(r"r\d{3}", rd.name):
                if (rd / "DONE").exists():
                    done += 1
                else:
                    in_flight += 1
    out["rounds_done"] = done
    out["rounds_in_flight"] = in_flight
    dig = TOUR / "TRIAGE_DIGESTS.md"
    out["digests"] = (dig.read_text().count("# Triage digest")
                      if dig.exists() else 0)
    return out


# ---------------------------------------------------------------------------
# page assembly
# ---------------------------------------------------------------------------

def main():
    st = json.loads(STATE.read_text())
    now = time.strftime("%Y-%m-%d %H:%M")

    # ---- live synthetic-data landscape (fail-soft: the tracker must build
    # even when /mnt/h or the corpus tooling is unavailable) ----
    mixture_counts, mixture_rows, mix_t, mix_e = {}, [], 0, 0
    bank_rows, kinds, rules_rows, tour = [], [], [], {}
    try:
        if SFT_V7_STATS.exists():
            mixture_rows, mix_t, mix_e = gather_mixture()
            mixture_counts = {
                fam: v["train"] + v["eval"]
                for fam, v in json.loads(SFT_V7_STATS.read_text())
                ["report"]["families"].items()}
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] mixture gather failed: {e}", file=sys.stderr)
    try:
        rules_rows = gather_rules()
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] rules gather failed: {e}", file=sys.stderr)
    try:
        bank_rows = gather_bank(mixture_counts)
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] bank gather failed: {e}", file=sys.stderr)
    try:
        kinds = gather_kinds(bank_rows,
                             n_rules=len(rules_rows) or None)
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] kinds gather failed: {e}", file=sys.stderr)
    try:
        tour = gather_tournament()
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] tournament gather failed: {e}", file=sys.stderr)

    parts = []
    parts.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sepalith — experiment tracker</title>
<style>
 body{{background:#0d1117;color:#c9d1d9;font:15px/1.55 -apple-system,'Segoe UI',sans-serif;margin:0;padding:32px 16px}}
 .wrap{{max-width:1080px;margin:0 auto}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px;border-bottom:1px solid #21262d;padding-bottom:6px}}
 h3{{font-size:15px;margin:22px 0 6px;color:#8b949e;text-transform:uppercase;letter-spacing:.04em}}
 .sub{{color:#8b949e;margin-bottom:16px}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #21262d;vertical-align:top}}
 th{{color:#8b949e;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
 tr:hover{{background:#161b22}}
 .card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px 18px;margin:12px 0}}
 .note{{color:#8b949e;font-size:13px}}
 .win{{color:#2ea043;font-weight:600}}
 code{{background:#21262d;padding:1px 5px;border-radius:4px;font-size:13px}}
 .tabs{{display:flex;gap:8px;border-bottom:1px solid #21262d;margin:18px 0 0}}
 .tabbtn{{background:none;border:none;border-bottom:2px solid transparent;color:#8b949e;font:inherit;font-size:15px;padding:8px 14px;cursor:pointer}}
 .tabbtn:hover{{color:#c9d1d9}}
 .tabbtn.active{{color:#c9d1d9;border-bottom-color:#f78166}}
</style></head><body><div class="wrap">
<h1>Sepalith <span class="note">— experiment tracker</span></h1>
<div class="sub">Open, R-specialized next-edit-suggestion model. Local-first (llama.cpp/GGUF), aimed at pharma/biostat.
Generated {esc(now)}. This page is a static snapshot — it changes when the orchestrator re-uploads it.</div>
<div class="tabs">
 <button class="tabbtn active" id="tb-exp" data-tab="exp">Experiments</button>
 <button class="tabbtn" id="tb-syn" data-tab="syn">Synthetic Data</button>
</div>
<script>
 document.querySelectorAll(".tabbtn").forEach(function(b){{b.addEventListener("click", function(){{showTab(b.dataset.tab);}});}});
function showTab(id){{
   for (const t of ['exp','syn']){{
     document.getElementById('tab-'+t).style.display = (t===id)?'block':'none';
     document.getElementById('tb-'+t).classList.toggle('active', t===id);
   }}
   try{{history.replaceState(null,'','#'+id);}}catch(e){{}}
 }}
 (function(){{var h=location.hash.replace('#','');if(h==='syn')showTab('syn');}})();
</script>
<div id="tab-exp">""")

    # currently running
    parts.append("<h2>Running now</h2>")
    parts.append(table(["what", "status", "note"],
                       [[r["what"], chip(r["status"]), r.get("note", "")] for r in st["running"]]))

    # experiments
    parts.append("<h2>Experiments &amp; results</h2>")
    parts.append('<div class="note" style="margin:-4px 0 8px">newest first — the state file is chronological, the page renders reversed</div>')
    for exp in reversed(st["experiments"]):
        parts.append(f'<div class="card"><div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap">'
                     f'<div style="font-weight:600;font-size:16px">{esc(exp["name"])}</div>'
                     f'<div>{chip(exp["status"])}</div></div>'
                     f'<div class="note" style="margin:6px 0">{exp.get("blurb", "")}</div>')
        if exp.get("table"):
            parts.append(table(exp["table"].get("headers", []),
                               [[c if str(c).startswith("<") else esc(c) for c in row]
                                for row in exp["table"].get("rows", [])]))
        if exp.get("verdict"):
            parts.append(f'<div style="margin-top:8px">{exp["verdict"]}</div>')
        parts.append("</div>")

    # data
    parts.append("<h2>Data pipeline</h2>")
    parts.append(table(["family", "rows", "note"],
                       [[r["family"], r["rows"], r.get("note", "")] for r in st["data"]]))

    # decisions
    parts.append("<h2>Standing decisions &amp; rules</h2><ul>")
    for d in st["decisions"]:
        parts.append(f"<li style='margin:6px 0'>{d}</li>")
    parts.append("</ul>")

    # next
    parts.append("<h2>Next up</h2><ul>")
    for n in st["next"]:
        parts.append(f"<li style='margin:6px 0'>{n}</li>")
    parts.append("</ul>")
    parts.append("</div>")                      # close tab-exp

    # ------------------------------------------------------------------
    # TAB 2 — synthetic data landscape
    # ------------------------------------------------------------------
    parts.append('<div id="tab-syn" style="display:none">')

    parts.append("<h2>Synthetic data — landscape</h2>")
    parts.append('<div class="note">Everything on this tab is computed live at build time '
                 'from the dataset files on /mnt/h, the case specs, the transform-rule '
                 'registry and the ideation-tournament outputs — no hardcoded counts. '
                 '"new since cut" = banked rows not yet in the sft_v7 mixture.</div>')

    # (a) aggregates
    parts.append("<h3>Bank — live per-family row counts</h3>")
    if bank_rows:
        parts.append(table(
            ["family", "bank rows (live)", "in sft_v7", "new since cut",
             "authors / models", "files"],
            bank_rows))
    else:
        parts.append('<div class="note">dataset files unavailable at build time</div>')

    parts.append("<h3>sft_v7 mixture composition</h3>")
    if mixture_rows:
        parts.append(f'<div class="note">{fmt_int(mix_t)} train / {fmt_int(mix_e)} eval rows, '
                     f'{len(mixture_rows)} families — share is of the train split</div>')
        parts.append(table(["family", "train", "eval", "share"], mixture_rows))

    # (b) scenario kinds
    parts.append("<h2>Scenario kinds</h2>")
    parts.append('<div class="note">The family catalog: one-liners are the specs\' own '
                 'description sentences where a spec exists (source column), curated '
                 'otherwise. Live bank rows and sft_v7 presence per family.</div>')
    for gtitle, rows in kinds:
        parts.append(f"<h3>{esc(gtitle)}</h3>")
        parts.append(table(["family", "what it teaches", "source",
                            "bank rows", "in v7"], rows))

    # (c) rules
    n_ex = sum(1 for r in rules_rows if r.get("before") is not None)
    parts.append("<h2>Modular rules (transform registry)</h2>")
    parts.append(f'<div class="note">{len(rules_rows)} registered rules in '
                 f'<code>cases/rules/</code>; the before&#8618;after examples are each '
                 f"rule's smallest positive SELFTEST snippet, executed through the real "
                 f"detector&#8594;rewrite at dashboard build time ({n_ex}/{len(rules_rows)} "
                 f"carry examples). verify() is a pure re-derivation &#8594; deterministic "
                 f"RL reward.</div>")
    if rules_rows:
        parts.append(render_rules(rules_rows))

    # (d) tournament
    parts.append("<h2>Ideation tournament output</h2>")
    if tour:
        parts.append('<div class="note">'
                     f'{tour.get("rounds_done", 0)} recurring rounds completed '
                     f'(+{tour.get("rounds_in_flight", 0)} in flight), '
                     f'{tour.get("bank_size", 0)} banked ideas, '
                     f'{len(tour.get("build_specs", []))} build specs, '
                     f'{tour.get("digests", 0)} triage digests — '
                     'event-paced loop, band policy BUILD/BANK/RECYCLE.</div>')
        if tour.get("build_specs"):
            parts.append("<h3>Build specs (wave-2 doc)</h3>")
            parts.append(table(["round", "title", "kind", "composite", ""],
                               tour["build_specs"]))
        if tour.get("top_banked"):
            parts.append("<h3>Top banked ideas by composite</h3>")
            parts.append(table(["composite", "title", "one-liner"],
                               tour["top_banked"]))

    parts.append("</div>")                      # close tab-syn

    parts.append(f'<div class="note" style="margin-top:40px">Artifacts: research log <code>docs/research/2026-08-19-night-results.md</code> · '
                 f'architecture dossier <code>docs/research/arch-dossier-v1.md</code> (red-team pending) · '
                 f'eval results <code>experiments/eval/results_*</code> · this page: <code>experiments/dashboard/</code></div>')
    parts.append("</div></body></html>")

    OUT.write_text("".join(parts))
    print(f"built {OUT} ({OUT.stat().st_size} bytes); "
          f"synthetic tab: {len(bank_rows)} bank families, "
          f"{len(rules_rows)} rules ({n_ex} with examples), "
          f"tournament bank {tour.get('bank_size', 0)}")


if __name__ == "__main__":
    main()
