#!/usr/bin/env python3
"""Build the Sepalith program log (dashboard v2) from state_v2.json.

DESIGNER OF RECORD: muse-spark. The visual identity, layout, treatments and
ALL static copy come from spark's outputs (design_v2.json + copy_v2.json,
authored through spark_design_v2.py under the frontend-design / show-me /
humanizer / orwell-writing skills). This script only implements and
enforces: postplan sanitization, one self-contained HTML file, JS-off
readability, real data from staticdata_v2.py.

Page (fixed IA):
  masthead + stand + live-now band
  1  What happened   verdict timeline (hero)
  2  The program tree  genealogy, killed branches as specimens
  3  What is queued   grouped lists parsed from EXPERIMENT-QUEUE.md
  4  Family glossary  families + jargon
  5  Data on disk     three inventories

CLI: build_dashboard_v2.py [--out FILE] (default index.html)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import staticdata_v2 as SD                                    # noqa: E402

STATE = HERE / "state_v2.json"
DESIGN = HERE / "design_v2.json"
COPY = HERE / "copy_v2.json"
DEFAULT_OUT = HERE / "index.html"

_SAFE = ("</b>", "<b>", "</code>", "<code>", "</i>", "<i>", "</em>", "<em>",
         "</strong>", "<strong>", "<br>", "<br/>")


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def safe_render(s) -> str:
    t = esc(s)
    for tag in _SAFE:
        t = t.replace(esc(tag), tag)
    return t


def fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:                                    # noqa: BLE001
        return {}


DESIGNJ = _load(DESIGN)
COPYJ = _load(COPY)

# spark copy lookups (fallback to the static drafts when absent)
_COPY_FAM = {f["name"]: f for f in COPYJ.get("families", [])}
_COPY_JARGON = {j["term"]: j for j in COPYJ.get("jargon", [])}
_COPY_TL = {t["id"]: t for t in COPYJ.get("timeline", [])}
_COPY_TREE = {t["id"]: t.get("note", "") for t in COPYJ.get("tree", [])
              if isinstance(t, dict) and t.get("id")}
_INTROS = COPYJ.get("section_intros", {})
_LABELS = COPYJ.get("timeline_labels",
                    {"ask": "ask", "measured": "found", "implies": "call"})

# verdict-class mapping (spark: sepal = ship/live, oxblood = kill, amber =
# parked/gate, ink = info) — applied to pills, rules, dots, tree tags
_VERDICT_CLS = {"good": "v-good", "bad": "v-bad", "neutral": "v-neutral",
                "pending": "v-pending"}
_NODE_CLS = {"done": "n-done", "win": "n-win", "killed": "n-killed",
             "gate": "n-gate", "running": "n-run", "parked": "n-parked",
             "proposed": "n-proposed", "blocked": "n-blocked"}


# ---------------------------------------------------------------------------
# renderers (spark's treatments)
# ---------------------------------------------------------------------------

def render_timeline(entries: list[dict]) -> str:
    out = ['<ol class="tl">']
    for e in reversed(entries):          # state chronological; hero newest-first
        cls = _VERDICT_CLS.get(e.get("verdict_class", "neutral"), "v-neutral")
        out.append(f'<li class="tl-e {cls}">')
        out.append(f'<div class="tl-meta"><span class="tl-date">'
                   f'{esc(e.get("date", ""))}</span>'
                   f'<span class="tl-id">{esc(e.get("id", ""))}</span></div>')
        out.append(f'<p class="tl-call">{safe_render(e.get("verdict", ""))}'
                   f'. <span class="tl-what">{safe_render(e.get("title", ""))}'
                   f'</span></p>')
        for lab, key in (("ask", "ask"), ("measured", "measured"),
                         ("implies", "implies")):
            v = e.get(key)
            if v:
                out.append(f'<p class="tl-line tl-{key}">'
                           f'<span class="lab">{esc(_LABELS.get(lab, lab))}</span>'
                           f'{safe_render(v)}</p>')
        out.append("</li>")
    out.append("</ol>")
    return "".join(out)


def _tree_node(node: dict) -> str:
    cls = _NODE_CLS.get(node.get("status", "proposed"), "n-proposed")
    note = _COPY_TREE.get(node["id"], node.get("note", ""))
    kids = node.get("children") or []
    out = (f'<li><div class="node {cls}">'
           f'<span class="node-tag">{esc(node.get("status", ""))}</span>')
    if node.get("status") == "killed":
        out += f'<span class="node-x">×</span>'
    out += f'<span class="node-label">{esc(node.get("label", ""))}</span>'
    if note:
        out += f'<span class="node-note">{esc(note)}</span>'
    out += "</div>"
    if kids:
        out += "<ul>"
        for k in kids:
            out += _tree_node(k)
        out += "</ul>"
    out += "</li>"
    return out


def render_tree() -> str:
    out = ['<div class="tree"><ul class="tree-root">']
    for n in SD.SKILL_TREE:
        out.append(_tree_node(n))
    out.append("</ul></div>")
    return "".join(out)


def render_queue(q: dict) -> str:
    if not q:
        return '<p class="empty">Queue unavailable this cycle.</p>'
    out = []
    if q.get("synced"):
        out.append(f'<p class="src">Parsed from docs/EXPERIMENT-QUEUE.md. '
                   f'Last sync: {esc(q["synced"])}</p>')
    for title, rows in (("Running", q.get("running", [])),
                        ("Parked", q.get("parked", [])),
                        ("Proposed", q.get("proposed", []))):
        out.append(f"<h3>{esc(title)}</h3>")
        if not rows:
            out.append(f'<p class="empty">Nothing {title.lower()}.</p>')
            continue
        out.append("<ul class='qlist'>")
        for r in rows:
            out.append(
                f'<li><span class="q-id">{esc(r.get("id", ""))}</span>'
                f'<span class="q-item">{safe_render(r.get("item", ""))}</span>'
                f'<span class="q-why">{safe_render(r.get("why", ""))}</span>'
                f"</li>")
        out.append("</ul>")
    wc = q.get("work_count")
    if isinstance(wc, int):
        out.append(f'<p class="src">Engineering backlog (queue section 4): '
                   f'{wc} W-items, not listed row by row.</p>')
    return "".join(out)


def render_glossary(bank: dict) -> str:
    out = []
    groups = COPYJ.get("group_titles") or [g[0] for g in SD.GLOSSARY_GROUPS]
    for (gtitle, fams), spark_title in zip(SD.GLOSSARY_GROUPS,
                                           list(groups) + [None] * 9):
        out.append(f"<h3>{esc(spark_title or gtitle)}</h3><dl class='gl'>")
        for f in fams:
            cf = _COPY_FAM.get(f["name"], {})
            plain = cf.get("decode", f["plain"])
            example = cf.get("example", f.get("example", ""))
            rows_txt = _gloss_rows(f["rows_key"], bank)
            out.append(f'<div class="gl-e"><dt><code>{esc(f["name"])}</code>'
                       f'<span class="gl-rows">{esc(rows_txt)}</span></dt>'
                       f'<dd class="serif">{esc(plain)}')
            if example:
                out.append(f'<pre class="rex">{esc(example)}</pre>')
            out.append(f'<span class="src">source: {esc(f["source"])}</span>'
                       f"</dd></div>")
        out.append("</dl>")
    return "".join(out)


def _gloss_rows(key: str, bank: dict) -> str:
    v = bank.get(key)
    if isinstance(v, int):
        return f"{fmt_int(v)} rows on disk"
    if key in SD.ROWS_PREFIX:
        pre = sum(x for k, x in bank.items() if k.startswith(key))
        if pre:
            return f"{fmt_int(pre)} rows on disk"
    return SD.ROWS_FALLBACK.get(key, "count unavailable")


def render_jargon() -> str:
    out = ['<table class="jx"><thead><tr><th>term</th><th>plain form</th>'
           '<th>what it means here</th></tr></thead><tbody>']
    for term, plain, definition in SD.JARGON:
        cj = _COPY_JARGON.get(term, {})
        out.append(f"<tr><td><code>{esc(term)}</code></td>"
                   f"<td>{esc(cj.get('plain', plain))}</td>"
                   f"<td class='serif'>{esc(cj.get('definition', definition))}"
                   "</td></tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _sft_table(inv: dict) -> str:
    rows = inv.get("synthetic", {}).get("mixtures") or []
    if not rows:
        return '<p class="empty">Mixture stats unavailable at build time.</p>'
    t = ["<div class=\"inv-scroll\"><table><caption>datasets/ sft_v1 to sft_v8_2</caption>"
         "<thead><tr><th>cut</th><th>train rows</th><th>eval rows</th>"
         "<th>families</th></tr></thead><tbody>"]
    for r in rows:
        t.append(f"<tr><td><code>{esc(r['name'])}</code></td>"
                 f'<td class="num">{fmt_int(r["train"])}</td>'
                 f'<td class="num">{fmt_int(r["eval"])}</td>'
                 f'<td class="num">{r["families"] or "n/a"}</td></tr>')
    t.append("</tbody></table></div>")
    return "".join(t)


def render_inventory(inv: dict) -> str:
    out = []
    real, syn, rl = inv.get("real", {}), inv.get("synthetic", {}), inv.get("rl", {})

    # --- 5a real-world corpus
    out.append('<div class="inv3"><section>')
    out.append("<h3>5a Real-world corpus</h3>")
    rows = []
    nd = real.get("normalized_dirs", -1)
    mp, mb = real.get("manifest_packages"), real.get("manifest_bytes")
    if nd >= 0:
        rows.append(["/mnt/h/sepalith/normalized",
                     f"{fmt_int(nd)} package trees",
                     (f"ingest log: {fmt_int(mp)} packages, "
                      f"{mb/1e9:.2f} GB" if mp else "manifest unavailable")])
    ps = real.get("packages_shards", -1)
    if ps >= 0:
        rows.append(["datasets/packages", f"{fmt_int(ps)} shards",
                     "one per package; removal = deleting the shard"])
    a2 = real.get("a2_strata") or []
    if a2:
        tot = sum(s["tokens"] for s in a2)
        det = ", ".join(f"{s['stream']} {s['tokens']/1e6:.0f}M" for s in a2)
        rows.append(["a2/r strata (local)",
                     f"{len(a2)} streams, {fmt_int(tot)} tok", det])
        rows.append(["A2 data program (full)", "13/13 strata, 7.55B tok",
                     "close-out 2026-08-31; HF pretraining/ (34 files, 30.3GB); local a2/r is the mixture-smoke subset"])
    ss = real.get("stack_staging_items", -1)
    if ss >= 0:
        rows.append(["stack_staging", f"{ss} top-level items",
                     "Stack v2 R subset: ~141k license-filtered repos / 22.4GB (queue section 5)"])
    rows.append(["git/ mirrors", "2,586 repos, ~91GB",
                 "queue W6: biggest unqueued R lever (license audit + holdout first)"])
    rows.append(["tarballs/", "raw CRAN tarballs offline",
                 ".tarball_sha256 links shards to originals"])
    out.append('<div class="inv-scroll"><table><caption>/mnt/h/sepalith and datasets/</caption>'
               "<thead><tr><th>where</th><th>counted</th><th>note</th></tr>"
               "</thead><tbody>"
               + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r)
                         + "</tr>" for r in rows)
               + "</tbody></table></div></section>")

    # --- 5b prepared synthetic
    out.append("<section><h3>5b Prepared synthetic</h3>")
    out.append(_sft_table(inv))
    ast = syn.get("astfim")
    notes = []
    if ast:
        notes.append(f"astfim_v1: {fmt_int(ast['rows'])} rows / "
                     f"{ast['est_tokens']/1e6:.1f}M tok from "
                     f"{fmt_int(ast['packages'])} packages; eval holdout "
                     f"{fmt_int(ast['eval_pkgs'])} packages.")
    if isinstance(syn.get("astfim_random_train"), int):
        notes.append(f"astfim_random_v1: {fmt_int(syn['astfim_random_train'])} "
                     "train rows.")
    r1, r2 = syn.get("rl_refine_v1"), syn.get("rl_refine_v2")
    if isinstance(r1, int) and isinstance(r2, int):
        notes.append(f"RL refinement rows: v1 {fmt_int(r1)}, v2 {fmt_int(r2)}.")
    pools = syn.get("pools") or {}
    if pools:
        notes.append("Other pools: "
                     + ", ".join(f"{k} {fmt_int(v)}" for k, v in pools.items())
                     + ".")
    bank = syn.get("bank_families") or {}
    if bank:
        top = sorted(bank.items(), key=lambda kv: -kv[1])[:12]
        notes.append("Family bank (cases_v1 + scenarios_v1): "
                     f"{fmt_int(syn.get('bank_total', 0))} rows, "
                     f"{len(bank)} families; largest "
                     + ", ".join(f"{k} {fmt_int(v)}" for k, v in top) + ".")
    for n in notes:
        out.append(f'<p class="src">{esc(n)}</p>')
    out.append("</section>")

    # --- 5c RL environments
    out.append("<section><h3>5c RL environments</h3>")
    quotas = rl.get("quotas") or {}
    if quotas:
        out.append('<p class="src">rl_smoke TRAIN-split quotas: '
                   + esc(", ".join(f"{k} {v}" for k, v in quotas.items()))
                   + '.</p>')
        quotas2 = rl.get("quotas_run2") or {}
        if quotas2:
            out.append('<p class="src">Run-2 no-op arm: '
                       + esc(", ".join(f"{k} {v}" for k, v in quotas2.items()))
                       + " (no_op about 30 percent of the draw).</p>")
    priors = rl.get("priors") or {}
    if priors:
        out.append('<div class="inv-scroll"><table><caption>scenario pools, pass-rate priors</caption>'
                   "<thead><tr><th>family</th><th>prior</th></tr></thead>"
                   "<tbody>"
                   + "".join(f"<tr><td><code>{esc(k)}</code></td>"
                             f"<td>{esc(v)}</td></tr>" for k, v in priors.items())
                   + "</tbody></table></div>")
    for c in rl.get("env_classes") or []:
        out.append(f'<p class="src">{esc(c)}</p>')
    if rl.get("el_tiers"):
        out.append(f'<p class="src">E1 tiers: {esc(rl["el_tiers"])}</p>')
    if rl.get("difficulty_gate"):
        out.append(f'<p class="src">{esc(rl["difficulty_gate"])}</p>')
    out.append('<p class="src">Counts from disk at build time, cached by '
               'file mtime (results/inventory_cache_v2.json).</p>')
    out.append("</section></div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

def _hex(name: str, default: str) -> str:
    for p in DESIGNJ.get("palette", []):
        if p.get("name", "").lower().find(name.lower()) >= 0:
            return p.get("hex", default)
    return default


PAPER = _hex("paper", "#F7F8F5")
INK = _hex("ink", "#22333B")
SEPAL = _hex("sepal", "#2E6E4E")
OXBLOOD = _hex("oxblood", "#A63A4A")
AMBER = _hex("amber", "#9A6B0F")
RBLUE = _hex("blue", "#2F5DA3")
SAGE = _hex("sage", "#E3E9E2")

SERIF = (DESIGNJ.get("type", {}).get("families") or [{}])[0].get(
    "stack", "Charter, 'Bitstream Charter', Georgia, serif")
MONO = ((DESIGNJ.get("type", {}).get("families") or [{}, {}])[-1].get(
    "stack", "ui-monospace, Menlo, Consolas, monospace"))

CSS = f"""
:root{{
 --paper:{PAPER}; --ink:{INK}; --sepal:{SEPAL}; --oxblood:{OXBLOOD};
 --amber:{AMBER}; --rblue:{RBLUE}; --sage:{SAGE};
 --rule:{INK}33; --hair:{INK}22; --faint:{INK}77;
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
@media (prefers-reduced-motion: reduce){{html{{scroll-behavior:auto}}}}
body{{margin:0;background:var(--paper);color:var(--ink);
font:13.5px/1.55 {MONO.split('monospace')[0]}monospace}}
.wrap{{max-width:880px;margin:0 auto;padding:0 20px 70px}}
.serif{{font-family:{SERIF};font-size:15px;line-height:1.55}}
h1,h2,h3{{font-family:{SERIF};font-weight:600;line-height:1.2;margin:0}}
h1{{font-size:27px;letter-spacing:.2px}}
h2{{font-size:21px;margin:44px 0 4px;padding:10px 0 6px;
border-top:2px solid var(--ink);scroll-margin-top:64px}}
h2 .sn{{color:var(--rblue);margin-right:8px}}
h3{{font-size:15.5px;margin:22px 0 6px}}
p{{margin:7px 0}}
a{{color:var(--rblue)}}
a:focus-visible,button:focus-visible,input:focus-visible{{
 outline:3px solid var(--rblue);outline-offset:2px}}
code{{font-family:{MONO.split('monospace')[0]}monospace;font-size:.95em;
background:var(--sage);padding:0 4px;border-radius:3px}}
pre{{font-size:12.5px;background:var(--sage);border-left:3px solid var(--rblue);
padding:9px 12px;margin:7px 0;overflow-x:auto;white-space:pre-wrap}}
table{{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:12.5px}}
caption{{text-align:left;color:var(--faint);padding-bottom:3px;font-size:12px}}
th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid var(--hair);
vertical-align:top}}
th{{font-weight:600;font-size:12px}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.sub{{color:var(--faint);margin:4px 0 0;max-width:70ch}}
.src{{color:var(--faint);font-size:12px;margin:5px 0}}
.empty{{color:var(--faint);font-style:italic}}
.upd{{color:var(--faint);font-size:12px;margin-top:36px;
border-top:1px solid var(--hair);padding-top:10px}}
/* masthead */
.mast{{display:flex;justify-content:space-between;align-items:baseline;
flex-wrap:wrap;gap:8px;padding:30px 0 10px}}
.mast .when{{color:var(--faint);font-size:12px}}
.stand{{font-family:{SERIF};font-size:19px;line-height:1.5;margin:14px 0 10px;
max-width:62ch;scroll-margin-top:64px}}
/* live-now band */
.live{{background:var(--sage);border-top:1px solid var(--hair);
border-bottom:1px solid var(--hair);margin:14px -20px 0;padding:10px 20px}}
.live-h{{color:var(--faint);font-size:12px;margin-bottom:4px}}
.live ul{{list-style:none;margin:0;padding:0}}
.live li{{display:flex;gap:12px;flex-wrap:wrap;padding:3px 0;
border-bottom:1px solid var(--hair)}}
.live li:last-child{{border-bottom:none}}
.dot{{width:8px;height:8px;border-radius:50%;background:var(--rblue);
display:inline-block;margin:5px 6px 0 0;flex:none}}
@media (prefers-reduced-motion: no-preference){{
.dot{{animation:pulse 2.4s ease-in-out infinite}}}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.35}}}}
.lv-what{{flex:1 1 46%}}
.lv-owner{{color:var(--rblue)}}
.lv-note{{color:var(--faint);flex:1 1 100%;padding-left:14px}}
/* nav */
nav.top{{position:sticky;top:0;background:var(--paper);
border-bottom:1px solid var(--rule);z-index:5;padding:7px 0;margin-top:16px}}
nav.top .wrap{{display:flex;flex-wrap:wrap}}
nav.top a{{margin-right:16px;text-decoration:none;color:var(--faint);
font-size:12.5px;padding:2px 0;border-bottom:2px solid transparent}}
nav.top a.on{{color:var(--ink);border-bottom-color:var(--rblue)}}
/* 1 timeline (hero) */
.filterbox{{display:none;margin:8px 0 2px}}
.filterbox input{{font:inherit;width:100%;max-width:420px;padding:5px 8px;
border:1px solid var(--rule);background:var(--paper);color:var(--ink)}}
ol.tl{{list-style:none;margin:14px 0;padding:0}}
.tl-e{{border-left:4px solid var(--faint);margin:0 0 26px;padding:0 0 0 16px}}
.tl-e.v-good{{border-left-color:var(--sepal)}}
.tl-e.v-bad{{border-left-color:var(--oxblood);
background:linear-gradient(to right,{OXBLOOD}0D,transparent 60%)}}
.tl-e.v-pending{{border-left-color:var(--amber)}}
.tl-meta{{color:var(--faint);font-size:12px;margin-bottom:2px}}
.tl-meta .tl-id{{color:var(--rblue);margin-left:8px}}
.tl-call{{font-family:{SERIF};font-size:18.5px;line-height:1.4;
margin:0 0 6px;max-width:58ch}}
.tl-call .tl-what{{color:var(--faint)}}
.tl-line{{margin:3px 0;color:var(--ink);font-size:12.5px;max-width:80ch}}
.lab{{color:var(--faint);display:inline-block;min-width:7ch}}
.tl-measured .lab,.tl-implies .lab{{min-width:7ch}}
/* 2 tree */
.tree ul{{list-style:none;margin:0;padding:0 0 0 24px;
border-left:2px solid var(--ink)}}
.tree ul.tree-root{{padding:0;border-left:none}}
.tree li{{margin:9px 0;position:relative}}
.tree ul li:before{{content:"";position:absolute;left:-24px;top:15px;
width:18px;border-top:2px solid var(--ink)}}
.tree ul li.n-killed-row:before{{border-top-style:dashed;
border-top-color:{OXBLOOD}88}}
.node{{display:inline-block;max-width:100%;background:var(--paper);
border:1px solid var(--ink);border-radius:6px;padding:5px 11px;
border-left-width:4px;border-left-color:var(--ink)}}
.node-tag{{display:inline-block;font-size:10.5px;border:1px solid;
border-radius:8px;padding:0 6px;margin-right:8px;vertical-align:1px}}
.node-label{{font-weight:600;font-size:12.5px}}
.node-note{{display:block;color:var(--faint);font-size:12px;margin-top:2px;
max-width:74ch}}
.n-win{{border-left-color:var(--sepal)}}
.n-win .node-tag{{color:var(--sepal);border-color:var(--sepal)}}
.n-killed{{border-style:dashed;border-color:{OXBLOOD}AA;
border-left-style:dashed;border-left-color:{OXBLOOD}AA;background:{OXBLOOD}08}}
.n-killed .node-tag{{color:var(--oxblood);border-color:{OXBLOOD}AA}}
.n-killed .node-label{{text-decoration:line-through;
text-decoration-color:{OXBLOOD}CC;text-decoration-thickness:1px}}
.n-killed .node-note{{color:{OXBLOOD}99}}
.node-x{{color:var(--oxblood);font-weight:700;margin-right:6px}}
.n-gate{{border-style:double;border-width:3px;border-color:var(--amber)}}
.n-gate .node-tag{{color:var(--amber);border-color:var(--amber)}}
.n-run{{border-left-color:var(--rblue)}}
.n-run .node-tag{{color:var(--rblue);border-color:var(--rblue)}}
.n-run .node-label{{font-weight:700}}
.n-parked .node-tag,.n-proposed .node-tag{{color:var(--amber);
border-color:{AMBER}AA}}
.n-blocked{{border-color:{INK}66;border-left-color:{INK}66}}
.n-blocked .node-tag{{color:var(--oxblood);border-color:{INK}66}}
/* 3 queue lists */
.qlist{{list-style:none;margin:6px 0 16px;padding:0}}
.qlist li{{display:flex;gap:10px;flex-wrap:wrap;padding:5px 0;
border-bottom:1px solid var(--hair);font-size:12.5px}}
.q-id{{color:var(--rblue);flex:0 0 7ch}}
.q-item{{flex:1 1 40%}}
.q-why{{color:var(--faint);flex:1 1 100%;padding-left:0}}
@media (min-width:760px){{.q-why{{flex:1 1 30%;padding-left:0}}}}
/* 4 glossary */
dl.gl{{margin:8px 0}}
.gl-e{{border-left:2px solid var(--rule);padding:2px 0 6px 13px;margin:0 0 13px}}
.gl-e dt{{font-weight:600;font-size:13px}}
.gl-rows{{color:var(--faint);font-size:11.5px;font-weight:400;margin-left:10px}}
.gl-e dd{{margin:2px 0 0}}
.rex{{margin:6px 0 4px;min-width:0}}
/* 5 inventory */
.inv3{{display:flex;flex-wrap:wrap;gap:22px}}
.inv3 section{{flex:1 1 300px;min-width:0}}
.inv-scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
@media (max-width:480px){{
 .tree ul{{padding-left:16px}}
 .tree ul li:before{{left:-16px;width:12px}}}}
/* footer */
.foot{{margin-top:46px;border-top:2px solid var(--ink);padding-top:8px}}
.foot ul{{margin:6px 0 16px;padding-left:18px}}
.foot li{{margin:5px 0;font-size:12.5px}}
"""

JS = """
(function(){
 var links=[].slice.call(document.querySelectorAll('nav.top a'));
 var map={};
 links.forEach(function(a){var id=a.getAttribute('href').slice(1);
  var el=document.getElementById(id);if(el)map[id]=a;});
 if('IntersectionObserver' in window){
  var obs=new IntersectionObserver(function(es){
   es.forEach(function(e){
    if(e.isIntersecting){
     links.forEach(function(a){a.classList.remove('on');});
     var a=map[e.target.id];if(a)a.classList.add('on');
    }});
  },{rootMargin:'-15% 0px -75% 0px'});
  Object.keys(map).forEach(function(id){
   var el=document.getElementById(id);if(el)obs.observe(el);});
 }
 // filter-as-you-type over the verdict timeline (progressive only)
 var box=document.querySelector('.filterbox');
 var tl=document.querySelector('.tl');
 if(box&&tl&&'querySelectorAll' in document){
  box.style.display='block';
  var input=box.querySelector('input');
  var entries=[].slice.call(tl.querySelectorAll('.tl-e'));
  var raf=null;
  input.addEventListener('input',function(){
   if(raf)cancelAnimationFrame(raf);
   raf=requestAnimationFrame(function(){
    var q=input.value.trim().toLowerCase();
    entries.forEach(function(e){
     e.style.display=(!q||e.textContent.toLowerCase().indexOf(q)>=0)?'':'none';
    });
   });
  });
 }
 // copy buttons on R examples (added here so JS-off keeps a clean page)
 [].slice.call(document.querySelectorAll('pre.rex')).forEach(function(pre){
  var b=document.createElement('button');
  b.type='button';b.textContent='copy';
  b.style.cssText='font:11px inherit;padding:1px 8px;margin-left:8px;'+
   'border:1px solid var(--rule);background:var(--paper);color:inherit;'+
   'border-radius:6px;cursor:pointer;vertical-align:1px';
  b.addEventListener('click',function(){
   try{navigator.clipboard.writeText(pre.textContent);}catch(e){}
   b.textContent='copied';setTimeout(function(){b.textContent='copy';},1200);
  });
  var head=previousLabel(pre);
  if(head){head.appendChild(document.createTextNode(' '));head.appendChild(b);}
 });
 function previousLabel(pre){
  // attach the button next to the glossary rows count (dt) of this example
  var e=pre.closest('.gl-e');return e?e.querySelector('dt'):null;
 }
})();
"""


def build(out_path: Path) -> str:
    st = json.loads(STATE.read_text())
    inv = SD.inventory()
    now = time.strftime("%Y-%m-%d %H:%M")
    meta = st.get("meta", {})
    bank = inv.get("synthetic", {}).get("bank_families") or {}
    title = COPYJ.get("title", "Sepalith")
    subtitle = COPYJ.get("subtitle", "")
    stand = meta.get("stand", COPYJ.get("stand", ""))
    live_label = COPYJ.get("live_label", "Live now")
    footer_line = COPYJ.get("footer_line", "")

    parts = []
    parts.append(
        f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style></head><body>
<div class="wrap">
<div class="mast"><h1>{esc(title)}</h1>
<span class="when">rebuilt {esc(now)}</span></div>
<p class="sub">{esc(subtitle)}</p>
<p class="stand" id="now">{safe_render(stand)}</p>
<div class="live"><div class="live-h">{esc(live_label)}</div><ul>""")

    for r in st.get("running") or []:
        eta = f' · eta {esc(r.get("eta", ""))}' if r.get("eta") else ""
        parts.append(
            f'<li><span class="dot"></span>'
            f'<span class="lv-what">{safe_render(r.get("what", ""))}</span>'
            f'<span class="lv-owner">{esc(r.get("owner", ""))}</span>'
            f'<span class="lv-note">{safe_render(r.get("note", ""))}'
            f'{eta}</span></li>')
    if not (st.get("running") or []):
        parts.append('<li><span class="lv-note">Nothing reported in flight.'
                     "</span></li>")
    parts.append("</ul></div>")

    parts.append("""
<nav class="top"><div class="wrap">
<a href="#now">Now</a>
<a href="#happened">1 What happened</a>
<a href="#tree">2 Program tree</a>
<a href="#queued">3 Queued</a>
<a href="#glossary">4 Glossary</a>
<a href="#data">5 Data on disk</a>
</div></nav>""")

    parts.append('<h2 id="happened"><span class="sn">1</span>What happened</h2>')
    parts.append(f'<p class="src">{esc(_INTROS.get("happened", ""))}</p>')
    parts.append('<div class="filterbox"><input type="text" '
                 'aria-label="filter verdicts" placeholder="filter verdicts">'
                 "</div>")
    parts.append(render_timeline(st.get("timeline") or []))

    parts.append('<h2 id="tree"><span class="sn">2</span>The program tree</h2>')
    parts.append(f'<p class="src">{esc(_INTROS.get("tree", ""))}</p>')
    parts.append(render_tree())

    parts.append('<h2 id="queued"><span class="sn">3</span>What is queued</h2>')
    parts.append(f'<p class="src">{esc(_INTROS.get("queued", ""))}</p>')
    parts.append(render_queue(st.get("queue") or {}))

    parts.append('<h2 id="glossary"><span class="sn">4</span>Family glossary'
                 "</h2>")
    parts.append(f'<p class="src">{esc(_INTROS.get("glossary", ""))}</p>')
    parts.append(render_glossary(bank))
    parts.append("<h3>Jargon used on this page</h3>")
    parts.append(render_jargon())

    parts.append('<h2 id="data"><span class="sn">5</span>Data on disk</h2>')
    parts.append(f'<p class="src">{esc(_INTROS.get("data", ""))}</p>')
    parts.append(render_inventory(inv))

    parts.append('<div class="foot"><h3>Standing decisions</h3><ul>')
    for d in st.get("decisions") or []:
        parts.append(f"<li>{safe_render(d)}</li>")
    parts.append("</ul><h3>Next up</h3><ul>")
    for n in st.get("next") or []:
        parts.append(f"<li>{safe_render(n)}</li>")
    parts.append("</ul></div>")

    parts.append(
        f'<p class="upd">{esc(footer_line)} Built {esc(now)} by '
        f"experiments/dashboard/build_dashboard_v2.py: state_v2.json "
        f"(muse-spark editorial patches, validated) + design_v2.json and "
        f"copy_v2.json (muse-spark, designer of record) + staticdata_v2.py "
        f"(repo ground truth). Verdict sources: EXPERIMENT-QUEUE.md, "
        f"base_bakeoff/RESULTS.md, poc_diff/RESULTS.md, TU1_RESULTS.md, "
        f"rl/O3_S0_RESULTS.md, p12-roofline-results.md, comms/board.md.</p>")

    parts.append(f"<script>{JS}</script></div></body></html>")

    html_doc = "".join(parts)
    out_path.write_text(html_doc)
    n_tl = len(st.get("timeline") or [])
    return (f"built {out_path} ({out_path.stat().st_size} bytes); "
            f"timeline {n_tl}; bank {len(bank)} families; "
            f"design tokens from design_v2.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output HTML path (default index.html)")
    args = ap.parse_args()
    print(build(Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
