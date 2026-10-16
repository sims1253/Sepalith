#!/usr/bin/env python3
"""Report generation for the typing-RL PoC: RESULTS.md + a fully
self-contained dashboard.html (no CDN, inline data, hand-rolled SVG)."""
from __future__ import annotations

import json
from pathlib import Path


def _by_task_round(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        out.setdefault(r["task_id"], {}).setdefault(r.get("round", 0),
                                                    []).append(r)
    return out


def _best(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r.get("scalar_total", -9))


def write_results_md(rows: list[dict], meta: dict, path: Path) -> None:
    lines = [
        "# Typing-RL PoC — results",
        "",
        f"Generated {meta['generated_at']} · {meta['rounds']} rounds × "
        f"{meta['rollouts']} rollouts · typer glm-5.3 · decision tier "
        f"`{meta['d3_backend']}` · wall {meta['wall_s']}s · glm usage "
        f"{meta['glm_usage'].get('prompt_tokens', 0)}p/"
        f"{meta['glm_usage'].get('completion_tokens', 0)}c tokens "
        f"({meta['glm_usage'].get('calls', 0)} calls)",
        "",
    ]
    for task_id, rounds in _by_task_round(rows).items():
        all_rows = [r for rr in rounds.values() for r in rr]
        best = _best(all_rows)
        lines += [f"## {task_id}", "",
                  "| round | mean scalar | mean corr | mean perf | "
                  "mean proc |", "|---|---|---|---|---|"]
        for rnd in sorted(rounds):
            rs = rounds[rnd]
            def m(k, d=0.0):
                vs = [r.get(k, d) for r in rs if r.get(k) is not None]
                return sum(vs) / len(vs) if vs else d
            lines.append(
                f"| {rnd} | {m('scalar_total'):+.3f} | {m('correctness'):.2f} "
                f"| {m('perf'):+.2f} | {m('process_reward'):+.3f} |")
        speed = (f"{best['base_s'] / best['bench_s']:.1f}x baseline"
                 if best.get("base_s") and best.get("bench_s") else "n/a")
        fails = [t["name"] for t in best.get("test_rows", [])
                 if not t["pass"]]
        j = best.get("glm_judge", {})
        lines += [
            "",
            f"Best: scalar {best.get('scalar_total'):+.3f}, correctness "
            f"{best.get('correctness', 0):.2f}, perf {best.get('perf')}, "
            f"{speed}, tests failed: {fails or 'none'}",
        ]
        if j and not j.get("parse_error"):
            lines.append(f"Judge: elegance {j.get('elegance')}/5, "
                         f"r_idiom {j.get('r_idiom')}/5, reward_hacking "
                         f"{j.get('reward_hacking')}/5 — {j.get('notes', '')}")
        lines += ["", "```r", best.get("code", ""), "```", ""]
    lines += [
        "## Task-difficulty admission (wiki advantage-collapse)", "",
        "Per-task mean visible correctness across all rollouts vs the "
        "informative band [0.25, 0.75]; GRPO admission should follow this, "
        "not intuition.", "",
        "| task | mean corr | mean heldout | mean rh_gap | verdict |",
        "|---|---|---|---|---|",
    ]
    for task_id in dict.fromkeys(r["task_id"] for r in rows):
        rs = [r for r in rows if r["task_id"] == task_id]

        def m(k):
            vs = [r[k] for r in rs if r.get(k) is not None]
            return sum(vs) / len(vs) if vs else None

        mc = m("correctness") or 0.0
        verdict = ("informative" if 0.25 <= mc <= 0.75 else
                   "saturating — retire/upgrade" if mc > 0.75 else
                   "too hard — split or scaffold")
        fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {task_id} | {fmt(mc)} | {fmt(m('heldout_correctness'))} "
                     f"| {fmt(m('rh_gap'))} | {verdict} |")
    # judge-only gain flag (wiki reward-hacking §Prime Agent): rounds whose
    # scalar rose while execution did not move are proxy-gain suspects
    flags = []
    for task_id, rounds in _by_task_round(rows).items():
        rds = sorted(rounds)

        def m(rr, k):
            vs = [r[k] for r in rr if r.get(k) is not None]
            return sum(vs) / len(vs) if vs else 0.0

        for a, b in zip(rds, rds[1:]):
            ds = m(rounds[b], "scalar_total") - m(rounds[a], "scalar_total")
            dc = m(rounds[b], "correctness") - m(rounds[a], "correctness")
            if ds > 0.03 and dc <= 0:
                flags.append(f"{task_id} r{a}→r{b} "
                             f"(Δscalar {ds:+.3f}, Δcorr {dc:+.2f})")
    lines += ["", "## Judge-only gain flags", ""]
    lines.append("; ".join(flags) if flags else "none — every round-over-"
                 "round scalar gain came with an execution gain")
    lines.append("")
    skipped = sum(1 for r in rows if r.get("cascade_skipped_glm"))
    dead = sum(1 for r in rows if r.get("skip_reason") == "dead")
    clean = sum(1 for r in rows if r.get("skip_reason") == "clean")
    agrees = [r["judge_agreement"] for r in rows
              if r.get("judge_agreement") is not None]
    agree_line = (f"{sum(agrees) / len(agrees):.3f} mean over {len(agrees)} "
                  "double-judged rollouts"
                  if agrees else "no double-judged rollouts recorded")
    lines += [
        "## Cascade", "",
        f"- glm-5.3 big-judge calls routed off: {skipped}/{len(rows)} "
        f"({100 * skipped / max(1, len(rows)):.0f}% — dead {dead}, "
        f"confident-clean {clean}; glm-5.3-flash rubric ran on all)",
        f"- glm-5.3 ↔ glm-5.3-flash style agreement: {agree_line} "
        "(multi-model-review ensemble directive)",
    ]
    if any(r.get("route_replay") for r in rows):
        rsaved = sum(1 for r in rows
                     if r.get("route_replay") not in (None, "run"))
        lines.append(
            f"- fixed-router replay (d3 score-normalization fix applied "
            f"post-run): {rsaved}/{len(rows)} big-judge calls would have "
            f"been routed off — see route_replay column / route_replay.py")
    lines += [
        f"- length audit (reward-hacking-equilibrium Dᵢ): LOC per task "
        + ", ".join(
            f"{t} {sorted(r.get('code_lines', 0) for r in rows if r['task_id'] == t)}"
            for t in dict.fromkeys(r["task_id"] for r in rows))
        + " — check reward-vs-length correlation before trusting "
        "the scalar",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# dashboard (self-contained)
# ---------------------------------------------------------------------------

_DASH = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Typing-RL PoC dashboard</title>
<style>
 :root { --ink:#1a2233; --dim:#5b6577; --line:#e3e7ee; --bg:#f7f8fb;
         --acc:#3b6ef5; --good:#1f9d55; --bad:#d64545; }
 * { box-sizing:border-box; }
 body { font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
        color:var(--ink); background:var(--bg); margin:0; padding:28px; }
 h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
 .meta { color:var(--dim); font-size:12.5px; margin-bottom:18px; }
 .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
         gap:18px; }
 .card { background:#fff; border:1px solid var(--line); border-radius:10px;
         padding:14px 16px; }
 .card h3 { margin:0 0 8px; font-size:13.5px; }
 pre { background:#0f1522; color:#dbe4f5; padding:12px; border-radius:8px;
       overflow:auto; font-size:12px; line-height:1.45; }
 table { border-collapse:collapse; width:100%; font-size:12.5px; }
 td,th { border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; }
 th { color:var(--dim); font-weight:600; }
 .pos { color:var(--good); } .neg { color:var(--bad); }
 .pill { display:inline-block; background:#eef2fd; color:var(--acc);
         border-radius:20px; padding:1px 10px; font-size:11.5px; margin-right:6px; }
 svg text { font:11px ui-sans-serif,system-ui; fill:var(--dim); }
</style></head><body>
<h1>Typing-RL PoC — agent types R code · decision tier + glm-5.3 judge</h1>
<div class="meta" id="meta"></div>
<div id="tasks"></div>
<h2>Reward-cascade ledger</h2>
<div class="card" id="ledger"></div>
<script>
const ROWS = __ROWS__;
const META = __META__;
document.getElementById('meta').innerHTML =
  `${META.rounds} rounds × ${META.rollouts} rollouts · typer glm-5.3 · ` +
  `decision tier <b>${META.d3_backend}</b> · wall ${META.wall_s}s · ` +
  `glm ${META.glm_usage.prompt_tokens}p/${META.glm_usage.completion_tokens}c ` +
  `tokens over ${META.glm_usage.calls} calls`;

function esc(s){return (s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function f3(x){return (x==null)?'—':(x>0?'+':'')+x.toFixed(3);}

function curve(taskId){
  const rs = ROWS.filter(r=>r.task_id===taskId);
  const rounds = [...new Set(rs.map(r=>r.round))].sort();
  const pts = rounds.map(rd=>{
    const g = rs.filter(r=>r.round===rd);
    return g.reduce((a,r)=>a+r.scalar_total,0)/g.length;
  });
  const W=320,H=150,P=28, min=Math.min(-0.2,...pts), max=Math.max(0.4,...pts+0.2);
  const x=i=>P+i*(W-2*P)/Math.max(1,rounds.length-1);
  const y=v=>H-P-(v-min)*(H-2*P)/(max-min);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  s+=`<line x1="${P}" y1="${y(0)}" x2="${W-P}" y2="${y(0)}" stroke="#e3e7ee"/>`;
  const path=pts.map((v,i)=>`${i?'L':'M'}${x(i)},${y(v)}`).join('');
  s+=`<path d="${path}" fill="none" stroke="var(--acc)" stroke-width="2"/>`;
  pts.forEach((v,i)=>{s+=`<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="var(--acc)"/>`;
    s+=`<text x="${x(i)}" y="${y(v)-8}" text-anchor="middle">${f3(v)}</text>`;});
  rounds.forEach((rd,i)=>s+=`<text x="${x(i)}" y="${H-8}" text-anchor="middle">r${rd}</text>`);
  return s+'</svg>';
}

const tasksDiv=document.getElementById('tasks');
[...new Set(ROWS.map(r=>r.task_id))].forEach(taskId=>{
  const rs=ROWS.filter(r=>r.task_id===taskId);
  const best=rs.reduce((a,b)=>b.scalar_total>a.scalar_total?b:a);
  const checks=best.process_checks||[];
  const fails=(best.test_rows||[]).filter(t=>!t.pass).map(t=>t.name);
  const j=best.glm_judge||{};
  const speed=(best.base_s&&best.bench_s)?(best.base_s/best.bench_s).toFixed(1)+'x baseline':'bench failed';
  let d=document.createElement('div');
  d.innerHTML=`<h2>${taskId}</h2><div class="grid">
   <div class="card"><h3>Mean scalar_total by round (feedback loop)</h3>${curve(taskId)}</div>
   <div class="card"><h3>Best rollout</h3>
     <table>
      <tr><td>scalar</td><td><b>${f3(best.scalar_total)}</b> (terminal ${f3(best.scalar)}, process ${f3(best.process_reward)})</td></tr>
      <tr><td>correctness</td><td>${best.correctness.toFixed(2)} ${fails.length?`<span class="neg">fails: ${esc(fails.join(', '))}</span>`:'<span class="pos">all pass</span>'}</td></tr>
      <tr><td>performance</td><td>${f3(best.perf)} — ${speed}</td></tr>
      <tr><td>style</td><td>${f3(best.style)}</td></tr>
      <tr><td>guard (anti-hack)</td><td>${f3(best.guard)}</td></tr>
      <tr><td>decision tier</td><td>${esc(JSON.stringify((best.d3_terminal||{}).quality?.label ?? '—'))}</td></tr>
      <tr><td>judge cascade</td><td>${best.skip_reason||'run'} · flash↔big agreement ${best.judge_agreement==null?'—':best.judge_agreement.toFixed(3)}</td></tr>
      ${j&&!j.parse_error?`<tr><td>glm-5.3 judge</td><td>elegance ${j.elegance}/5 · r_idiom ${j.r_idiom}/5 · anti-hack ${j.reward_hacking}/5<br><i>${esc(j.notes||'')}</i></td></tr>`:''}
     </table></div>
   <div class="card"><h3>Best code</h3><pre>${esc(best.code)}</pre>
     <h3 style="margin-top:10px">Process checks while typing (d3 on growing prefix)</h3>
     ${checks.length?`<table><tr><th>chars</th><th>focus</th><th>syntax</th><th>risky</th><th>r</th></tr>`+
       checks.map(c=>`<tr><td>${c.chars}</td><td>${esc(c.labels.focus??'—')}</td><td>${esc(String(c.labels.syntax_ok??'—'))}</td><td>${esc(String(c.labels.risky??'—'))}</td><td class="${c.r>=0?'pos':'neg'}">${f3(c.r)}</td></tr>`).join('')+'</table>':'no checks recorded'}</div>
  </div>`;
  tasksDiv.appendChild(d);
});

const skipped=ROWS.filter(r=>r.cascade_skipped_glm).length;
const dead=ROWS.filter(r=>r.skip_reason==='dead').length;
const clean=ROWS.filter(r=>r.skip_reason==='clean').length;
const replay=ROWS.filter(r=>r.route_replay&&r.route_replay!==(r.skip_reason||'run'));
const replaySaved=ROWS.filter(r=>r.route_replay&&r.route_replay!=='run').length;
const judgeErr=ROWS.filter(r=>(r.glm_judge||{}).parse_error&&!r.cascade_skipped_glm).length;
const agree=ROWS.filter(r=>r.judge_agreement!=null).map(r=>r.judge_agreement);
const meanAgree=agree.length?(agree.reduce((a,b)=>a+b,0)/agree.length).toFixed(3):'—';
document.getElementById('ledger').innerHTML=`
 <span class="pill">${ROWS.length} rollouts</span>
 <span class="pill">big-judge routed off: ${skipped}/${ROWS.length} (dead ${dead}, confident-clean ${clean})</span>
 <span class="pill">glm-5.3↔flash style agreement: ${meanAgree}</span>
 <span class="pill">judge parse errors: ${judgeErr}</span>
 <span class="pill">decision-tier backend: ${esc(String(META.d3_backend))}</span>
 ${replay.length?`<span class="pill">fixed-router replay: ${replaySaved}/${ROWS.length} big-judge calls would be skipped</span>`:''}
 <table style="margin-top:10px"><tr><th>task</th><th>round</th><th>rollout</th>
 <th>scalar</th><th>corr</th><th>heldout</th><th>gap</th><th>perf</th><th>style</th><th>guard</th><th>proc</th><th>d3 quality</th><th>route</th><th>agree</th><th>loc</th></tr>
 ${ROWS.map(r=>`<tr><td>${r.task_id}</td><td>${r.round}</td><td>${r.rollout}</td>
   <td class="${r.scalar_total>=0?'pos':'neg'}">${f3(r.scalar_total)}</td>
   <td>${r.correctness.toFixed(2)}</td>
   <td>${r.heldout_correctness==null?'—':r.heldout_correctness.toFixed(2)}</td>
   <td class="${(r.rh_gap||0)>0.34?'neg':''}">${r.rh_gap==null?'—':r.rh_gap.toFixed(2)}</td>
   <td>${f3(r.perf)}</td><td>${f3(r.style)}</td>
   <td>${f3(r.guard)}</td><td>${f3(r.process_reward)}</td>
   <td>${esc(String((r.d3_terminal||{}).quality?.label_fixed ?? (r.d3_terminal||{}).quality?.label ?? '—'))}</td>
   <td>${r.skip_reason||'run'}${r.route_replay&&r.route_replay!==(r.skip_reason||'run')?` →${r.route_replay}`:''}</td><td>${r.judge_agreement==null?'—':r.judge_agreement.toFixed(3)}</td>
   <td>${r.code_lines||'—'}</td></tr>`).join('')}</table>`;
</script></body></html>
"""


def write_dashboard(rows: list[dict], meta: dict, path: Path) -> None:
    html = (_DASH
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
            .replace("__META__", json.dumps(meta)))
    path.write_text(html)
