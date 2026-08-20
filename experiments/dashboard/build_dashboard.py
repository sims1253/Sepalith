#!/usr/bin/env python3
"""Build the Sepalith status dashboard (self-contained HTML) from state JSON.

Update workflow: edit dashboard_state.json as results land, then
  uv run python experiments/dashboard/build_dashboard.py
  npx postplan upload experiments/dashboard/index.html
"""
import html
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "dashboard_state.json"
OUT = HERE / "index.html"


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


def main():
    st = json.loads(STATE.read_text())
    now = time.strftime("%Y-%m-%d %H:%M")

    parts = []
    parts.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sepalith — experiment tracker</title>
<style>
 body{{background:#0d1117;color:#c9d1d9;font:15px/1.55 -apple-system,'Segoe UI',sans-serif;margin:0;padding:32px 16px}}
 .wrap{{max-width:1080px;margin:0 auto}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px;border-bottom:1px solid #21262d;padding-bottom:6px}}
 .sub{{color:#8b949e;margin-bottom:24px}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
 th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid #21262d}}
 th{{color:#8b949e;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
 tr:hover{{background:#161b22}}
 .card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:14px 18px;margin:12px 0}}
 .note{{color:#8b949e;font-size:13px}}
 .win{{color:#2ea043;font-weight:600}}
 code{{background:#21262d;padding:1px 5px;border-radius:4px;font-size:13px}}
</style></head><body><div class="wrap">
<h1>Sepalith <span class="note">— experiment tracker</span></h1>
<div class="sub">Open, R-specialized next-edit-suggestion model. Local-first (llama.cpp/GGUF), aimed at pharma/biostat.
Generated {esc(now)}. This page is a static snapshot — it changes when the orchestrator re-uploads it.</div>""")

    # currently running
    parts.append("<h2>Running now</h2>")
    parts.append(table(["what", "status", "note"],
                       [[r["what"], chip(r["status"]), r.get("note", "")] for r in st["running"]]))

    # experiments
    parts.append("<h2>Experiments &amp; results</h2>")
    for exp in st["experiments"]:
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

    parts.append(f'<div class="note" style="margin-top:40px">Artifacts: research log <code>docs/research/2026-08-19-night-results.md</code> · '
                 f'architecture dossier <code>docs/research/arch-dossier-v1.md</code> (red-team pending) · '
                 f'eval results <code>experiments/eval/results_*</code> · this page: <code>experiments/dashboard/</code></div>')
    parts.append("</div></body></html>")

    OUT.write_text("".join(parts))
    print(f"built {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
