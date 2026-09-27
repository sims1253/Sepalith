#!/usr/bin/env python3
"""Build a self-contained tracker and a script-free PostPlan snapshot."""
from pathlib import Path
import base64
import html
import json
import re
from campaign import CampaignStore, validate_manifest

ROOT = Path(__file__).resolve().parent
esc = lambda value: html.escape(str(value), quote=True)


def list_html(items, ordered=False):
    tag = 'ol' if ordered else 'ul'
    return f'<{tag}>' + ''.join(f'<li>{esc(x)}</li>' for x in items) + f'</{tag}>'


def static_brief(task):
    return (f'<p>{esc(task["summary"])}</p><h4>Work through</h4>'
            + list_html(task['steps'], True) + '<h4>Done when</h4>'
            + list_html(task['acceptance']) + '<h4>Hand back</h4>'
            + list_html(task['outputs']) + '<h4>Stop or escalate</h4>'
            + f'<p>{esc(task["stopRule"])}</p>'
            + '<p class="task-deps">Prerequisites: '
            + (', '.join(f'<a href="#{esc(x)}">{esc(x)}</a>' for x in task['dependsOn']) or 'None')
            + '</p>')


def build():
    manifest = validate_manifest(json.loads((ROOT / 'tasks.json').read_text()))
    state = CampaignStore(ROOT).read_state()
    tasks = manifest['tasks']
    phase_markup, nav = [], []
    short_titles = {'prepare':'Preparation', 'prompt':'Prompt & context', 'data':'Data & evaluation', 'sft':'Primary SFT', 'rl':'Main RL', 'parallel':'Cloud branches', 'runtime':'Serving & drafts', 'release':'Release'}
    for phase in manifest['phases']:
        items = [t for t in tasks if t['phase'] == phase['id']]
        rows = []
        for task in items:
            status = state['tasks'].get(task['id'], {}).get('status', 'todo')
            mark = '✓' if status == 'done' else ''
            rows.append(f'''<details class="task" id="{esc(task['id'])}" data-task="{esc(task['id'])}">
<summary><span class="task-dot {esc(status)}" aria-hidden="true">{mark}</span><span class="task-body"><span class="task-title">{esc(task['title'])}</span><span class="task-meta">{esc(task['id'])} <span>{esc(task['role'])}</span> <span>{esc(task['effort'])}</span></span></span><span class="kind {esc(task['kind'])}">{esc(task['kind'].capitalize())}</span></summary>
<div class="inline-brief">{static_brief(task)}</div></details>''')
        phase_markup.append(f'<section class="phase-group" id="phase-{esc(phase["id"])}" data-phase="{esc(phase["id"])}"><div class="phase-heading"><h3>{esc(phase["title"])}</h3><span>{len(items)} tasks</span></div><p class="phase-goal">{esc(phase["goal"])}</p>{"".join(rows)}</section>')
        nav.append(f'<a class="phase-link" href="#phase-{esc(phase["id"])}" data-filter-phase="{esc(phase["id"])}"><span>{esc(short_titles.get(phase["id"],phase["title"]))}</span><span class="phase-count">{len(items)}</span></a>')
    docs = {}
    for name in ('LEAD-AGENT.md', 'DELEGATION.md'):
        path = ROOT / name
        docs[name] = path.read_text() if path.exists() else 'This guide is being prepared.'
    payload = {'manifest': manifest, 'state': state, 'docs': docs}
    fonts = ''
    for license_file in sorted((ROOT / 'assets').glob('*-OFL.txt')):
        fonts += '/* Bundled font license: ' + license_file.name + '\n' + license_file.read_text().replace('*/', '* /') + '\n*/\n'
    for filename, family, weight in [('barlow-600.ttf', 'Barlow Semi Condensed', '600'), ('public-400.ttf', 'Public Sans', '400'), ('public-600.ttf', 'Public Sans', '600')]:
        path = ROOT / 'assets' / filename
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode()
            fonts += f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};font-display:swap;src:url(data:font/ttf;base64,{encoded}) format('truetype');}}\n"
    template = (ROOT / 'board.template.html').read_text()
    rendered = template.replace('%%FONTS%%', fonts).replace('%%NAV%%', ''.join(nav)).replace('%%TASKS%%', ''.join(phase_markup)).replace('%%COUNT%%', str(len(tasks)))
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    rendered = rendered.replace('%%APP_DATA%%', encoded)
    (ROOT / 'board.html').write_text(rendered)
    # The hosted copy has no scripts, local paths, private notes or embedded state.
    public = re.sub(r'<script\b[^>]*>[\s\S]*?</script>', '', rendered, flags=re.I)
    public = re.sub(r'/home/[^<\s]+', '[local artifact]', public)
    public = public.replace('data-edition="local"', 'data-edition="published"')
    public = public.replace('Snapshot mode. Open the local tracker to save progress.', 'Published snapshot. Expand a task to read its brief. Update progress in the local tracker.')
    (ROOT / 'postplan.html').write_text(public)
    print(f'Built {len(tasks)} tasks: {ROOT / "board.html"}')
    print(f'PostPlan snapshot: {ROOT / "postplan.html"}')


if __name__ == '__main__':
    build()
