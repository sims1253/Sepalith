#!/usr/bin/env python3
"""spark_design_v2.py — muse-spark as DESIGNER OF RECORD for dashboard v2.

The user's correction: spark designs and writes; I (the agent) enforce,
implement, and ground. This driver runs the design conversation in bounded
turns (each call's output stays under the 4k-token budget; skill-doc texts
ride in the prompt):

  turn identity   frontend-design + show-me + design brief -> design tokens
  turn families   humanizer + orwell + family facts       -> glossary copy
  turn verdicts   humanizer + orwell + verdict facts      -> timeline + tree copy
  turn chrome     humanizer + orwell + page facts         -> intros/labels/stand/jargon
  turn critique   current design/copy + executor RENDER NOTES -> revise or sign off

Outputs (committed, consumed by build_dashboard_v2.py):
  design_v2.json  spark's token system + layout decisions
  copy_v2.json    spark's static copy (glossary, tree notes, intros, labels)

Every call is logged with token stats to results/spark_design_log.jsonl.
Spark may not invent numbers: FACTS blocks carry only verified values.

CLI:
  spark_design_v2.py identity|families|verdicts|chrome    run one authoring turn
  spark_design_v2.py critique NOTES_FILE                  run a critique turn
  spark_design_v2.py log                                  print the log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import staticdata_v2 as SD                                    # noqa: E402
SYN = HERE.parent / "synthetic-data"
RESULTS = HERE / "results"
SKILLS = RESULTS / "skillrefs"
DESIGN_OUT = HERE / "design_v2.json"
COPY_OUT = HERE / "copy_v2.json"
LOG = RESULTS / "spark_design_log.jsonl"

DESIGN_BRIEF = """\
DESIGN BRIEF (the client's ask, verbatim constraints)

Product: Sepalith — an open, R-specialized next-edit-suggestion model.
Local-first (llama.cpp/GGUF), aimed at pharma and biostat R work. The page
is a PROGRAM LOG: a status dashboard the researcher reloads to answer, in
one pass: what happened, what does the experiment family tree look like,
what is queued right now, what do the synthetic-data family names mean
(the user explicitly said they can no longer decode them), and what data
exists on disk. Audience: one expert user (the program owner) plus
collaborators who see the public URL.

Sections (fixed IA — do not reorder or rename):
  Live-now strip (owners/ETAs) at top after a one-paragraph stand.
  1  What happened — verdict timeline; each entry: asked / measured / so-what.
  2  The program tree — genealogy of experiment lines; killed branches are
     load-bearing results and must read as distinct from live lines.
  3  What is queued — running/parked/proposed one-liners from the queue file.
  4  Family glossary — every synthetic family decoded in plain words, with
     tiny R examples where they clarify; then a jargon table (one term per
     thing, defined once).
  5  Data on disk — three inventories with real counted numbers.

Hard medium constraints (executor-enforced, design around them):
  - One self-contained HTML file; NO external requests (no webfonts, no
    CDN, no images). System font stacks only.
  - Page must read fully with JavaScript disabled (JS = one small
    progressive enhancement only).
  - Responsive to mobile; keyboard focus visible; reduced-motion respected.
  - No inline event handlers (executor sanitizes).
Subject matter to ground the identity in: R source code, editing, git
diffs, CRAN packages, a research lab notebook, CPU/GPU training runs,
verdict-first science writing. The hero should be the most characteristic
thing this program produces (the verdict timeline, the tree, or the
live-now strip — your call, argue it).
"""


def _skill(name: str) -> str:
    return (SKILLS / f"{name}.md").read_text()


def _log_call(turn: str, prompt_len: int, out_len: int, stats: dict, note: str):
    RESULTS.mkdir(exist_ok=True)
    ent = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), turn=turn,
               prompt_chars=prompt_len, output_chars=out_len, stats=stats,
               note=note)
    with open(LOG, "a") as f:
        f.write(json.dumps(ent, ensure_ascii=False) + "\n")


def call_spark(turn: str, prompt: str) -> dict:
    sys.path.insert(0, str(SYN))
    from cases.backends import OpencodeSparkBackend, extract_json_object
    be = OpencodeSparkBackend()
    raw = be.complete(prompt)
    stats = be.stats_summary()
    obj = extract_json_object(raw)
    _log_call(turn, len(prompt), len(raw), stats,
              "ok" if obj is not None else "PARSE FAIL")
    if obj is None:
        raise ValueError(f"{turn}: spark output not JSON: {raw[:300]!r}")
    (RESULTS / f"spark_design_{turn}.json").write_text(
        json.dumps(obj, indent=1, ensure_ascii=False))
    return obj


# ---------------------------------------------------------------------------
# fact packs (verified numbers only; spark writes words, not numbers)
# ---------------------------------------------------------------------------

def facts_families() -> str:
    lines = []
    for gtitle, fams in SD.GLOSSARY_GROUPS:
        lines.append(f"## GROUP {gtitle}")
        for f in fams:
            ex = f" EXAMPLE_TEXT: {f['example']}" if f.get("example") else ""
            lines.append(f"- {f['name']}: {f['plain']}{ex}")
    lines.append("## JARGON TERMS (term: current draft definition)")
    for term, plain, definition in SD.JARGON:
        lines.append(f"- {term} ({plain}): {definition}")
    return "\n".join(lines)


def facts_verdicts() -> str:
    lines = ["## TIMELINE ENTRIES (id | date | verdict | measured FACTS | draft ask | draft so-what)"]
    for e in SD.TIMELINE_SEED:
        lines.append(f"- {e['id']} | {e['date']} | {e['verdict']} | "
                     f"FACTS: {e['measured']} | draft ask: {e['ask']} | "
                     f"draft so-what: {e['implies']}")
    lines.append("\n## TREE NODES (id | status | label | FACTS note)")
    def walk(nodes, depth=0):
        for n in nodes:
            lines.append(f"{'  '*depth}- {n['id']} [{n['status']}] "
                         f"{n['label']} :: {n.get('note','')}")
            walk(n.get("children") or [], depth + 1)
    walk(SD.SKILL_TREE)
    return "\n".join(lines)


def facts_chrome() -> str:
    return """\
## PAGE FACTS
- The page opens with: title "Sepalith", a one-line subtitle (draft: program
  log for an open R next-edit model, local-first, pharma/biostat), then the
  stand paragraph, then the live-now table (columns: what, owner, note+eta).
- Section 1 intro (draft): every verdict banked, newest first, numbers quoted
  from results docs.
- Section 2 intro (draft): how lines depend on each other; killed branches
  are load-bearing; gates are pre-registered.
- Section 3 intro: parsed from the queue file each cycle.
- Section 4 intro: every family name decoded with live row counts.
- Section 5 intro: three inventories counted from disk, cached between cycles.
- Footer: standing decisions list, next-up list, build provenance line.
- Timeline field labels currently: asked / measured / so-what.
- Verdict pill vocabulary (repo standard, keep): WIN, DONE, KILLED, DEAD,
  VALIDATED, NOT LAND, GO, ADOPT..., ELIMINATED, plus TIE/MEASURED variants.
- Status tags on tree nodes: done, win, killed, gate, running, parked,
  proposed, blocked.
"""


# ---------------------------------------------------------------------------
# turn prompts
# ---------------------------------------------------------------------------

def prompt_identity() -> str:
    return f"""You are the DESIGN LEAD for the Sepalith program-log dashboard rework. A separate executor implements and enforces hard constraints; you own every design and copy decision. This call: the visual identity and token system.

{_skill('frontend-design')}

{_skill('show-me')}

{DESIGN_BRIEF}

Work the frontend-design process: brainstorm, then review your plan against the brief (kill anything that reads as a default you would produce for any similar brief), then commit. Respond ONLY with a JSON object, no prose outside it:

{{
 "concept": "one paragraph: the identity idea grounded in THIS subject and why",
 "hero": "which element is the hero (the one bold thing) and how it is treated",
 "palette": [{{"name": str, "hex": "#RRGGBB", "role": str}}],   // 4-7 named colors
 "type": {{"families": [{{"stack": str, "role": str}}], "scale_notes": str}},  // 1-2 families, system stacks only (no webfonts)
 "layout": {{"prose": str, "alignment": str, "wireframe": "ASCII wireframe of the page skeleton"}},
 "structure_devices": "what borders/numbering/dividers encode (must encode real information)",
 "timeline_treatment": "how a verdict entry reads visually (date/id/verdict pill/fields)",
 "tree_treatment": "how the genealogy reads: edges, killed vs live vs gate vs running distinctions",
 "table_treatment": "queue + inventory tables",
 "motion": "what the single JS enhancement is (or none), reduced-motion note",
 "anti_defaults": "which generic tells you deliberately avoided and what you chose instead"
}}"""

_COPY_RULES = f"""\
COPY RULES (humanizer + orwell, both reproduced in full below — they govern every word you write):

{_skill('humanizer')}

{_skill('orwell-writing')}

Extra hard rules from the client:
- Keep EVERY claim; invent NOTHING. Numbers, dates, names, verdict words come
  from the FACTS block only — you may drop a number, never add or alter one.
- One consistent term per thing (no synonyms for the same concept).
- Jargon gets decoded in plain words; define once.
- The reader is the program owner: expert, in a hurry, allergic to filler.
- No emoji. No exclamation marks. Plain sentences. Verdict-first.
"""


def prompt_families() -> str:
    return f"""{_COPY_RULES}

TASK: author the FAMILY GLOSSARY copy for the Sepalith program log. Below is the fact pack: every synthetic-data family with its verified description, its tiny R example text where one exists, and the jargon table. Rewrite each family's decode line so a reader who has forgotten what these names mean understands what the family trains, in one or two plain sentences. Keep the R examples as-is where present (you may lightly adjust wording inside an example only if it clarifies; keep it valid R). Also rewrite each jargon definition. Do not rename families or terms.

FACT PACK:
{facts_families()}

Respond ONLY with a JSON object:
{{
 "families": [{{"name": str, "decode": str, "example": str-or-empty}}],
 "jargon": [{{"term": str, "plain": str, "definition": str}}],
 "group_titles": [str],
 "notes": "one line on what you changed and why"
}}
All names/terms must match the fact pack exactly. decode at most 2 sentences, 320 chars."""


def prompt_verdicts(half: str = "a") -> str:
    seed = SD.TIMELINE_SEED
    if half == "a":
        tl, tree = seed[:10], []
        scope = ("This call covers TIMELINE entries 1-10 ONLY (the pack below "
                 "is pre-filtered). Return only their timeline objects.")
    elif half == "b":
        tl, tree = seed[10:], []
        scope = ("This call covers TIMELINE entries 11-20 ONLY (the pack "
                 "below is pre-filtered). Return only their timeline objects.")
    else:
        tl, tree = [], SD.SKILL_TREE
        scope = ("This call covers ALL TREE nodes ONLY (no timeline entries).")
    lines = [f"- {e['id']} | {e['date']} | {e['verdict']} | FACTS: "
             f"{e['measured']} | draft ask: {e['ask']} | draft so-what: "
             f"{e['implies']}" for e in tl]
    tlines: list[str] = []
    def walk(nodes, depth=0):
        for n in nodes:
            tlines.append(f"{'  '*depth}- {n['id']} [{n['status']}] "
                          f"{n['label']} :: {n.get('note','')}")
            walk(n.get("children") or [], depth + 1)
    walk(tree)
    return f"""{_COPY_RULES}

TASK: author verdict TIMELINE and PROGRAM-TREE copy for the Sepalith program log. {scope} For each timeline entry rewrite "ask" (the question, one sentence) and "implies" (what the verdict CHANGES for the program, one sentence); keep every number in "measured" but you may reword its sentences to read cleaner. For each tree node rewrite its "note" (one clause-rich line, 180 chars max). The verdict WORD stays exactly as given (repo vocabulary). Do not add or merge entries.

TIMELINE FACT PACK:
{chr(10).join(lines)}

TREE FACT PACK:
{chr(10).join(tlines) if tlines else "(not in this call)"}

Respond ONLY with a JSON object:
{{
 "timeline": [{{"id": str, "ask": str, "measured": str, "implies": str}}],
 "tree": [{{"id": str, "note": str}}],
 "notes": "one line on what you changed and why"
}}
Include "tree" only when tree nodes were in this call's pack. ids must match exactly. ask at most 140 chars; implies at most 220 chars."""


def prompt_chrome() -> str:
    return f"""{_COPY_RULES}

TASK: author the page CHROME copy for the Sepalith program log (intros, labels, stand). Facts below. Rules: the section order and meaning are fixed; the stand paragraph must state where the program stands (decided / in flight / blocked on what) using only the fact pack; field labels replace my drafts if you have better ones (they repeat on every timeline entry — short, lowercase, information-carrying).

{facts_chrome()}

Timeline FACTS for the stand paragraph (newest first):
{chr(10).join(f"- {e['id']} {e['date']} {e['verdict']}: {e['measured'][:220]}" for e in reversed(SD.TIMELINE_SEED))}

Live-now FACTS: B13 LFM2.5-2.6B rung training (blocks the base-pick GO); H1 harness bake-off pipeline running; TU2 teacher solve running; V1d pairwise judging running; S0 trace freeze running. Standing: gate B-beta = three-way tie, production rec b4-config; serving model sft_v7; P12 KERNEL-DAY GO; TU1 DEAD; O3-S0 NOT LAND; W29 cloud-ready.

Respond ONLY with a JSON object:
{{
 "title": str, "subtitle": str,
 "stand": str,                       // 2-4 sentences, where the program stands
 "live_label": str,                  // live-now strip heading
 "section_intros": {{"happened": str, "tree": str, "queued": str, "glossary": str, "data": str}},
 "timeline_labels": {{"ask": str, "measured": str, "implies": str}},
 "footer_line": str,                 // build provenance, one sentence
 "notes": "one line on what you changed and why"
}}"""


def prompt_critique(notes_path: Path) -> str:
    notes = notes_path.read_text()
    design = DESIGN_OUT.read_text() if DESIGN_OUT.exists() else "{}"
    copy = COPY_OUT.read_text() if COPY_OUT.exists() else "{}"
    return f"""{_COPY_RULES}

You are the design lead reviewing YOUR OWN shipped design after the executor rendered it. The executor cannot show you pixels, so the RENDER NOTES below are your eyes: structural digest, measurements, and the executor's honest observations of what reads badly. Decide: revise (output concrete replacements) or sign off. You may revise any field of your earlier design/copy; the executor will apply exactly what you output.

CURRENT DESIGN (your earlier output, as implemented):
{design}

CURRENT COPY (your earlier output, as implemented):
{copy}

EXECUTOR RENDER NOTES:
{notes}

Respond ONLY with a JSON object:
{{
 "verdict": "signoff" | "revise",
 "design_changes": [{{"path": str, "value": str}}],   // path names a field of the design JSON, e.g. "palette[2].hex", "hero"
 "copy_changes": [{{"path": str, "value": str}}],     // path names a copy field, e.g. "families[3].decode", "section_intros.tree"
 "css_notes": [str],                                  // plain-language instructions the executor must translate to CSS
 "reasoning": "at most 3 sentences"
}}"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def merge_copy(obj: dict, part: str):
    """Merge one authoring turn's output into copy_v2.json."""
    cur = json.loads(COPY_OUT.read_text()) if COPY_OUT.exists() else {}
    if part == "families":
        cur["families"] = obj.get("families", [])
        cur["jargon"] = obj.get("jargon", [])
        cur["group_titles"] = obj.get("group_titles", [])
    elif part == "verdicts":
        have = {t.get("id") for t in cur.get("timeline", [])}
        for t in obj.get("timeline", []):
            if t.get("id") not in have:
                cur.setdefault("timeline", []).append(t)
        tmap = {t.get("id"): t for t in cur.get("tree", [])}
        for t in obj.get("tree", []):
            tmap[t.get("id")] = t
        cur["tree"] = list(tmap.values())
    elif part == "chrome":
        for k in ("title", "subtitle", "stand", "live_label",
                  "section_intros", "timeline_labels", "footer_line"):
            if k in obj:
                cur[k] = obj[k]
    cur.setdefault("_turns", []).append(part)
    COPY_OUT.write_text(json.dumps(cur, indent=1, ensure_ascii=False))


def apply_critique(obj: dict) -> str:
    """Apply a critique turn's changes to design_v2.json / copy_v2.json.
    Paths: dotted keys, any segment may carry [n] (e.g. timeline[3].implies)."""
    import re as _re
    _TOK = _re.compile(r"^([A-Za-z0-9_]+)(?:\[(\d+)\])?$")

    def get_set(root, path, value):
        node = root
        segs = path.split(".")
        for si, seg in enumerate(segs):
            m = _TOK.match(seg.strip())
            if not m:
                return f"SKIP bad seg {path}"
            key, idx = m.group(1), m.group(2)
            last = si == len(segs) - 1 and idx is None
            if isinstance(node, dict):
                if key not in node:
                    if last:
                        node[key] = value
                        return f"set {path}"
                    return f"SKIP missing key {path}"
                if last:
                    node[key] = value
                    return f"set {path}"
                node = node[key]
            elif isinstance(node, list) and key.isdigit():
                i = int(key)
                if i >= len(node):
                    return f"SKIP index {path}"
                if last:
                    node[i] = value
                    return f"set {path}"
                node = node[i]
            else:
                return f"SKIP bad node {path}"
            if idx is not None:
                i = int(idx)
                if not (isinstance(node, list) and i < len(node)):
                    return f"SKIP index {path}"
                if si == len(segs) - 1:
                    node[i] = value
                    return f"set {path}"
                node = node[i]
        return f"SKIP unresolved {path}"

    applied = []
    design = json.loads(DESIGN_OUT.read_text()) if DESIGN_OUT.exists() else {}
    copy = json.loads(COPY_OUT.read_text()) if COPY_OUT.exists() else {}
    for ch in obj.get("design_changes", []):
        applied.append(get_set(design, ch["path"], ch["value"]))
    for ch in obj.get("copy_changes", []):
        applied.append(get_set(copy, ch["path"], ch["value"]))
    DESIGN_OUT.write_text(json.dumps(design, indent=1, ensure_ascii=False))
    COPY_OUT.write_text(json.dumps(copy, indent=1, ensure_ascii=False))
    css = obj.get("css_notes") or []
    if css:
        (RESULTS / "spark_css_notes.md").write_text(
            "\n".join(f"- {c}" for c in css))
    return "\n".join(applied) or "(no changes)"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "log":
        if LOG.exists():
            print(LOG.read_text())
        return 0
    if cmd == "identity":
        obj = call_spark("identity", prompt_identity())
        DESIGN_OUT.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        print(f"design_v2.json written ({DESIGN_OUT.stat().st_size}B); "
              f"hero: {obj.get('hero', '')[:100]}")
        return 0
    if cmd in ("families", "verdicts-a", "verdicts-b", "verdicts-c", "chrome"):
        part = cmd.rsplit("-", 1)[0] if cmd.startswith("verdicts") else cmd
        prompt = (prompt_verdicts(cmd.rsplit("-", 1)[1]) if part == "verdicts"
                  else getattr(sys.modules[__name__], f"prompt_{cmd}")())
        obj = call_spark(cmd, prompt)
        merge_copy(obj, part)
        print(f"copy_v2.json merged turn={cmd}: {obj.get('notes', '')[:200]}")
        return 0
    if cmd == "critique":
        if len(sys.argv) < 3:
            print("usage: critique NOTES_FILE")
            return 2
        obj = call_spark("critique", prompt_critique(Path(sys.argv[2])))
        print("verdict:", obj.get("verdict"))
        print(apply_critique(obj))
        print("reasoning:", obj.get("reasoning", ""))
        return 0
    if cmd == "apply":
        if len(sys.argv) < 3:
            print("usage: apply results/spark_design_critique.json")
            return 2
        obj = json.loads(Path(sys.argv[2]).read_text())
        print(apply_critique(obj))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
