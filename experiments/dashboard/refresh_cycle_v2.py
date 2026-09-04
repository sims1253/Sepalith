#!/usr/bin/env python3
"""One dashboard-v2 refresh cycle: gather -> queue parse -> muse-spark
editorial patch -> build -> upload.

Same fail-soft contract as refresh_cycle.py (v1), new state schema:

  state_v2.json = {
    meta      {stand, editor_note, updated}         <- spark
    running   [{what, owner, status, eta, note}]    <- spark (live-now strip)
    timeline  [{date, id, title, ask, measured,
                verdict, verdict_class, implies}]   <- spark appends/updates
    queue     {synced, running, parked, proposed,   <- DETERMINISTIC parse of
               work_count}                             EXPERIMENT-QUEUE.md,
                                                       overwritten every cycle,
                                                       never touched by spark
    next      [str]                                  <- spark
    decisions [str]                                  <- spark appends
  }

Static sections of the page (skill tree, family glossary, jargon, data
inventory) are repo ground truth built by staticdata_v2.py at build time;
spark cannot reach them, so its patch stays small.

Hard postplan rules enforced by the validator: no tag-shaped "<" in any
state string (only <b>/<code> tags pass), no inline event handlers, no
script/javascript URIs, no unknown keys.

CLI:  refresh_cycle_v2.py [--no-spark] [--no-upload] [--dry-run]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
STATE = HERE / "state_v2.json"
RESULTS = HERE / "results"
LOOP_STATE = RESULTS / "loop_state_v2.json"
BUILD_SCRIPT = HERE / "build_dashboard_v2.py"
# overridable so an offline verify (--dry-run) never clobbers the live
# index.html while the v1 daemon still owns it
OUT_HTML = HERE / os.environ.get("DASHBOARD_V2_OUT", "index.html")
SYN = HERE.parent / "synthetic-data"

# digest source files
F_QUEUE = REPO / "docs" / "EXPERIMENT-QUEUE.md"
F_BOARD = REPO / "comms" / "board.md"
F_REGISTRY = REPO / "comms.md"
F_BAKEOFF = REPO / "experiments" / "training" / "base_bakeoff" / "RESULTS.md"
F_POCDIFF = REPO / "experiments" / "training" / "poc_diff" / "RESULTS.md"
F_TU1 = REPO / "experiments" / "synthetic-data" / "TU1_RESULTS.md"
F_O3 = REPO / "experiments" / "training" / "rl" / "O3_S0_RESULTS.md"
F_P12 = REPO / "docs" / "research" / "2026-09-04-p12-roofline-results.md"

LOOP_S_DEFAULT = 1800
RUN_STATUSES = {"run", "running", "done", "blocked", "pending"}
VERDICT_CLASSES = {"good", "bad", "neutral", "pending"}
_SAFE_TAGS = ("</b>", "<b>", "</i>", "<i>", "</em>", "<em>", "</strong>",
              "<strong>", "</code>", "<code>", "<br/>", "<br />", "<br>")
_RE_HANDLER = re.compile(r"\bon[a-z]+\s*=", re.I)
_RE_TAGISH = re.compile(r"<\s*/?\s*[A-Za-z]")
_RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_STR = {
    "stand": 900, "editor_note": 400, "what": 160, "owner": 60, "eta": 80,
    "note": 300, "date": 10, "id": 40, "title": 140, "ask": 260,
    "measured": 420, "verdict": 80, "implies": 340, "next_item": 300,
    "cell": 300, "synced": 400,
}


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def _clip(s, n):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _budgeted(lines: list[str], budget: int) -> str:
    out, used = [], 0
    for ln in lines:
        if used + len(ln) > budget:
            break
        out.append(ln)
        used += len(ln) + 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. gather (bounded digest; same sources as v1 + the new results files)
# ---------------------------------------------------------------------------

def gather_queue(budget: int = 6500) -> str:
    text = F_QUEUE.read_text(errors="replace")
    lines: list[str] = []
    m = re.search(r"Last synced:(.{0,900}?)(?:\n\n|\n---)", text, re.S)
    if m:
        lines.append("LAST-SYNC: " + _clip(m.group(1), 400))
    statusy = re.compile(
        r"RUNNING|DONE|KILLED|ELIMINATED|PREPARED|PARKED|COMPLETE|"
        r"PROPOSED|VALIDATED|user GO", re.I)
    section = ""
    for raw in text.splitlines():
        if raw.startswith("## "):
            section = _clip(raw, 80)
            lines.append(section)
            continue
        if not raw.startswith("|") or set(raw.replace("|", "").strip()) \
                <= {"-", " "}:
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        rid = _clip(cells[0], 24) if cells else "?"
        joined = _clip(" | ".join(cells[1:4]), 170)
        if statusy.search(joined) or section.startswith("## 1."):
            lines.append(f"{rid}: {joined}")
    return _budgeted(lines, budget)


def gather_board(n_posts: int = 40, budget: int = 5200) -> str:
    """Newest-first: the freshest heartbeats (owners, ETAs, live verdicts)
    must survive the budget cut, so the tail is walked backwards."""
    text = F_BOARD.read_text(errors="replace")
    posts = re.split(r"\n(?=## \[)", text)
    lines = []
    for post in reversed(posts[-n_posts:]):
        pl = [l for l in post.splitlines() if l.strip()]
        if not pl or not pl[0].startswith("## ["):
            continue
        header = _clip(pl[0].lstrip("# "), 150)
        body = " ".join(pl[1:3]) if len(pl) > 1 else ""
        lines.append(f"{header} || {_clip(body, 150)}")
    return _budgeted(lines, budget)


def _results_digest(path: Path, budget: int) -> str:
    text = path.read_text(errors="replace").splitlines()
    lines = []
    for ln in text:
        if ln.startswith("## ") or "VERDICT" in ln.upper():
            prefix = "V> " if "VERDICT" in ln.upper() else ""
            lines.append(prefix + _clip(ln.lstrip("# "), 230))
    table_rows = [l for l in text
                  if l.startswith("|") and not set(l.replace("|", "")
                                                   .strip()) <= {"-", " "}]
    for ln in table_rows[-10:]:
        lines.append(_clip(ln, 130))
    return _budgeted(lines, budget)


def gather_registry(budget: int = 900) -> str:
    lines = []
    in_reg = False
    for ln in F_REGISTRY.read_text(errors="replace").splitlines():
        if ln.startswith("## Registry"):
            in_reg = True
            continue
        if in_reg:
            if ln.startswith("## "):
                break
            if ln.startswith("|") and "active" in ln:
                lines.append(_clip(ln, 130))
    return _budgeted(lines, budget)


def gather_counters() -> str:
    d = _read_loop_state()
    keep = {k: d.get(k) for k in ("cycles", "cycles_ok", "spark_ok",
                                  "spark_fail", "upload_fail_streak",
                                  "last_ok_ts", "last_url")}
    return json.dumps(keep, ensure_ascii=False)


def build_digest() -> str:
    def soft(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:                              # noqa: BLE001
            return f"[gather failed: {type(e).__name__}]"
    parts = [
        "=== A. EXPERIMENT QUEUE (docs/EXPERIMENT-QUEUE.md, status rows) ===",
        soft(gather_queue, 6500),
        "=== B. AGENT BOARD TAIL (comms/board.md, last posts: owners, ETAs, "
        "new verdicts) ===",
        soft(gather_board, 40, 5200),
        "=== C. BASE BAKE-OFF RESULTS headlines (B-series, gates) ===",
        soft(_results_digest, F_BAKEOFF, 2200),
        "=== D. POC_DIFF RESULTS headlines (X1/M1/D-grid) ===",
        soft(_results_digest, F_POCDIFF, 1800),
        "=== E. NEW-SERIES VERDICT FILES (TU1 / O3-S0 / P12) ===",
        soft(_results_digest, F_TU1, 700),
        soft(_results_digest, F_O3, 700),
        soft(_results_digest, F_P12, 700),
        "=== F. ACTIVE REGISTRY ROWS (comms.md) ===",
        soft(gather_registry, 900),
        "=== G. THIS DAEMON'S COUNTERS ===",
        soft(gather_counters),
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 1b. deterministic queue parse -> state.queue
# ---------------------------------------------------------------------------

def _clean_cell(c: str) -> str:
    c = re.sub(r"\*\*", "", c)
    c = re.sub(r"`", "", c)
    return re.sub(r"\s+", " ", c).strip()


def _rows_from(text: str, start_after: str, stop_at: tuple) -> list[list[str]]:
    """Markdown table rows between the section marker and the next stop."""
    rows: list[list[str]] = []
    active = False
    for raw in text.splitlines():
        if raw.startswith(start_after):
            active = True
            continue
        if active and any(raw.startswith(s) for s in stop_at):
            break
        if not active or not raw.startswith("|"):
            continue
        if set(raw.replace("|", "").strip()) <= {"-", " ", ":"}:
            continue
        rows.append([_clean_cell(c) for c in raw.strip().strip("|").split("|")])
    return rows


def parse_queue_file() -> dict:
    """RUNNING / PARKED / PROPOSED one-liners straight from the queue file."""
    q: dict = {"running": [], "parked": [], "proposed": [], "work_count": 0}
    try:
        text = F_QUEUE.read_text(errors="replace")
    except OSError:
        return q
    m = re.search(r"Last synced:(.{0,600}?)(?:\n\n|\n---)", text, re.S)
    if m:
        q["synced"] = _clip(re.sub(r"\s+", " ", m.group(1)), 380)

    _RE_TABLE_HEAD = re.compile(r"^(#|Experiment|Class / cost)")
    _RE_DEAD = re.compile(
        r"\b(DONE|KILLED|ELIMINATED|COMPLETE|CANCELLED|PREPARED)\b")

    def mkrow(cells, why_i):
        rid = _clip(cells[0], 18) if cells else "?"
        item = _clip(cells[1] if len(cells) > 1 else "", 95)
        why = _clip(cells[why_i] if len(cells) > why_i else "", 165)
        return {"id": rid, "item": item, "why": why}

    def live_rows(rows, why_i):
        out = []
        for cells in rows:
            if not cells or _RE_TABLE_HEAD.match(cells[0]) \
                    or not cells[0][:1].isalnum():
                continue                       # markdown header / rule rows
            if _RE_DEAD.search(" ".join(cells)):
                continue                       # verdict already landed
            out.append(mkrow(cells, why_i))
        return out

    # section 1: RUNNING (only rows still RUNNING; DONE verdicts are the
    # timeline's job)
    for cells in _rows_from(text, "## 1.", ("## 2.", "## 3.", "## 4.", "## 5.")):
        stat = " ".join(cells[2:3])
        if re.search(r"\bRUNNING\b", stat):
            row = mkrow(cells, 2)
            row["why"] = _clip(stat, 165)
            q["running"].append(row)

    # sections 2, 2b, 2c: PARKED
    q["parked"] = live_rows(
        _rows_from(text, "## 2.", ("## 3.", "## 4.", "## 5."))[:48], 3)[:22]

    # section 3 + its subsections (E/H/O/TU/S): PROPOSED
    q["proposed"] = live_rows(
        _rows_from(text, "## 3.", ("## 4.", "## 5."))[:80], 3)[:36]

    # section 4: WORK backlog count only
    q["work_count"] = sum(
        1 for cells in _rows_from(text, "## 4.", ("## 5.",))
        if cells and re.match(r"^W\d+", cells[0]))

    q["running"], q["parked"], q["proposed"] = \
        q["running"][:10], q["parked"][:22], q["proposed"][:36]
    return q


# ---------------------------------------------------------------------------
# 2. validation (postplan hard rules + v2 schema)
# ---------------------------------------------------------------------------

def safe_html(s, field: str) -> tuple[bool, str]:
    if not isinstance(s, str):
        return False, f"{field}: not a string"
    if _RE_HANDLER.search(s):
        return False, f"{field}: inline event handler rejected"
    t = s
    for tag in _SAFE_TAGS:
        t = t.replace(tag, "")
    if _RE_TAGISH.search(t) or "javascript:" in t.lower():
        return False, f"{field}: tag-shaped '<' rejected (postplan hard rule)"
    cap = MAX_STR.get(field.split(":")[0], 1200)
    if len(s) > cap:
        return False, f"{field}: {len(s)} chars > cap {cap}"
    return True, ""


def _check_strs(obj, path: str, errs: list[str], field: str = "cell"):
    if isinstance(obj, str):
        ok, why = safe_html(obj, f"{field}:{path}")
        if not ok:
            errs.append(why)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_strs(v, f"{path}[{i}]", errs, field)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # dict keys carry their own cap (stand, measured, synced, ...)
            _check_strs(v, f"{path}.{k}", errs, k if k in MAX_STR else field)


def _check_running(rows, where, errs, cap=10):
    if not isinstance(rows, list) or len(rows) > cap:
        errs.append(f"{where}: bad/too long")
        return
    for r in rows:
        if not isinstance(r, dict) or set(r) - {"what", "owner", "status",
                                                "eta", "note"} \
                or "what" not in r or r.get("status") not in RUN_STATUSES:
            errs.append(f"{where} row bad: {_clip(r, 80)}")


def validate_state(st) -> list[str]:
    errs: list[str] = []
    if not isinstance(st, dict):
        return ["state: not an object"]
    want = {"meta", "running", "timeline", "queue", "next", "decisions"}
    if set(st) != want:
        errs.append(f"state: keys {sorted(set(st))} != {sorted(want)}")
        return errs
    meta = st["meta"]
    if not isinstance(meta, dict) or set(meta) - {"stand", "editor_note",
                                                  "updated"}:
        errs.append("meta: bad keys")
    elif not isinstance(meta.get("stand", ""), str):
        errs.append("meta.stand: not a string")
    _check_running(st.get("running"), "running", errs)
    tl = st.get("timeline")
    if not isinstance(tl, list) or len(tl) > 80:
        errs.append("timeline: bad/too long")
    else:
        for e in tl:
            if not isinstance(e, dict) or not {"id", "title", "verdict"} <= set(e):
                errs.append(f"timeline row missing id/title/verdict: "
                            f"{_clip(e, 60)}")
                continue
            if set(e) - {"date", "id", "title", "ask", "measured", "verdict",
                         "verdict_class", "implies"}:
                errs.append(f"timeline unknown keys: {_clip(e.get('id'), 40)}")
            if e.get("date") and not _RE_DATE.match(e["date"]):
                errs.append(f"timeline bad date {e['date']!r}")
            if e.get("verdict_class") and e["verdict_class"] not in \
                    VERDICT_CLASSES:
                errs.append(f"timeline bad verdict_class "
                            f"{e['verdict_class']!r}")
    q = st.get("queue")
    if not isinstance(q, dict):
        errs.append("queue: not an object")
    else:
        if set(q) - {"synced", "running", "parked", "proposed", "work_count"}:
            errs.append("queue: unknown keys")
        for k in ("running", "parked", "proposed"):
            for r in q.get(k) or []:
                if not isinstance(r, dict) or set(r) - {"id", "item", "why"}:
                    errs.append(f"queue.{k} row bad")
    for k in ("next", "decisions"):
        if not isinstance(st[k], list) or len(st[k]) > 24:
            errs.append(f"{k}: bad/too long")
    _check_strs(st, "state", errs)
    return errs


PATCH_KEYS = {"stand", "running", "append_timeline", "update_timeline",
              "next", "append_decisions", "editor_note"}


def validate_patch(p) -> list[str]:
    errs: list[str] = []
    if not isinstance(p, dict):
        return ["patch: not an object"]
    unknown = set(p) - PATCH_KEYS
    if unknown:
        return [f"patch: unknown keys {sorted(unknown)}"]
    if "stand" in p and not isinstance(p["stand"], str):
        errs.append("patch.stand type")
    if "running" in p:
        _check_running(p["running"], "patch.running", errs, cap=8)
    if "append_timeline" in p:
        at = p["append_timeline"]
        if not isinstance(at, list) or len(at) > 3:
            return ["patch.append_timeline: bad/too long"]
        for e in at:
            if not isinstance(e, dict) or not {"id", "title", "verdict"} <= set(e):
                errs.append("patch.append_timeline missing id/title/verdict")
                continue
            if set(e) - {"date", "id", "title", "ask", "measured", "verdict",
                         "verdict_class", "implies"}:
                errs.append(f"patch.append_timeline unknown keys: "
                            f"{_clip(e.get('id'), 40)}")
            if e.get("verdict_class") and e["verdict_class"] not in \
                    VERDICT_CLASSES:
                errs.append(f"patch.append_timeline bad verdict_class")
            if e.get("date") and not _RE_DATE.match(e["date"]):
                errs.append("patch.append_timeline bad date")
    if "update_timeline" in p:
        ut = p["update_timeline"]
        if not isinstance(ut, list) or len(ut) > 6:
            return ["patch.update_timeline: bad/too long"]
        for u in ut:
            if not isinstance(u, dict) or set(u) != {"find", "set"} \
                    or not isinstance(u.get("find"), str) \
                    or not isinstance(u.get("set"), dict) \
                    or set(u["set"]) - {"title", "ask", "measured", "verdict",
                                        "verdict_class", "implies"}:
                errs.append("patch.update_timeline row shape")
    if "next" in p and (not isinstance(p["next"], list) or len(p["next"]) > 8):
        return ["patch.next: bad/too long"]
    if "append_decisions" in p and (not isinstance(p["append_decisions"], list)
                                    or len(p["append_decisions"]) > 3):
        return ["patch.append_decisions: bad/too long"]
    if "editor_note" in p and not isinstance(p["editor_note"], str):
        errs.append("patch.editor_note type")
    _check_strs(p, "patch", errs)
    return errs


# ---------------------------------------------------------------------------
# 3. editorial step (muse-spark patch on the dynamic sections only)
# ---------------------------------------------------------------------------

def compact_state(st: dict, budget: int = 8000) -> str:
    lines = ["STAND: " + _clip(st["meta"].get("stand", ""), 300)]
    lines.append("RUNNING-NOW: " + " ; ".join(
        f"{_clip(r.get('what'), 60)}[{_clip(r.get('owner'), 24)}]" for r in st["running"]))
    lines.append("TIMELINE (oldest->newest; append ONLY after the last):")
    for e in st["timeline"]:
        lines.append(f"- {e.get('date', '?')} {e.get('id', '?')} "
                     f"{{verdict={e.get('verdict', '?')}}} "
                     f"{_clip(e.get('title', ''), 80)}")
    lines.append("NEXT-UP: " + " ; ".join(_clip(n, 80) for n in st["next"]))
    lines.append("DECISIONS: " + str(len(st["decisions"])) + " standing "
                 "(append-only)")
    return _budgeted(lines, budget)


def spark_prompt(digest: str, compact: str) -> str:
    return f"""You are the editor of the Sepalith program-log dashboard (a postplan.dev page). It has five sections; you maintain ONLY the dynamic ones (stand paragraph, live-now strip, verdict timeline, next-up). The queue view is parsed deterministically from the queue file and is not yours. Return ONLY a JSON object — an update patch, no prose, no markdown fence.

CURRENT STATE (compact):
{compact}

FRESH SIGNALS (repo digest NOW; owners/ETAs/new verdicts live here):
{digest}

PATCH SCHEMA (all keys optional; omit what you cannot ground in the signals):
{{
 "stand": str,
 "running": [{{"what": str, "owner": str, "status": "run"|"done"|"blocked"|"pending", "eta": str, "note": str}}],
 "append_timeline": [{{"date": "YYYY-MM-DD", "id": str, "title": str, "ask": str, "measured": str, "verdict": str, "verdict_class": "good"|"bad"|"neutral"|"pending", "implies": str}}],
 "update_timeline": [{{"find": str, "set": {{"verdict"?: str, "verdict_class"?: str, "measured"?: str, "implies"?: str}}}}],
 "next": [str],
 "append_decisions": [str],
 "editor_note": str
}}

SEMANTICS:
- "stand": ONE short paragraph, where the program stands right now (what is decided, what is in flight, what is blocked on what). REPLACES the old one.
- "running": REPLACES the live-now strip (3-6 rows; what is actually in flight per the digest). Prefer live board HEARTBEAT posts (training runs, long pipelines, API passes: they carry owner sessions, PIDs, ETAs) over static queue status rows; owner and eta from the heartbeat when present, else empty strings.
- "append_timeline": ONLY experiments whose verdict LANDED AFTER the newest existing timeline entry. Max 2. Fields in plain language: "ask" = the question in one sentence; "measured" = the real numbers from the digest; "verdict" = the verdict vocabulary word (WIN/DONE/KILLED/DEAD/VALIDATED/NOT LAND/GO...); "verdict_class" = good|bad|neutral|pending (bad = a line was killed or a suspicion confirmed); "implies" = one sentence on what changes for the program. Never duplicate an existing id.
- "update_timeline": "find" = a distinctive substring of one EXISTING timeline id or title; refresh its verdict/measured/implies if the digest carries a newer readout.
- "next": REPLACES next-up with the current asks visible in the digest (max 5, user-facing).
- "append_decisions": only durable standing rules that CHANGED (max 2).
- "editor_note": one line, what you changed and why.

HARD RULES (a violation discards your whole patch):
- Plain text only in every string. The ONLY permitted markup is <b> and <code> tags. Any other "<" character anywhere is forbidden. Never any on...= handler.
- Use ONLY facts/numbers present in FRESH SIGNALS or CURRENT STATE. Never invent metrics, dates, owners or ETAs.
- Plain everyday English in ask/implies; quote the real numbers in measured.
- If nothing material changed, return {{"editor_note": "no material change"}}."""


def spark_edit(digest: str, compact: str) -> dict:
    sys.path.insert(0, str(SYN))
    from cases.backends import (BackendError, OpencodeSparkBackend,
                                extract_json_object)
    be = OpencodeSparkBackend()
    prompt = spark_prompt(digest, compact)
    log(f"spark call: model={be.model} prompt={len(prompt)}B")
    raw = be.complete(prompt)
    stats = be.stats_summary()
    log(f"spark stats: {stats}")
    obj = extract_json_object(raw)
    if obj is None:
        raise ValueError(f"spark output not a JSON object: {raw[:200]!r}")
    return obj


def apply_patch(st: dict, patch: dict) -> tuple[dict, list[str]]:
    st = copy.deepcopy(st)
    audit: list[str] = []
    if "stand" in patch:
        st["meta"]["stand"] = patch["stand"]
        audit.append("stand := refreshed")
    if "running" in patch:
        st["running"] = patch["running"]
        audit.append(f"running := {len(patch['running'])} rows")
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())[:20]
    for e in patch.get("append_timeline", []):
        if norm(e["id"]) and any(norm(e["id"]) == norm(x.get("id", ""))
                                 for x in st["timeline"]):
            audit.append(f"SKIP append (duplicate id): {e['id'][:40]}")
            continue
        e.setdefault("date", time.strftime("%Y-%m-%d"))
        e.setdefault("verdict_class", "neutral")
        st["timeline"].append(e)
        audit.append(f"append timeline: {e['id'][:40]}")
    for u in patch.get("update_timeline", []):
        find = u["find"].lower()
        hit = next((e for e in st["timeline"]
                    if find in e.get("id", "").lower()
                    or find in e.get("title", "").lower()), None)
        if hit is None:
            audit.append(f"SKIP timeline update (no match): {u['find'][:40]}")
            continue
        hit.update(u["set"])
        audit.append(f"update timeline [{hit.get('id', '?')[:40]}]: "
                     f"{sorted(u['set'])}")
    if "next" in patch:
        st["next"] = patch["next"]
        audit.append(f"next := {len(patch['next'])} items")
    st["decisions"] = st.get("decisions", []) + patch.get(
        "append_decisions", [])
    if patch.get("append_decisions"):
        audit.append(f"decisions += {len(patch['append_decisions'])}")
    return st, audit


# ---------------------------------------------------------------------------
# 4. build + upload + counters
# ---------------------------------------------------------------------------

def _read_loop_state() -> dict:
    if LOOP_STATE.exists():
        try:
            return json.loads(LOOP_STATE.read_text())
        except ValueError:
            pass
    return {"cycles": 0, "cycles_ok": 0, "spark_ok": 0, "spark_fail": 0,
            "upload_fail_streak": 0, "upload_fail_total": 0,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _write_json_atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def run_build() -> bool:
    r = subprocess.run(["nice", "-n", "19", sys.executable,
                        str(BUILD_SCRIPT), "--out", str(OUT_HTML)],
                       capture_output=True, text=True, timeout=900)
    log(f"build rc={r.returncode}: {_clip(r.stdout.strip(), 160)}")
    if r.returncode != 0:
        log(f"build stderr: {_clip(r.stderr.strip(), 400)}")
    return r.returncode == 0


def run_upload() -> tuple[bool, str]:
    env = dict(os.environ)
    if not env.get("POSTPLAN_API_KEY"):
        return False, "POSTPLAN_API_KEY not in env"
    npx = env.get("NPX_BIN", "npx")
    try:
        r = subprocess.run([npx, "postplan", "upload", str(OUT_HTML)],
                           cwd=str(HERE), capture_output=True, text=True,
                           timeout=240, env=env)
    except subprocess.TimeoutExpired:
        return False, "upload timeout"
    out = (r.stdout + r.stderr).strip()
    url = next((ln.strip().removeprefix("URL:") for ln in out.splitlines()
                if "postplan.dev" in ln and "Raw HTML" not in ln), "").strip()
    ok = r.returncode == 0 and "http" in (url or out)
    log(f"upload rc={r.returncode} url={url or '-'} out={_clip(out, 200)}")
    return ok, url or _clip(out, 160)


# ---------------------------------------------------------------------------
# cycle
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-spark", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    ctr = _read_loop_state()
    ctr["cycles"] += 1
    ctr["last_cycle_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    ok_cycle = False
    try:
        # 1. gather
        digest = build_digest()
        (RESULTS / "digest_latest_v2.txt").write_text(digest)
        log(f"digest built: {len(digest)}B")

        # 2. editorial (fail-soft)
        if not args.no_spark:
            try:
                st = json.loads(STATE.read_text())
                patch = spark_edit(digest, compact_state(st))
                (RESULTS / "last_patch_v2.json").write_text(
                    json.dumps(patch, indent=1, ensure_ascii=False))
                errs = validate_patch(patch)
                if errs:
                    log(f"spark patch REJECTED (keep last good state): "
                        f"{errs[:4]}")
                    ctr["spark_fail"] += 1
                else:
                    new_st, audit = apply_patch(st, patch)
                    new_st["queue"] = parse_queue_file()
                    new_st["meta"]["updated"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S")
                    errs2 = validate_state(new_st)
                    if errs2:
                        log(f"patched state REJECTED (keep last good): "
                            f"{errs2[:4]}")
                        ctr["spark_fail"] += 1
                    else:
                        editor_note = patch.get("editor_note", "")
                        log(f"spark patch ok: {editor_note}")
                        for a in audit:
                            log(f"  applied: {a}")
                        if not args.dry_run:
                            (RESULTS / "state_backup_v2.json").write_text(
                                json.dumps(st, indent=1, ensure_ascii=False))
                            _write_json_atomic(STATE, new_st)
                        ctr["spark_ok"] += 1
                        ctr["last_editor_note"] = editor_note
            except Exception as e:                          # noqa: BLE001
                ctr["spark_fail"] += 1
                log(f"spark FAILED (keep last good state): "
                    f"{type(e).__name__}: {_clip(e, 300)}")
        else:
            # deterministic-only refresh: queue view still refreshes
            try:
                st = json.loads(STATE.read_text())
                st["queue"] = parse_queue_file()
                st["meta"]["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                errs = validate_state(st)
                if errs:
                    log(f"queue refresh REJECTED: {errs[:4]}")
                elif not args.dry_run:
                    _write_json_atomic(STATE, st)
                    log("queue view refreshed (no-spark mode)")
            except Exception as e:                          # noqa: BLE001
                log(f"queue refresh failed: {type(e).__name__}: {_clip(e, 200)}")

        # 3. build (always)
        if not run_build():
            raise RuntimeError("dashboard build failed")

        # 4. upload
        if args.dry_run or args.no_upload:
            log("upload skipped (--dry-run/--no-upload); cycle ok")
            ok_cycle = True
            ctr["upload_fail_streak"] = 0
        else:
            up_ok, url = run_upload()
            if up_ok:
                ctr["upload_fail_streak"] = 0
                ctr["last_url"] = url
                ok_cycle = True
            else:
                ctr["upload_fail_streak"] += 1
                ctr["upload_fail_total"] += 1
                ctr["last_error"] = url
                log(f"UPLOAD FAILED (streak {ctr['upload_fail_streak']}): "
                    f"{url}")
                return 3
        return 0
    finally:
        if ok_cycle:
            ctr["cycles_ok"] += 1
            ctr["last_ok_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            ctr["last_error"] = None
        _write_json_atomic(LOOP_STATE, ctr)


if __name__ == "__main__":
    sys.exit(main())
