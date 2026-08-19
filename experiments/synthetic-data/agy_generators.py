#!/usr/bin/env python3
"""Synthetic generators on the free Google AI Pro quota via the agy CLI.

Two modes, both writing provenance-tagged rows to the NAS:
  c2c      — comments for comment-to-code candidate blocks (reuses the
             .c2c_cache.json candidate list; a different model family than
             glm widens the comment-style distribution; multiple comments
             per block are welcome — the comment_drafting flip turns each
             into its own training row)
  analyst  — analyst-style R scripts via grid.py cells + the shared
             3-layer validation gate (same pattern as analyst_direct.py)

Usage:
  agy_generators.py c2c 600 /mnt/h/sepalith/datasets/scenarios_v1/comment_to_code_gemini.jsonl
  agy_generators.py analyst 7200 /mnt/h/sepalith/datasets/synthetic_analyst_v1/analyst_gemini.jsonl
  (second arg = seconds to run; resumable via done-key sidecars)
"""
import json, os, random, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid
from validate import validate, ANALYST_SCHEMA

# --new-project per call: the CLI is stateful and shares session context by
# default (canned "currently running on..." responses when the shared
# language-server wedges); isolation keeps programmatic calls stateless
AGY = ["agy", "--print", "--new-project", "--model", "gemini-3.7-flash-low", "--effort", "low"]
CACHE = Path("/home/m0hawk/Documents/Sepalith/experiments/synthetic-data/.c2c_cache.json")
DEADLINE = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 3600)
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/agy_out.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

STYLES = [
    'Write ONE concise R comment (max 80 chars, no code, describes what this block does). Code:\n\n{code}\n\nReply with ONLY this JSON: {{"comment": string}}',
    'Write ONE short R comment (max 80 chars) explaining WHY this block does what it does. Code:\n\n{code}\n\nReply with ONLY this JSON: {{"comment": string}}',
    'Write ONE terse R comment (max 60 chars, telegraphic). Code:\n\n{code}\n\nReply with ONLY this JSON: {{"comment": string}}',
    'Write ONE R comment (max 80 chars) naming the inputs, transform, and result. Code:\n\n{code}\n\nReply with ONLY this JSON: {{"comment": string}}',
    'Write ONE R comment (max 80 chars) flagging the edge case or gotcha this block handles (or describe the block if none). Code:\n\n{code}\n\nReply with ONLY this JSON: {{"comment": string}}',
]


def call_agy(prompt: str, tries: int = 2) -> str | None:
    for a in range(tries):
        try:
            r = subprocess.run(AGY + ["--prompt", prompt], capture_output=True,
                               text=True, timeout=150)
            txt = r.stdout.strip()
            if txt:
                return txt
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return None


def parse_json(txt: str) -> dict | None:
    s = txt.strip().removeprefix("```json").removeprefix("```")
    s = s.removesuffix("```").strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1:
        return None
    try:
        return json.loads(s[i: j + 1])
    except ValueError:
        return None


def plain(code) -> str:
    if isinstance(code, (list, tuple)):
        return "\n".join(str(l) for l in code)
    return str(code)


def mode_c2c():
    blob = json.loads(CACHE.read_text())
    cands = blob["cands"]
    done_path = OUT.with_suffix(".done.jsonl")
    done = set()
    if done_path.exists():
        done = {json.loads(l)["key"] for l in open(done_path) if l.strip()}
    rng = random.Random(7)
    order = list(range(len(cands)))
    rng.shuffle(order)
    n_ok = n_rej = 0
    with open(OUT, "a") as fo, open(done_path, "a") as fd:
        while time.time() < DEADLINE:
            idx = next((i for i in order if f"c{i}" not in done), None)
            if idx is None:
                print("all candidates done"); break
            c = cands[idx]
            key = f"c{idx}"
            prompt = rng.choice(STYLES).format(code=plain(c.get("block") or c.get("prefix") or ""))
            raw = call_agy(prompt)
            obj = parse_json(raw) if raw else None
            cmt = obj.get("comment") if isinstance(obj, dict) else None
            if not (cmt and str(cmt).strip() and len(str(cmt)) <= 160):
                n_rej += 1
                fd.write(json.dumps({"key": key, "ok": False}) + "\n"); fd.flush()
                done.add(key)
                continue
            rec = dict(
                prefix=c.get("prefix"), region_old=[""], cursor_idx=0,
                region_new=c.get("block") or [], suffix=[],
                event_diff="", family="comment_to_code_gemini",
                package=c.get("package", "?"), path=c.get("path", "?"),
                note=f"gemini comment: {str(cmt).strip()}",
                generator="agy/gemini-3.7-flash", model="gemini-3.7-flash-low",
            )
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
            fd.write(json.dumps({"key": key, "ok": True}) + "\n"); fd.flush()
            done.add(key)
            n_ok += 1
            if (n_ok + n_rej) % 10 == 0:
                print(f"ok={n_ok} rej={n_rej}", flush=True)
    print(json.dumps({"ok": n_ok, "rej": n_rej}))


def mode_analyst():
    rng = random.Random(int(time.time()))
    done_path = OUT.with_suffix(".done.jsonl")
    seen = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                seen.add(" ".join(json.loads(l)["code"].split()))
            except Exception:
                pass
    n_ok = n_rej = 0
    with open(OUT, "a") as fo:
        while time.time() < DEADLINE:
            c = grid.cell(rng)
            prompt = grid.ANALYST_PROMPT.format(**c)
            raw = call_agy(prompt, tries=3)
            obj = parse_json(raw) if raw else None
            try:
                ok, layer, info, jw = validate(obj, ANALYST_SCHEMA, "code")
            except Exception:
                ok, jw = False, []
            key = " ".join((obj or {}).get("code", "").split())
            if ok and key and key not in seen:
                seen.add(key)
                rec = dict(obj, grid_cell=c, model="gemini-3.7-flash-low",
                           generator="agy_generators.py", source="google-ai-pro",
                           license=None, source_url=None, full_prompt=prompt,
                           generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           jarl_warnings=jw, valid=True, family="analyst")
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
                n_ok += 1
            else:
                n_rej += 1
            if (n_ok + n_rej) % 10 == 0:
                print(f"ok={n_ok} rej={n_rej}", flush=True)
    print(json.dumps({"ok": n_ok, "rej": n_rej}))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "c2c"
    (mode_c2c if mode == "c2c" else mode_analyst)()
