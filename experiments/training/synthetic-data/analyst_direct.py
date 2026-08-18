#!/usr/bin/env python3
"""Minimal direct z.ai analyst-script generator (no state machine).
Uses grid.py cells + validate.py gate; appends to analyst_direct.jsonl.
"""
import json, os, random, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid
from validate import validate, ANALYST_SCHEMA

ENDPOINT = "https://api.z.ai/api/coding/paas/v4/chat/completions"
OUT = Path("/mnt/h/sepalith/datasets/synthetic_analyst_v1/analyst_direct.jsonl")
LOG = Path("/mnt/h/sepalith/datasets/synthetic_analyst_v1/analyst_direct.log")
DEADLINE = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 7200)


def call(prompt):
    body = json.dumps({
        "model": "glm-5.3", "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": 2500, "temperature": 0.8,
    }).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        "Authorization": f"Bearer {os.environ['ZAI_API_KEY']}",
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def main():
    rng = random.Random(int(time.time()))
    seen = set()
    if OUT.exists():
        for l in open(OUT):
            try: seen.add(" ".join(json.loads(l)["code"].split()))
            except Exception: pass
    n_ok = n_rej = 0
    while time.time() < DEADLINE:
        c = grid.cell(rng)
        prompt = grid.ANALYST_PROMPT.format(**c)
        try:
            raw = call(prompt)
            obj = json.loads(raw)
            ok, layer, info, jw = validate(obj, ANALYST_SCHEMA, "code")
        except Exception as e:
            n_rej += 1
            LOG.parent.mkdir(exist_ok=True)
            with open(LOG, "a") as f:
                f.write(f"ERR {type(e).__name__} {str(e)[:100]}\n")
            time.sleep(3)
            continue
        key = " ".join(obj.get("code", "").split())
        if ok and key not in seen:
            seen.add(key)
            rec = dict(obj, grid_cell=c, model="glm-5.3",
                       generator="analyst_direct.py", source="zai-coding",
                       license=None, source_url=None, full_prompt=prompt,
                       generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       jarl_warnings=jw, valid=True, family="analyst")
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_ok += 1
        else:
            n_rej += 1
        if (n_ok + n_rej) % 10 == 0:
            print(f"ok={n_ok} rej={n_rej}", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
