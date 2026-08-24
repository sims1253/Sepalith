#!/usr/bin/env python3
"""Panel judge for the coding simulator (user directive 2026-08-24:
gemma-free is too small; use ox-alpha, gemini, muse — glm rejoins at
quota reset). Majority vote; every judge's verdict recorded.

ACCEPTANCE SEMANTICS (user baseline): accept ONLY if the developer would
take the suggestion EXACTLY as proposed — no edits needed. "Usefully
starts what they'd type" is NOT acceptance (they'd have to edit it).

Judges:
  ox     openrouter stealth/ox-alpha (OPENROUTER_API_KEY)
  gemini agy CLI gemini-3.7-flash-low (subprocess)
  muse   opencode zen GO responses API muse-spark-1.2-contributor
         (OPENCODE_API_KEY)

For card-comparison rounds pass --ox-vote off when ox authored one side
(conflict of interest): gemini+muse vote, tie = no winner.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.request

ZEN_RESP = "https://opencode.ai/zen/go/v1/responses"  # GO tier (the proven path)
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
AGY_CMD = ["agy", "--print", "--new-project", "--model",
           "gemini-3.7-flash-low", "--effort", "low"]


def _j_of(resp: dict) -> str:
    # openrouter/zen chat shape
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def ask_ox(prompt: str, max_tokens: int = 300) -> str:
    req = urllib.request.Request(
        OR_URL,
        data=json.dumps({"model": "stealth/ox-alpha",
                         "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": prompt}]}
                        ).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return _j_of(json.loads(r.read()))


def ask_muse(prompt: str, max_tokens: int = 300) -> str:
    req = urllib.request.Request(
        ZEN_RESP,
        data=json.dumps({
            "model": "muse-spark-1.2-contributor",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt}]}],
            "max_output_tokens": 2000,
            "reasoning": {"effort": "low"}}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0",   # plain urllib UA gets 403
                 "Authorization": f"Bearer {os.environ['OPENCODE_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
        # the muse contract: text lands in output[] items of type "message"
        return "".join(
            c.get("text", "")
            for o in d.get("output", []) if o.get("type") == "message"
            for c in o.get("content", []) if isinstance(c, dict))


def ask_gemini(prompt: str) -> str:
    r = subprocess.run(AGY_CMD + ["--prompt", prompt],
                       capture_output=True, text=True, timeout=150)
    return (r.stdout or "").strip()


def verdict(text: str, options=("accept", "reject", "wrong")) -> str | None:
    t = (text or "").strip().lower()
    for v in options:
        if re.search(rf"\b{v}\b", t):
            return v
    return None


class Panel:
    """Three judges, majority; per-judge verdicts recorded."""

    def __init__(self, judges=("ox", "gemini", "muse")):
        self.judges = judges
        self.ask = {"ox": ask_ox, "gemini": ask_gemini, "muse": ask_muse}

    def vote(self, prompt: str, options=("accept", "reject", "wrong")):
        per = {}
        for j in self.judges:
            for attempt in range(2):
                try:
                    per[j] = verdict(self.ask[j](prompt), options)
                    if per[j]:
                        break
                except Exception:
                    time.sleep(2 + attempt * 3)
            per.setdefault(j, None)
        votes = [v for v in per.values() if v]
        if not votes:
            return "bad", per
        for v in options:                      # majority in option priority
            if votes.count(v) * 2 > len(votes):
                return v, per
        return ("split", per) if len(set(votes)) > 1 else (votes[0], per)


ACCEPT_PROMPT = """You are simulating a developer's acceptance decision for a code suggestion shown as ghost text in their editor.

THE RULE (strict): accept ONLY if the developer would take the suggestion EXACTLY as proposed, with NO EDITS needed. If they would accept part of it, modify it, or type something different — reject. If it is wrong for this position — wrong.

THEIR GOAL (this commit): {goal}

CODE AT THE CURSOR (they just paused here):
```
{ctx}
```

WHAT THEY ACTUALLY TYPED NEXT:
```
{gt}
```

THE SUGGESTION SHOWN:
```
{prop}
```

Reply with ONLY one word: accept, reject, or wrong."""
