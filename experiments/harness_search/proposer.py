"""proposer.py — the harness-search proposer (plan §1.4).

Backend: experiments/synthetic-data/cases/backends.py — ZaiBackend (glm-5.3)
default, OpencodeSparkBackend fallback. EVERY token is counted: the wrapper
captures the API `usage` block of each call and appends it to the arm's
proposer ledger (jsonl). Budget accounting is part of the H1 verdict.

The proposer sees (per plan §1.4): current config + archived configs +
per-example outcomes (pass/fail + parse-fail flags + latency) + the search
space schema (which encodes the source-file knob semantics). M=3
candidates/iteration, returned as STRICT JSON.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "synthetic-data" / "cases"))
sys.path.insert(0, str(HERE))

import backends as cases_backends                    # noqa: E402
from backends import extract_json_object             # noqa: E402
from harness_config import (SPACE, DEFAULT_CONFIG, DEFAULT_TEXTS,  # noqa: E402
                            validate_config, validate_texts, fingerprint)


def _load_key_from_zshrc() -> str | None:
    """The Bash tool shell does not source ~/.zshrc; judges in this repo
    document 'source ~/.zshrc first'. Parse the export line, never print."""
    import os
    import re
    if os.environ.get("ZAI_API_KEY"):
        return None
    try:
        for line in open(Path.home() / ".zshrc"):
            m = re.match(r'\s*export\s+ZAI_API_KEY="([^"]+)"', line)
            if m:
                os.environ["ZAI_API_KEY"] = m.group(1)
                return m.group(1)
    except OSError:
        pass
    return None


class _UsageZai(cases_backends.ZaiBackend):
    last_usage: dict | None = None

    def _extract(self, payload):
        self.last_usage = payload.get("usage") or {}
        return super()._extract(payload)


class _UsageSpark(cases_backends.OpencodeSparkBackend):
    last_usage: dict | None = None

    def _extract(self, payload):
        self.last_usage = payload.get("usage") or {}
        return super()._extract(payload)


def _norm_usage(u: dict | None) -> dict:
    if not u:
        return {}
    out = dict(prompt_tokens=u.get("prompt_tokens", u.get("input_tokens", 0)) or 0,
               completion_tokens=u.get("completion_tokens",
                                       u.get("output_tokens", 0)) or 0)
    out["total_tokens"] = u.get("total_tokens",
                                out["prompt_tokens"] + out["completion_tokens"]) or 0
    det = u.get("completion_tokens_details") or {}
    if isinstance(det, dict) and det.get("reasoning_tokens") is not None:
        out["reasoning_tokens"] = det["reasoning_tokens"]
    return out


INTRO = """You are the proposer inside a harness-search loop. A small frozen R-code
FIM model (MiniCPM-class, Q8_0, llama.cpp CPU serving, temperature 0) is FIXED.
You optimize only its HARNESS config: how the prompt is built (context caps,
scope sections), how the completion is decoded (max_tokens, stop markers) and
how the raw text is parsed into predicted region lines. Score = fraction of
256 D_harness edit rows whose parsed prediction passes the exact scenario
validator (both of 2 temp-0 rollouts must pass; disagreements count as
unstable/fail). Markers (<<<<<<< CURRENT, =======, >>>>>>> UPDATED, <[fim-*>],
<|user_cursor|>, <|outline|>) are FROZEN — never propose changing them."""


def _space_block() -> str:
    s = {k: (v if not isinstance(v, list) or len(v) <= 8 else v)
         for k, v in SPACE.items()}
    return json.dumps(s, indent=0, separators=(",", ":"))


def build_prompt(kind: str, ctx: dict) -> str:
    """kind: 'config' (arms a/b) or 'gepa' (arm c)."""
    parts = [INTRO]
    if kind == "config":
        parts.append(f"""KNOB SPACE (propose values EXACTLY from these lists):
{_space_block()}
Knob semantics: prefix_suffix_cap chars shared by file-prefix+suffix around the
cursor (prefix truncated from its start, tail kept; scenario rows have short
prefixes so this mostly bites the noopFP guardrail); pin = prepend-protect the
enclosing R function (MAX_PIN_CHARS); outline = a '<|outline|>' section of
top-level function signatures (entries, chars caps) between prefix and region;
suffix_truncation = protect-pin (cut oldest/deepest lines, never the pinned
function) vs hard-cut (cut strictly at the budget); max_tokens = decode cap
(320 is the shipped extension default); stops = 7-marker echo set vs minimal;
parse_rep_cut = cut degenerate repetition at the 3rd identical line;
parse_marker_drop = drop echoed marker-only lines; comment_heuristic = insert
a newline when the cursor sits in a comment and the prediction is code;
full_region_replace = replace the whole cursor line when the prediction
re-emits the typed partial (else glue after it).""")
    else:
        parts.append("""GEPA MODE: the render/decode/parse config is FROZEN at defaults.
You evolve ONLY the free-text slots the format can carry today:
  instruction_line — one line placed directly above the '<<<<<<< CURRENT' marker
  checklist_line   — one line placed directly above instruction_line
  outline_header   — INERT this run (outline section disabled); keep unchanged
Slot rules: single line, <= 220 chars, guidance text only (the model completes
R code; do not write R code in the slots), never contain a frozen marker
substring. Empty string means the slot is absent.""")
    parts.append(f"""CURRENT BEST ({ctx.get('best_kind', 'config')}):
{json.dumps(ctx['best'], indent=0, separators=(',', ':'))}
score: exact={ctx['best_exact']:.4f} unstable={ctx['best_unstable']:.4f}
p95_latency={ctx['best_p95']}s  noopFP={ctx['best_noop']:.4f}""")
    if ctx.get("archive"):
        arch = "\n".join(
            f"  {a['fp']}: exact={a['exact']:.4f} "
            f"{json.dumps(a.get('cfg') or a.get('texts'), separators=(',', ':'))}"
            for a in ctx["archive"][:8])
        parts.append(f"ARCHIVE (evaluated, best-first):\n{arch}")
    if ctx.get("outcomes"):
        parts.append("PER-EXAMPLE OUTCOMES of the current best:\n" +
                     "\n".join(ctx["outcomes"][:14]))
    if ctx.get("deme_hint"):
        parts.append(ctx["deme_hint"])
    if kind == "config":
        parts.append("""Propose EXACTLY 3 NEW candidate configs as JSON objects with EVERY knob
set. Prefer small mutations of the current best (1-2 knobs) or remixes of
archive parents; avoid fingerprints already in the archive. Reply with STRICT
JSON only: {"candidates":[{...},{...},{...}],"rationale":"one line"}""")
    else:
        parts.append("""Propose EXACTLY 3 NEW text-slot triples (all three slots in each candidate,
use "" for absent). Keep the best-performing text and vary meaningfully.
Reply with STRICT JSON only:
{"candidates":[{"instruction_line":"","checklist_line":"","outline_header":"<|outline|>"},x3],"rationale":"one line"}""")
    return "\n\n".join(parts)


class Proposer:
    def __init__(self, ledger_path: Path, prefer: str = "zai"):
        _load_key_from_zshrc()
        self.backends = []
        for name in ([prefer] + [b for b in ("zai", "opencode-spark") if b != prefer]):
            try:
                cls = _UsageZai if name == "zai" else _UsageSpark
                self.backends.append((name, cls()))
            except Exception:
                continue
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.totals = dict(calls=0, ok=0, prompt_tokens=0, completion_tokens=0,
                           total_tokens=0, reasoning_tokens=0, fallbacks=0,
                           repair_calls=0)

    def _log(self, rec: dict):
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.totals[k] += rec.get(k, 0) or 0
        if rec.get("reasoning_tokens"):
            self.totals["reasoning_tokens"] += rec["reasoning_tokens"]

    def _call(self, prompt: str, arm: str, iteration: int, kind: str,
              note: str) -> str | None:
        self.totals["calls"] += 1
        for i, (name, be) in enumerate(self.backends):
            try:
                t0 = time.time()
                be.last_usage = None
                text = be.complete(prompt)
                dt = round(time.time() - t0, 1)
                usage = _norm_usage(be.last_usage)
                self.totals["ok"] += 1
                if i > 0:
                    self.totals["fallbacks"] += 1
                self._log(dict(ts=time.strftime("%FT%T"), arm=arm, iter=iteration,
                               kind=kind, backend=name, note=note, ok=True,
                               latency_s=dt, **usage))
                return text
            except Exception as e:
                self._log(dict(ts=time.strftime("%FT%T"), arm=arm, iter=iteration,
                               kind=kind, backend=name, note=note, ok=False,
                               error=str(e)[:200]))
        return None

    def propose(self, kind: str, ctx: dict, arm: str, iteration: int):
        """-> (candidates: list[dict], raw: str|None). Candidates validated;
        invalid entries dropped by the caller (which fills fallbacks)."""
        prompt = build_prompt(kind, ctx)
        raw = self._call(prompt, arm, iteration, kind, "propose")
        cands = self._parse(raw)
        if not cands:  # one repair call, then give up (fallback mutations)
            self.totals["repair_calls"] += 1
            raw2 = self._call(prompt + "\n\nIMPORTANT: reply with the STRICT JSON "
                              "object only, no prose, no fences.", arm, iteration,
                              kind, "repair")
            cands = self._parse(raw2)
        return cands, raw

    def _parse(self, raw: str | None) -> list[dict]:
        if not raw:
            return []
        obj = extract_json_object(raw)
        if not obj or not isinstance(obj.get("candidates"), list):
            return []
        out = []
        for c in obj["candidates"][:3]:
            if not isinstance(c, dict):
                continue
            try:
                if "instruction_line" in c or "checklist_line" in c:
                    out.append(validate_texts(c))
                else:
                    out.append(validate_config(c))
            except ValueError:
                continue
        return out
