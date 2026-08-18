#!/usr/bin/env python3
"""Generate validated synthetic analyst-style R scripts via FREE LLM endpoints.

Sources (round-robin with per-source pacing + failover):
  primary    OpenCode Zen  deepseek-v4-flash-free     (json_object)
  secondary  OpenRouter    z-ai/glm-5.2:free          (strict json_schema)
  tertiary   OpenRouter    dots-studio/dots-3-note-preview:free (json_object)

Families
  analyst — grid-driven snippets from grid.py's coverage grid (target N=4000)
  na_rm   — dedicated na_rm_propagation-style contexts: a dplyr summarise() with
            >=2 mean()/sd() calls that lack na.rm=TRUE (target N=800)

Gate (validate.py): jsonschema -> R parse -> jarl (hard-fail >=5 warnings).
na_rm family swaps jarl for a regex gate: a summarise() call plus >=2 un-na.rm'd
mean/sd calls. Dedup guard: whitespace-collapsed code must be unique among
accepted records.

Sampling: core grid cells (domain x packages x construct, 640) are drawn without
replacement; each carries a shuffled deck of (style, line_target) pairs, also
drawn without replacement; when a cell's deck is exhausted, style/length are
resampled at random (counted as 'recycled').

Output: /mnt/h/sepalith/datasets/synthetic_analyst_v1/
  analyst_scripts.jsonl, na_rm_contexts.jsonl, stats.json
  (+ rejects.jsonl failure log and state.json for resume)

Usage:
  generate_analyst.py --pilot 20          # pilot: N analyst + N//4 na_rm attempts
  generate_analyst.py                     # full run (resumable)
  generate_analyst.py --oneshot           # append one-shot example to analyst prompt
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grid  # noqa: E402
from validate import ANALYST_SCHEMA, validate  # noqa: E402

OUT_DIR = Path("/mnt/h/sepalith/datasets/synthetic_analyst_v1")

# Cloudflare returns "error code: 1010" (signature ban) for the default
# Python-urllib User-Agent on opencode.ai; a browser UA is REQUIRED there.
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# packages_used is added to the strict OpenRouter schema (coordinator's minimal
# version omitted it) because the validation gate requires it in every record.
GLM_STRICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "snip",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "code": {"type": "string"},
                "packages_used": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["intent", "code", "packages_used"],
            "additionalProperties": False,
        },
    },
}

SOURCES = [
    {
        "name": "zai-glm53",
        "provider": "zai-coding",
        "model": "glm-5.3",
        "endpoint": "https://api.z.ai/api/coding/paas/v4/chat/completions",
        "key_env": "ZAI_API_KEY",
        "response_format": {"type": "json_object"},
        "max_tokens": 2500,
        "temperature": 0.8,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "min_interval": 1.0,
        "cooldown_429": 300.0,
        "inflight_cap": 2,
    },
    {
        "name": "zen-deepseek",
        "provider": "opencode-zen",
        "model": "deepseek-v4-flash-free",
        "endpoint": "https://opencode.ai/zen/v1/chat/completions",
        "key_env": "OPENCODE_API_KEY",
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,   # reasoner: reasoning_content eats budget before content
        "temperature": 0.8,
        "ua": BROWSER_UA,
        "min_interval": 6.0,
        "cooldown_429": 300.0,  # its free quota closes for long stretches
        "inflight_cap": 2,
    },
    {
        "name": "or-glm52",
        "provider": "openrouter",
        "model": "z-ai/glm-5.2:free",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "response_format": GLM_STRICT_FORMAT,
        "max_tokens": 1500,
        "temperature": 0.8,
        "ua": None,
        "min_interval": 3.0,
        "cooldown_429": 60.0,
        "inflight_cap": 2,
        # ~50% of calls return "Provider returned error" / "temporarily
        # rate-limited upstream" (HTTP 429/503); internal retries with backoff
        # clear it, so allow a few extra attempts.
        "max_retries": 6,
    },
    {
        "name": "or-dots3",
        "provider": "openrouter",
        "model": "dots-studio/dots-3-note-preview:free",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        # response_format json_object empirically yields 200 + EMPTY content on
        # this model; omitting it works (prompt enforces JSON, strip handles
        # the leading whitespace it emits).
        "response_format": None,
        # reasoner: on multi-line code tasks 3000 tokens get eaten by reasoning
        # and content comes back EMPTY; 8000 leaves room for both.
        "max_tokens": 8000,
        "temperature": 0.8,
        "reasoning": {"effort": "low"},
        "ua": None,
        "min_interval": 3.0,
        "cooldown_429": 60.0,
        # no hard rate-limit seen at modest concurrency, but the free endpoint
        # serializes generations server-side and pushing past ~5 in-flight
        # starts drawing 429s without adding throughput
        "inflight_cap": 5,
    },
]

RETRYABLE_HTTP = {403, 408, 425, 429, 500, 502, 503, 504}
LINE_TARGETS = [8, 12, 16, 20, 25]

# Rscript/jarl subprocesses are CPU-light but capped per task instructions.
_VALIDATE_SEM = threading.BoundedSemaphore(4)


def gated_validate(obj, schema, code_key, run_jarl):
    with _VALIDATE_SEM:
        return validate(obj, schema, code_key, run_jarl=run_jarl)

NA_RM_PROMPT = """You write realistic R code for pharmaceutical/biostatistical analysis teams.

Task: generate ONE authentic analyst-style R snippet built around a very common
real-world slip: a grouped summary where mean()/sd() are called WITHOUT na.rm = TRUE
(the analyst forgot; NA values will propagate through the summary).

Domain: {domain}
Primary packages: dplyr (tidyr if needed)
What the code should do: grouped summarization with multiple stats
Style: use tidyverse pipes consistently
Target length: ~{line_target} lines of code.

Hard requirements:
- The snippet must contain a dplyr summarise() (or summarize()) call.
- Inside that summarise(), write 2 to 4 separate mean(...) or sd(...) calls.
- Do NOT pass na.rm = TRUE (or na.rm = T) to those mean()/sd() calls. No na.rm
  argument at all on them - that omission is the entire point of this snippet.
- Everything else must be valid R: plausible column names in the domain's
  conventions (e.g. USUBJID, TRTP, PARAMCD, AVISIT, AVAL, BASE, CHG), a filter()
  or mutate() step around the summary, at most one short comment.
- ONLY use functions that actually exist in base R or dplyr/tidyr. No invented names.

Respond ONLY with a JSON object:
{{"intent": "<one sentence describing what the snippet does>",
  "code": "<the R code, as a single string with \\n line breaks>",
  "packages_used": ["<pkg>", ...]}}"""

ONE_SHOT = """

Example response (shape/style reference ONLY - generate completely different content):
{{"intent": "Summarise mean change from baseline by treatment and visit for lab parameters",
  "code": "library(dplyr)\\n\\n# lab summary by treatment arm\\nadlb <- read.csv('adlb.csv')\\nsummary_tbl <- adlb |>\\n  filter(!is.na(AVAL)) |>\\n  group_by(TRTP, AVISIT) |>\\n  summarise(\\n    mean_aval = mean(AVAL),\\n    mean_chg = mean(CHG),\\n    sd_chg = sd(CHG),\\n    n_pts = n()\\n  )\\nprint(summary_tbl)",
  "packages_used": ["dplyr"]}}"""


class TransportError(Exception):
    pass


# ---------------------------------------------------------------- source pool
_SRC_LOCK = threading.Lock()
for _s in SOURCES:
    _s["_pace_lock"] = threading.Lock()
    _s["_next_allowed"] = 0.0
    _s["_cool_until"] = 0.0
    _s["_sem"] = threading.BoundedSemaphore(_s.get("inflight_cap", 2))
    _s["stats"] = {"requests": 0, "retries": 0, "http_errors": {},
                   "empty_content": 0, "accepted": 0, "attempts": 0}
ACTIVE_SOURCES = [s for s in SOURCES if os.environ.get(s["key_env"])]
_RR_COUNTER = [0]


def _attempt_source(src, prompt, blocking=False):
    """Run api_call on src under its in-flight semaphore so global worker count
    can exceed per-endpoint politeness caps (<=2 concurrent per endpoint)."""
    if blocking:
        if not src["_sem"].acquire(timeout=300):
            raise TransportError(f"{src['name']}: in-flight cap wait timed out")
    elif not src["_sem"].acquire(blocking=False):
        raise _Busy()
    try:
        return api_call(src, prompt)
    finally:
        src["_sem"].release()


class _Busy(TransportError):
    """Source at its in-flight cap; caller should try the next source."""


def _cooling(src, now=None):
    return time.time() if now is None else now < src["_cool_until"]


def _cooldown(src, seconds):
    with _SRC_LOCK:
        src["_cool_until"] = max(src["_cool_until"], time.time() + seconds)


def _acquire_order(k):
    """Round-robin rotation starting at k; waits (bounded) if every source is
    cooling after a 429, then prefers non-cooling sources."""
    with _SRC_LOCK:
        n = len(ACTIVE_SOURCES)
        rot = [ACTIVE_SOURCES[(k + i) % n] for i in range(n)]
        if all(_cooling(s) for s in rot):
            wait = min(s["_cool_until"] for s in rot) - time.time()
        else:
            wait = 0.0
    if wait > 0:
        time.sleep(min(wait, 120.0))
    avail = [s for s in rot if not _cooling(s)]
    cool = [s for s in rot if _cooling(s)]
    return avail + cool  # try live sources first; cooling ones as last resort


def api_call(src, prompt):
    """One generation attempt on a source. Paces requests, retries retryable
    HTTP/network errors with exponential backoff (+Retry-After), retries empty
    content briefly. 429 => cooldown + immediate raise so caller fails over."""
    key = os.environ.get(src["key_env"])
    if not key:
        raise TransportError(f"{src['key_env']} not set")
    body = {"model": src["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": src["max_tokens"]}
    if src.get("response_format"):
        body["response_format"] = src["response_format"]
    if src.get("temperature") is not None:
        body["temperature"] = src["temperature"]
    if src.get("reasoning"):
        body["reasoning"] = src["reasoning"]
    if src.get("thinking"):
        body["thinking"] = src["thinking"]
    if src.get("reasoning_effort"):
        body["reasoning_effort"] = src["reasoning_effort"]
    payload = json.dumps(body).encode()
    max_retries = src.get("max_retries", 5)
    empty_seen = 0
    for attempt in range(max_retries):
        # per-source pacing
        with src["_pace_lock"]:
            wait = src["_next_allowed"] - time.time()
            if wait > 0:
                time.sleep(wait)
            src["_next_allowed"] = time.time() + src["min_interval"]
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json",
                   "Accept": "application/json"}
        if src.get("ua"):
            headers["User-Agent"] = src["ua"]
        req = urllib.request.Request(src["endpoint"], data=payload, headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            content = (data["choices"][0]["message"].get("content") or "").strip()
            if not content:
                empty_seen += 1
                with _SRC_LOCK:
                    src["stats"]["empty_content"] += 1
                if empty_seen <= 2:
                    time.sleep(2.0)
                    continue
                raise TransportError(f"empty content x{empty_seen} from {src['model']}")
            with _SRC_LOCK:
                src["stats"]["requests"] += 1
            return content, time.time() - t0, attempt, src["name"]
        except urllib.error.HTTPError as e:
            snippet = ""
            try:
                snippet = e.read()[:150].decode("utf-8", "replace")
            except Exception:
                pass
            with _SRC_LOCK:
                src["stats"]["http_errors"][str(e.code)] = \
                    src["stats"]["http_errors"].get(str(e.code), 0) + 1
            if e.code == 429:
                if "free-models-per-day" in snippet:
                    # account-level DAILY quota exhausted: retrying is pure
                    # waste; cool down until X-RateLimit-Reset (cap 6h)
                    reset_ms = e.headers.get("X-RateLimit-Reset")
                    try:
                        until = min(float(reset_ms) / 1000.0,
                                    time.time() + 6 * 3600)
                    except (TypeError, ValueError):
                        until = time.time() + 6 * 3600
                    _cooldown(src, max(60.0, until - time.time()))
                    raise TransportError(
                        f"429 daily free-model quota on {src['name']}")
                if src["provider"] == "openrouter" and attempt < max_retries - 1:
                    # OpenRouter free models 429 with "temporarily rate-limited
                    # upstream ... retry shortly"; in-call backoff clears it.
                    if attempt:
                        with _SRC_LOCK:
                            src["stats"]["retries"] += 1
                    ra = e.headers.get("Retry-After")
                    delay = float(ra) if ra and ra.strip().isdigit() \
                        else min(30.0, 4.0 * (attempt + 1))
                    time.sleep(delay)
                    continue
                # Zen free quota: fail over instead of hammering; cooldown
                _cooldown(src, src["cooldown_429"])
                raise TransportError(f"429 on {src['name']}: {snippet}")
            if e.code in RETRYABLE_HTTP and attempt < max_retries - 1:
                if attempt:
                    with _SRC_LOCK:
                        src["stats"]["retries"] += 1
                ra = e.headers.get("Retry-After")
                delay = float(ra) if ra and ra.strip().isdigit() \
                    else min(60.0, 2.0 ** attempt)
                time.sleep(delay)
                continue
            _cooldown(src, 30.0)
            raise TransportError(f"HTTP {e.code} on {src['name']}: {snippet}")
        except TransportError:
            raise
        except Exception as e:  # network blips, timeouts
            if attempt < max_retries - 1:
                if attempt:
                    with _SRC_LOCK:
                        src["stats"]["retries"] += 1
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            _cooldown(src, 30.0)
            raise TransportError(f"{type(e).__name__} on {src['name']}: "
                                 f"{str(e)[:150]}")
    raise TransportError(f"retries exhausted on {src['name']}")


def parse_json_obj(raw):
    s = raw.strip()  # dots-3 may emit leading whitespace before the JSON
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("no JSON object in response")


# ---------------------------------------------------------------- samplers
class CellSampler:
    """Draws (domain, packages, construct) core cells without replacement; each
    cell carries a shuffled deck of (style, line_target) pairs, also without
    replacement; exhausted decks fall back to random style/length resampling."""

    def __init__(self, seed=1234):
        self.rng = random.Random(seed)
        self.cores = [(d, p, c) for d in grid.DOMAINS
                      for p in grid.PACKAGES for c in grid.CONSTRUCTS]
        self.rng.shuffle(self.cores)
        self.decks = {}
        for core in self.cores:
            pairs = [(s, l) for s in grid.REAL_ROXYGEN_HINTS for l in LINE_TARGETS]
            self.rng.shuffle(pairs)
            self.decks[core] = pairs
        self.pos = 0
        self.recycled = 0
        self.drawn = 0

    def next(self):
        core = self.cores[self.pos % len(self.cores)]
        self.pos += 1
        deck = self.decks[core]
        if deck:
            style, line_target = deck.pop()
        else:
            style = self.rng.choice(grid.REAL_ROXYGEN_HINTS)
            line_target = self.rng.choice(LINE_TARGETS)
            self.recycled += 1
        d, p, c = core
        self.drawn += 1
        return {"domain": d, "packages": p, "construct": c,
                "style": style, "line_target": line_target}

    def state(self):
        return {"pos": self.pos, "recycled": self.recycled, "drawn": self.drawn,
                "decks": {json.dumps(k): v for k, v in self.decks.items()}}

    def restore(self, st):
        self.pos = st["pos"]
        self.recycled = st["recycled"]
        self.drawn = st["drawn"]
        self.decks = {tuple(json.loads(k)): v for k, v in st["decks"].items()}


class NaRmSampler:
    def __init__(self, seed=4321):
        self.rng = random.Random(seed)
        self.domains = list(grid.DOMAINS)
        self.rng.shuffle(self.domains)
        self.pos = 0

    def next(self):
        dom = self.domains[self.pos % len(self.domains)]
        self.pos += 1
        return {"domain": dom,
                "packages": "tidyverse (dplyr/tidyr/purrr)",
                "construct": "grouped summarization with multiple stats (na.rm omitted)",
                "style": "use tidyverse pipes consistently",
                "line_target": self.rng.choice([8, 12, 16])}


# ---------------------------------------------------------------- na_rm check
def count_unna_rm_mean_sd(code):
    """Count mean()/sd() calls whose argument list has no na.rm=TRUE.
    Comments are stripped first; call extents found via paren matching."""
    code = re.sub(r"#.*", "", code)
    n = 0
    for m in re.finditer(r"\b(?:mean|sd)\s*\(", code):
        depth, i = 1, m.end()
        while i < len(code) and depth > 0:
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
            i += 1
        if depth == 0:
            inner = code[m.end():i - 1]
            if not re.search(r"na\.rm\s*=\s*(TRUE|T)\b", inner):
                n += 1
    return n


# ---------------------------------------------------------------- generator
class Generator:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.seen = set()          # normalized code, across families
        self.counts = {f: {"attempts": 0, "accepted": 0, "lat_sum": 0.0,
                           "jarl_warn_sum": 0, "lines_sum": 0,
                           "rejects": {}} for f in ("analyst", "na_rm")}
        self.consecutive_transport_fail = 0
        self.zero_accept_streak = 0
        self.outages = 0
        self.stop_reason = None
        self.elapsed_prior = 0.0
        self.t0 = time.time()
        self.sampler = CellSampler(seed=1234)
        self.narm_sampler = NaRmSampler(seed=4321)
        self.state_path = OUT_DIR / "state.json"
        self.rejects_path = OUT_DIR / "rejects.jsonl"
        self.files = {
            "analyst": OUT_DIR / "analyst_scripts.jsonl",
            "na_rm": OUT_DIR / "na_rm_contexts.jsonl",
        }

    # ---- persistence
    def resume(self):
        if self.args.fresh or not self.state_path.exists():
            return
        try:
            st = json.loads(self.state_path.read_text())
            self.sampler.restore(st["sampler"])
            self.narm_sampler.pos = st.get("narm_pos", 0)
            self.elapsed_prior = st.get("elapsed_s", 0.0)
            for fam, c in st.get("counts", {}).items():
                self.counts[fam].update(c)
            for fam, f in self.files.items():
                if f.exists():
                    n_recs = 0
                    for line in f.read_text(errors="ignore").splitlines():
                        try:
                            rec = json.loads(line)
                            self.seen.add(re.sub(r"\s+", " ", rec["code"]).strip())
                            n_recs += 1
                        except Exception:
                            continue
                    # jsonl is ground truth (kills between state writes can
                    # leave counts behind)
                    self.counts[fam]["accepted"] = n_recs
            print(f"[resume] analyst accepted={self.counts['analyst']['accepted']} "
                  f"na_rm accepted={self.counts['na_rm']['accepted']} "
                  f"seen={len(self.seen)} sampler_drawn={self.sampler.drawn}",
                  flush=True)
        except Exception as e:
            print(f"[resume] failed ({e}); continuing without state", flush=True)

    def write_state(self):
        st = {
            "sampler": self.sampler.state(),
            "narm_pos": self.narm_sampler.pos,
            "elapsed_s": self.elapsed_prior + (time.time() - self.t0),
            "counts": {f: {"attempts": c["attempts"], "accepted": c["accepted"]}
                       for f, c in self.counts.items()},
            "seen_count": len(self.seen),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(self.state_path)

    # ---- jobs
    def next_job(self):
        a, n = self.counts["analyst"], self.counts["na_rm"]
        # proportional interleave (5:1) once na_rm is behind its share; the
        # na_rm family stays continuously exercised instead of starting only
        # after the analyst target is met
        if n["accepted"] < self.args.narm_n and n["attempts"] < \
                self.args.attempt_cap and (a["accepted"] >= self.args.analyst_n
                                           or n["accepted"] * 5 < a["accepted"]):
            fam = "na_rm"
        elif a["accepted"] < self.args.analyst_n and \
                a["attempts"] < self.args.attempt_cap:
            fam = "analyst"
        else:
            return None
        if fam == "analyst":
            cell = self.sampler.next()
            prompt = grid.ANALYST_PROMPT.format(**cell)
            if self.args.oneshot:
                prompt += ONE_SHOT
        else:
            cell = self.narm_sampler.next()
            prompt = NA_RM_PROMPT.format(**cell)
        return fam, cell, prompt

    def run_job(self, job):
        fam, cell, prompt = job
        rec = {"family": fam, "ok": False, "layer": "", "info": "",
               "latency": 0.0, "source": None}
        with _SRC_LOCK:
            k = _RR_COUNTER[0]
            _RR_COUNTER[0] += 1
        last_err = None
        order = _acquire_order(k)
        got = None
        for src in order:
            try:
                got = _attempt_source(src, prompt, blocking=False)
                break
            except _Busy:
                continue  # at in-flight cap; try next source
            except TransportError as e:
                last_err = str(e)
                continue  # fail over to next source
        if got is None:
            # every source was at its cap (or failed): block-wait on the
            # preferred non-cooling source instead of rejecting the job
            wait_src = next((s for s in order if not _cooling(s)), order[0])
            try:
                got = _attempt_source(wait_src, prompt, blocking=True)
            except TransportError as e:
                last_err = str(e)
        if got is None:
            rec.update(layer="transport",
                       info=(last_err or "all sources failed")[:180])
            return rec
        raw, lat, _attempt, srcname = got
        src_obj = next(s for s in ACTIVE_SOURCES if s["name"] == srcname)
        rec.update(latency=lat, source=srcname)
        with _SRC_LOCK:
            src_obj["stats"]["attempts"] += 1
        try:
            obj = parse_json_obj(raw)
        except Exception:
            rec.update(layer="json", info=f"unparseable: {raw[:100]!r}")
            return rec
        try:
            ok, layer, info, jw = gated_validate(
                obj, ANALYST_SCHEMA, "code", run_jarl=(fam == "analyst"))
        except Exception as e:
            rec.update(layer="internal", info=f"validate crash: {str(e)[:120]}")
            return rec
        if not ok:
            rec.update(layer=layer, info=str(info)[:180])
            return rec
        code = obj["code"]
        if fam == "na_rm":
            if not re.search(r"\bsummar(?:is|z)e\s*\(", code):
                rec.update(layer="na_rm_check", info="no summarise() call")
                return rec
            n_bad = count_unna_rm_mean_sd(code)
            if n_bad < 2:
                rec.update(layer="na_rm_check",
                           info=f"only {n_bad} un-na.rm'd mean/sd call(s)")
                return rec
        norm = re.sub(r"\s+", " ", code).strip()
        with self.lock:
            if norm in self.seen:
                rec.update(layer="dedup", info="duplicate normalized code")
                return rec
            self.seen.add(norm)
        record = {
            "intent": obj["intent"],
            "code": code,
            "packages_used": obj["packages_used"],
            "grid_cell": cell,
            "model": src_obj["model"],
            "source": srcname,
            "provider": src_obj["provider"],
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"),
                "valid": True,
                "family": fam,
                "jarl_warnings": jw,
            }
        rec.update(ok=True, layer="ok", record=record, jarl_warnings=jw)
        return rec

    # ---- result handling
    def handle(self, fut):
        rec = fut.result()
        fam = rec["family"]
        with self.lock:
            c = self.counts[fam]
            c["attempts"] += 1
            if rec["ok"]:
                c["accepted"] += 1
                c["lat_sum"] += rec["latency"]
                c["jarl_warn_sum"] += rec.get("jarl_warnings", 0)
                c["lines_sum"] += rec["record"]["code"].count("\n") + 1
                with self.files[fam].open("a") as f:
                    f.write(json.dumps(rec["record"]) + "\n")
                for s in ACTIVE_SOURCES:
                    if s["name"] == rec["source"]:
                        with _SRC_LOCK:
                            s["stats"]["accepted"] += 1
                        break
                self.consecutive_transport_fail = 0
                self.zero_accept_streak = 0
            else:
                c["rejects"][rec["layer"]] = c["rejects"].get(rec["layer"], 0) + 1
                with self.rejects_path.open("a") as f:
                    f.write(json.dumps({
                        "family": fam, "layer": rec["layer"], "info": rec["info"],
                        "source": rec["source"],
                        "at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"),
                        "code": rec.get("record", {}).get("code", "")[:2000],
                    }) + "\n")
                if rec["layer"] == "transport":
                    self.consecutive_transport_fail += 1
                else:
                    self.consecutive_transport_fail = 0
                self.zero_accept_streak += 1

    def stop_needed(self):
        if self.stop_reason:
            return True
        if self.consecutive_transport_fail >= self.args.transport_stop:
            if self.args.wait_on_outage:
                # unattended mode: sit out full-outage windows (e.g. daily
                # free quota exhausted) and resume when a source reopens
                self.outages += 1
                print(f"[outage #{self.outages}] all sources failing for "
                      f"{self.consecutive_transport_fail} jobs; sleeping 600s",
                      flush=True)
                time.sleep(600)
                self.consecutive_transport_fail = 0
                return False
            self.stop_reason = (f"{self.args.transport_stop} consecutive "
                                "transport failures (all sources down or "
                                "key/rate issues)")
            return True
        if self.zero_accept_streak >= self.args.zero_accept_stop:
            self.stop_reason = (f"{self.args.zero_accept_stop} consecutive "
                                "attempts with zero accepts")
            return True
        if self.elapsed_prior + (time.time() - self.t0) > self.args.max_hours * 3600:
            self.stop_reason = f"max wall-clock ({self.args.max_hours}h) reached"
            return True
        return False

    def run(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.resume()
        if not ACTIVE_SOURCES:
            print("no API keys available (OPENCODE_API_KEY / OPENROUTER_API_KEY)",
                  file=sys.stderr)
            sys.exit(2)
        if not shutil.which("jarl"):
            print("[warn] jarl not on PATH; jarl layer will reject as 'internal'",
                  flush=True)
        print(f"[sources] " + ", ".join(
            f"{s['name']} ({s['model']})" for s in ACTIVE_SOURCES), flush=True)
        t_progress = time.time()
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.concurrency) as ex:
            futures = {}

            def fill():
                while len(futures) < self.args.concurrency * 2:
                    job = None if self.stop_needed() else self.next_job()
                    if job is None:
                        break
                    fut = ex.submit(self.run_job, job)
                    futures[fut] = job[0]

            fill()
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    self.write_state()
                    self.print_progress()
                    self._progress_ticks = getattr(self, "_progress_ticks", 0) + 1
                    if self._progress_ticks % 10 == 0:  # ~every 5 min
                        try:
                            (OUT_DIR / "stats.json").write_text(
                                json.dumps(self.build_stats(), indent=1))
                        except Exception:
                            pass
                    t_progress = time.time()
                    continue
                for fut in done:
                    futures.pop(fut)
                    try:
                        self.handle(fut)
                    except Exception as e:
                        print(f"[error] handle(): {e}", flush=True)
                fill()
                if time.time() - t_progress > 60:
                    self.print_progress()
                    t_progress = time.time()
        self.finish()

    def print_progress(self):
        a, n = self.counts["analyst"], self.counts["na_rm"]
        elapsed = self.elapsed_prior + (time.time() - self.t0)
        rate = ((a["accepted"] + n["accepted"]) / elapsed * 3600) if elapsed else 0
        src_bits = []
        for s in ACTIVE_SOURCES:
            st = s["stats"]
            src_bits.append(f"{s['name']}:{st['accepted']}/{st['attempts']}"
                            f"+{st['retries']}r")
        print(f"[{elapsed/60:6.1f}m] analyst {a['accepted']}/{self.args.analyst_n} "
              f"(tries {a['attempts']}, rejects {sum(a['rejects'].values())}) | "
              f"na_rm {n['accepted']}/{self.args.narm_n} "
              f"(tries {n['attempts']}, rejects {sum(n['rejects'].values())}) | "
              f"{rate:.0f} acc/h | {' '.join(src_bits)} | cells "
              f"{self.sampler.drawn} (recyc {self.sampler.recycled}) | "
              f"tf-streak {self.consecutive_transport_fail}", flush=True)

    def finish(self):
        self.write_state()
        stats = self.build_stats()
        (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps(stats, indent=1), flush=True)

    def build_stats(self):
        fams = {}
        for fam, c in self.counts.items():
            acc, att = c["accepted"], c["attempts"]
            fams[fam] = {
                "target": self.args.analyst_n if fam == "analyst"
                else self.args.narm_n,
                "attempts": att,
                "accepted": acc,
                "acceptance_rate": round(acc / att, 3) if att else None,
                "rejected_by_layer": c["rejects"],
                "mean_latency_s": round(c["lat_sum"] / acc, 2) if acc else None,
                "mean_jarl_warnings": round(c["jarl_warn_sum"] / acc, 2)
                if acc else None,
                "mean_code_lines": round(c["lines_sum"] / acc, 1) if acc else None,
            }
        elapsed = self.elapsed_prior + (time.time() - self.t0)
        total_acc = sum(c["accepted"] for c in self.counts.values())
        stats = {
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stop_reason": self.stop_reason or "targets reached",
            "outages_waited_out": self.outages,
            "patience": {"transport_stop": self.args.transport_stop,
                         "zero_accept_stop": self.args.zero_accept_stop,
                         "wait_on_outage": self.args.wait_on_outage},
            "families": fams,
            "sources": {
                s["name"]: {"model": s["model"], "provider": s["provider"],
                            "requests": s["stats"]["requests"],
                            "attempts": s["stats"]["attempts"],
                            "accepted": s["stats"]["accepted"],
                            "retries": s["stats"]["retries"],
                            "empty_content": s["stats"]["empty_content"],
                            "http_errors": dict(s["stats"]["http_errors"])}
                for s in ACTIVE_SOURCES
            },
            "throughput": {
                "elapsed_hours": round(elapsed / 3600, 2),
                "total_accepted": total_acc,
                "accepted_per_hour": round(total_acc / elapsed * 3600, 1)
                if elapsed else None,
            },
            "grid": {
                "cells_drawn": self.sampler.drawn,
                "core_cells": len(grid.DOMAINS) * len(grid.PACKAGES)
                * len(grid.CONSTRUCTS),
                "style_length_resamples": self.sampler.recycled,
                "oneshot_prompt": self.args.oneshot,
            },
            "endpoint_quirks": [
                "opencode.ai (Cloudflare 1010) signature-bans the default "
                "Python-urllib User-Agent; browser UA header required",
                "deepseek-v4-flash-free is a reasoner: reasoning_content consumes "
                "max_tokens before content; 4096 used; at low budgets it returns "
                "200 + empty content",
                "deepseek free tier returns 429 FreeUsageLimitError (no Retry-After) "
                "in bursts; source cooldown 90s + failover used",
                "openrouter z-ai/glm-5.2:free intermittently 503s 'Provider "
                "returned error' (~50%); retried 6x with backoff",
                "dots-studio/dots-3-note-preview:free may emit leading whitespace "
                "before the JSON; stripped before json.loads",
            ],
            "validation": "jsonschema -> R parse (Rscript) -> jarl (hard-fail >=5); "
                          "na_rm adds regex gate >=2 un-na.rm'd mean/sd in "
                          "summarise()",
        }
        return stats


def pilot(args):
    """Small pilot: N analyst + N//4 na_rm attempts, stats to stdout only."""
    gen = Generator(args)
    n_an, n_nr = args.pilot, max(1, args.pilot // 4)
    jobs = []
    for _ in range(n_an):
        cell = gen.sampler.next()
        p = grid.ANALYST_PROMPT.format(**cell)
        if args.oneshot:
            p += ONE_SHOT
        jobs.append(("analyst", cell, p))
    for _ in range(n_nr):
        cell = gen.narm_sampler.next()
        jobs.append(("na_rm", cell, NA_RM_PROMPT.format(**cell)))
    t0 = time.time()
    samples = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency) as ex:
        for i, rec in enumerate(ex.map(gen.run_job, jobs), 1):
            with gen.lock:
                c = gen.counts[rec["family"]]
                c["attempts"] += 1
                if rec["ok"]:
                    c["accepted"] += 1
                    c["lat_sum"] += rec["latency"]
                    c["jarl_warn_sum"] += rec.get("jarl_warnings", 0)
                    c["lines_sum"] += rec["record"]["code"].count("\n") + 1
                    if len(samples) < 6:
                        samples.append((rec["source"], rec["family"],
                                        rec["record"]["intent"]))
                else:
                    c["rejects"][rec["layer"]] = \
                        c["rejects"].get(rec["layer"], 0) + 1
            if rec["layer"] == "transport":
                print("TRANSPORT FAIL:", rec["info"][:150], flush=True)
            if i % 10 == 0:
                print(f"  pilot {i}/{len(jobs)}", flush=True)
    elapsed = time.time() - t0
    stats = gen.build_stats()
    stats["pilot_elapsed_s"] = round(elapsed, 1)
    stats["throughput"]["accepted_per_hour"] = round(
        sum(c["accepted"] for c in gen.counts.values()) / elapsed * 3600, 1)
    print(json.dumps(stats, indent=1))
    print("\n--- sample accepted (source, family, intent) ---")
    for s in samples:
        print(f"  [{s[0]}/{s[1]}] {s[2][:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyst-n", type=int, default=4000)
    ap.add_argument("--narm-n", type=int, default=800)
    ap.add_argument("--concurrency", type=int, default=3,
                    help="max concurrent jobs (be polite: 3)")
    ap.add_argument("--oneshot", action="store_true",
                    help="append one-shot JSON/code example to analyst prompt")
    ap.add_argument("--pilot", type=int, default=0,
                    help="pilot mode: this many analyst attempts, stats only")
    ap.add_argument("--attempt-cap", type=int, default=3,
                    help="max attempts per family, as multiple of its target")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--transport-stop", type=int, default=30,
                    help="consecutive transport failures before graceful stop")
    ap.add_argument("--zero-accept-stop", type=int, default=200,
                    help="consecutive zero-accept attempts before graceful stop")
    ap.add_argument("--wait-on-outage", action="store_true",
                    help="on full outage, sleep 600s and retry instead of "
                         "stopping (for unattended quota-window waits)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore prior state (does NOT delete existing jsonl)")
    args = ap.parse_args()
    args.attempt_cap *= max(args.analyst_n, args.narm_n)

    if not ACTIVE_SOURCES:
        print("no API keys found (OPENCODE_API_KEY / OPENROUTER_API_KEY); "
              "source ~/.zshrc first", file=sys.stderr)
        sys.exit(2)

    if args.pilot:
        pilot(args)
        return
    gen = Generator(args)
    try:
        gen.run()
    except KeyboardInterrupt:
        gen.stop_reason = "interrupted (KeyboardInterrupt)"
        print("\n[interrupt] writing state + stats...", flush=True)
        gen.finish()


if __name__ == "__main__":
    main()
