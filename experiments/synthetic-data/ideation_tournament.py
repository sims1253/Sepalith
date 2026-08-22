#!/usr/bin/env python3
"""ideation_tournament.py — RECURRING cross-model ideation tournament (v2+).

Repo-resident productionization of the one-shot /tmp runner behind
ideation-tournament-v1. Runs one round per invocation; the supervisor
(results/ideation_tournament/run_loop.sh, nohup-detached) is EVENT-PACED
— the next round starts 45 minutes after the previous round finishes
(the cool-down bakes round N's bank/recycle lists into round N+1's brief
and stops fast-fail spins), hard-capped at MAX_ROUNDS_PER_DAY=6 rounds
per rolling 24h, until results/ideation_tournament/STOP_TOURNAMENT
appears.

Round protocol (resumable; state lives in results/ideation_tournament/
rounds/rNNN/, raw model output never deleted):
  1 propose    round1/<L>.json   cast of N models (default 3, ROTATED
                                between rounds so different model triples
                                judge different idea pools) each propose
                                8-10 diverse ideas as JSON; each round's
                                propose brief carries 2-3 AMBIENT DOMAIN
                                SEEDS (real vignette/README snippets from
                                odd CRAN domains, deterministically rotated
                                by round number, cached in
                                domain_seeds.json + ledgered in the round's
                                raw_calls.jsonl — "do not imitate, just let
                                them widen your sense of where R runs")
  2 review     round2/<L>.json   blind NxN rating matrix: every model
                                rates every anonymized proposal 0-5 on
                                signal/novelty/buildability/risk plus
                                ready_to_build (models rate raw — NO
                                separate judge, NO zai/glm: that quota
                                belongs to the data program)
  3 band       bands.json        QUALITY BANDS, not top-N (nothing is
                                discarded):
                                  BUILD   composite >= 4.4 OR unanimous
                                          ready_to_build 5 -> appended to
                                          build_specs.{jsonl,md} (the
                                          wave-2 spec doc) immediately
                                  BANK    3.5 <= composite < 4.4 ->
                                          appended to spec_bank.jsonl
                                          (append-only idea bank; ripens
                                          until infrastructure catches
                                          up, e.g. the wrong-but-passing
                                          probe set)
                                  RECYCLE composite < 3.5 -> kept in the
                                          round's raw JSONL and fed into
                                          the NEXT round's context brief
                                          as "previously proposed and not
                                          selected" (title + one-line why)
                                          so casts push DEPTH, not repeats
  4 deepen     round3/<L>.json   BUILD band (+ top BANK fill to 3, kind
                                diverse) expanded into implementation
                                specs by rotated assignment
  5 re-rate    round3/<L>_reviews.json + aggregate3.json  blind re-rating
                                of the expanded specs; deepened ideas are
                                re-banded on the re-rated composite
  6 finalize   round_summary.json + DONE
  7 triage     triage/triage.json + triage_digest.md + TRIAGE_DONE
                                ox-alpha pass over the round's outputs
                                (primary GO ox-alpha-free; fallback
                                openrouter stealth/ox-alpha, ~50% serve
                                rate): DEDUPE new build specs against the
                                spec bank, all prior build specs and the
                                family inventory (by idea, not wording);
                                MERGE partial overlaps into one spec with
                                both angles; RANK by build-readiness x
                                novelty x distinctness; PROMOTE banked
                                ideas a later round strengthened; emit one
                                compact TRIAGE DIGEST per round (new-build-
                                ready / promoted / deferred-with-reason /
                                deduped-against-what). Humans and the main
                                session read digests, not the spec
                                firehose — which is why the supervisor's
                                6-rounds/day cap is a digest-reading
                                budget and can be raised later.

composite = mean(signal, novelty, buildability, risk) — the same 4-axis
scale as v1, so band thresholds (4.4 / 3.5) are calibrated on v1's actual
score distribution. ready_to_build is the separate unanimity axis.

All API access goes through the existing backend contracts in
cases/backends.py (live-tested configs only; see BACKENDS below). glm/zai
is deliberately absent from every cast. Deterministic seeding: pid/xid
shuffles seeded by round number; cast rotation is a pure function of the
round index. Polite pacing + 429 cooldowns are inherited from the backend
classes; a patient outer loop (20 min/call) plus cast SUBSTITUTION (next
pool backend, logged into cast.json) handles dead quotas — the round
never blocks a whole night on one 429'd model.

Usage (system python3 from experiments/synthetic-data):
  python3 ideation_tournament.py round            # next unfinished round
  python3 ideation_tournament.py round --n 2      # force/resume round 2
  python3 ideation_tournament.py round --cast go-ox,or-gemma,agy-gemini
  python3 ideation_tournament.py brief --n 3      # preview context brief
  python3 ideation_tournament.py status

Env: OPENCODE_API_KEY (zen free + GO tiers), OPENROUTER_API_KEY,
nothing else (agy is a local CLI; zai intentionally unused).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases.backends import (AgyBackend, BackendError, OpencodeBackend,  # noqa: E402
                            OpencodeSparkBackend, OpenrouterBackend)

ROOT = HERE / "results" / "ideation_tournament"
ROUNDS = ROOT / "rounds"
BANK = ROOT / "spec_bank.jsonl"
BUILD_JSONL = ROOT / "build_specs.jsonl"
BUILD_MD = ROOT / "build_specs.md"
TRIAGE_STATUS = ROOT / "triage_status.json"
TRIAGE_DIGESTS = ROOT / "TRIAGE_DIGESTS.md"
V1_AGGREGATE = ROOT / "aggregate.json"          # v1 artifacts live at root
V1_ANONYMIZED = ROOT / "anonymized.json"
V1_WINNERS = ROOT / "winners.json"
V1_ROUND3 = ROOT / "round3" / "anonymized.json"
V1_AGGREGATE3 = ROOT / "aggregate3.json"
STOP_FILE = ROOT / "STOP_TOURNAMENT"

BAND_BUILD = 4.4        # composite >= this OR unanimous ready_to_build 5
BAND_BANK = 3.5         # composite >= this -> BANK
DEEPEN_TARGET = 3       # deepen BUILD band, filled from top BANK to this
PROPOSALS_MAX = 4       # cap on BUILD-band proposals deepened per round
PATIENT_S = 20 * 60     # per-call never-give-up window
ROUND_DEADLINE_S = 4 * 3600
JSON_FAIL_BAIL = 3      # consecutive empty/unparseable replies before
                        # a backend is declared dead for this call

# ---------------------------------------------------------------------------
# Cast pool — every backend below is the live-tested contract from
# cases/backends.py / rewrite_author_spark.py. Pool ORDER fixes the
# rotation: cast(r) = [pool[(3*(r-2)+i) % len] for i in range(3)], so
# round 2 = (go-ox, or-gemma, agy-gemini) — three families, three quotas,
# all orthogonal to whatever data waves are running — and later rounds
# rotate through the rest (zen-xpreview / go-muse share quota with the
# rewrite waves, so they only host rounds when those waves idle).
# ---------------------------------------------------------------------------

class GoOxBackend(OpencodeBackend):
    """ox-alpha-free on the GO subscription tier via chat-completions
    json-mode (rewrite_author_spark.OxAlphaGoBackend contract: separate
    reasoning, subscription rate limits). max_tokens 12000: round 2's
    first propose came back 200-with-empty-content at 6000 — the known
    ox pattern where hidden reasoning burns the completion budget (the
    live-tested oxalpha-12k fix)."""
    name = "go-ox"
    model = "ox-alpha-free"
    url = "https://opencode.ai/zen/go/v1/chat/completions"
    timeout_s = 240.0
    pace_gap_s = 1.5

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 12000,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}


class OrGemmaBackend(OpenrouterBackend):
    """google/gemma-4-31b-it:free on openrouter (v1 tournament model C:
    complete JSON every call, no reasoning param — gemma rejects it)."""
    name = "or-gemma"
    model = "google/gemma-4-31b-it:free"
    timeout_s = 240.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}]}


class AgyGeminiBackend(AgyBackend):
    """gemini-3.7-flash-low via the agy CLI (Google AI Pro quota, fully
    separate from every HTTP quota; prompt MUST go through --prompt —
    handled by the base class)."""
    name = "agy-gemini"
    pace_gap_s = 2.0
    retries = 2


class ZenXpreviewBackend(OpencodeBackend):
    """x-preview-f-free on opencode zen CHAT-completions (v1 model A:
    reasoning_effort low is REQUIRED or reasoning eats the budget and
    content comes back empty)."""
    name = "zen-xpreview"
    model = "x-preview-f-free"
    url = "https://opencode.ai/zen/v1/chat/completions"
    timeout_s = 240.0
    pace_gap_s = 3.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 6000,
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}


class GoMuseBackend(OpencodeSparkBackend):
    """muse-spark-1.2-contributor on the GO tier via the Responses API
    (cases/backends._OpencodeResponsesBackend contract; generous output
    budget because reasoning burns first)."""
    name = "go-muse"
    timeout_s = 300.0
    pace_gap_s = 2.0

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt}]}],
                "max_output_tokens": 8000,
                "reasoning": {"effort": "low"},
                "text": {"format": {"type": "json_object"}}}



class NineSolHighBackend(GoOxBackend):
    """gpt-5.6-sol on HIGH reasoning via the user's ChatGPT Plus
    (9router local proxy) — user-designated tournament proposer (depth).
    Gentle pacing: personal-account tier."""
    name = "gpt56sol-high"
    env_key = "NINE_ROUTER_API_KEY"

    def _payload(self, prompt: str) -> dict:
        return {"model": "cx/gpt-5.6-sol", "max_tokens": 8000,
                "reasoning_effort": "high",
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}

    @property
    def url(self):
        import os
        return os.environ.get("NINE_ROUTER_BASE_URL",
                              "http://localhost:20128").rstrip("/") + "/v1/chat/completions"


CAST_POOL = [GoOxBackend, OrGemmaBackend, AgyGeminiBackend,
             ZenXpreviewBackend, GoMuseBackend, NineSolHighBackend]
LETTERS = "ABC"


class OxStealthBackend(OpenrouterBackend):
    """openrouter stealth/ox-alpha — the TRIAGE fallback only (user
    designation; never a proposing/rating cast member, never pointed at
    data generation). Serves roughly half of requests ("healed to 50%"),
    so the triage caller pushes through failures patiently. max_tokens
    8000: measured ~3k hidden reasoning tokens count against the
    completion budget, so smaller budgets truncate the JSON."""
    name = "ox-stealth"
    model = "stealth/ox-alpha"
    timeout_s = 240.0
    pace_gap_s = 6.5
    json_fail_limit = 6      # ~50% serve rate: empties interleave with
                             # successes, so bail later than the default

    def _payload(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": 8000,
                "messages": [{"role": "user", "content": prompt}]}


def cast_for_round(n: int) -> list[str]:
    """Deterministic 3-model cast rotation over the pool."""
    if n < 2:
        return []
    start = 3 * (n - 2) % len(CAST_POOL)
    return [CAST_POOL[(start + i) % len(CAST_POOL)].name for i in range(3)]


def make_backend(name: str):
    for cls in CAST_POOL:
        if cls.name == name:
            return cls()
    raise KeyError(f"unknown tournament backend {name!r}; "
                   f"known: {[c.name for c in CAST_POOL]}")


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------

def log(rd: Path, msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "recurring.log", "a") as f:
        f.write(line + "\n")
    if rd is not None:
        with open(rd / "round.log", "a") as f:
            f.write(line + "\n")


def load_json(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def dump_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1))


def append_jsonl(p: Path, obj) -> None:
    with open(p, "a") as f:
        f.write(json.dumps(obj) + "\n")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def extract_json(text: str):
    """v1's battle-tested extractor: fence strip + brace slice, then
    repair of truncated JSON (dangling string/key closure)."""
    if not text or not text.strip():
        return None
    s = text.strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1:
        return None
    if j > i:
        try:
            return json.loads(s[i:j + 1])
        except ValueError:
            pass
    t = s[i:]
    t = re.sub(r'"[^"]*$', "", t)
    t = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", t)
    t = t.rstrip().rstrip(",")
    stack, instr, esc = [], False, False
    for ch in t:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"':
            instr = not instr
        elif not instr and ch in "{[":
            stack.append(ch)
        elif not instr and ch in "}]":
            if stack:
                stack.pop()
    t += "".join("}" if c == "{" else "]" for c in reversed(stack))
    try:
        return json.loads(t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# patient call layer (pacing + 429 backoff live in the backend classes)
# ---------------------------------------------------------------------------

class RoundCtx:
    """Per-round state: cast (letters -> backend instances + substitution
    history), raw-output ledger, clock."""

    def __init__(self, n: int, rd: Path, cast_names: list[str]):
        self.n, self.rd = n, rd
        self.t0 = time.time()
        cast_p = rd / "cast.json"
        prev = load_json(cast_p, {}) or {}
        self.letters = dict(prev.get("letters") or
                            {L: cast_names[i] for i, L in enumerate(LETTERS)})
        self.history = list(prev.get("substitutions") or [])
        self.backends: dict[str, object] = {}
        self._write_cast(prev.get("requested_cast") or cast_names)

    def _write_cast(self, requested) -> None:
        dump_json(self.rd / "cast.json", {
            "round": self.n,
            "requested_cast": requested,
            "letters": self.letters,
            "substitutions": self.history,
            "note": "letters are the anonymized identities used in every "
                    "prompt/rating; real backend names live only here"})

    def backend(self, L: str):
        if L not in self.backends or self.backends[L].get("name") != \
                self.letters[L]:
            b = make_backend(self.letters[L])
            self.backends[L] = {"name": b.name, "obj": b,
                                "model": getattr(b, "model", "?")}
        return self.backends[L]

    def substitute(self, L: str, why: str) -> bool:
        used = set(self.letters.values())
        for cls in CAST_POOL:
            if cls.name not in used:
                old = self.letters[L]
                self.letters[L] = cls.name
                self.backends.pop(L, None)
                self.history.append({"letter": L, "replaced": old,
                                     "with": cls.name, "why": why,
                                     "at": time.strftime("%F %T")})
                self._write_cast(self.letters)
                log(self.rd, f"SUBSTITUTE {L}: {old} -> {cls.name} ({why})")
                return True
        return False

    def record_raw(self, L: str, stage: str, raw: str, ok: bool) -> None:
        be = self.backends.get(L, {})
        append_jsonl(self.rd / "raw_calls.jsonl", {
            "ts": time.strftime("%F %T"), "round": self.n, "letter": L,
            "stage": stage, "backend": be.get("name"),
            "model": be.get("model"), "ok": ok,
            "chars": len(raw or ""), "raw": (raw or "")[:40000]})

    def left(self) -> float:
        return ROUND_DEADLINE_S - (time.time() - self.t0)

    def stats(self) -> dict:
        out = {}
        for L, be in self.backends.items():
            obj = be.get("obj")
            if obj is not None:
                out[f"{L}={be['name']}"] = obj.stats_summary()
        return out


def patient_call(ctx: RoundCtx, L: str, stage: str, prompt: str,
                 want_json: bool = True) -> str | None:
    """One model call that never gives up on 429s inside PATIENT_S. The
    backend classes already pace, cooldown and retry; this layer adds the
    long-haul loop, escalating sleeps, and an early bail after
    JSON_FAIL_BAIL consecutive empty/unparseable replies (empty content
    is a payload problem — retrying the same payload forever just burns
    the round; bail out so substitution can happen)."""
    deadline = time.time() + PATIENT_S
    attempt = 0
    json_fails = 0
    while time.time() < deadline and ctx.left() > 60:
        be = ctx.backend(L)
        try:
            raw = be["obj"].complete(prompt)
            if raw and raw.strip():
                ctx.record_raw(L, stage, raw, True)
                return raw
            json_fails += 1
            if json_fails >= JSON_FAIL_BAIL:
                log(ctx.rd, f"{L}: {stage}: {json_fails} consecutive "
                            f"empty replies; bailing to substitution")
                return None
            log(ctx.rd, f"{L}: empty output ({stage}), retrying")
            time.sleep(10)
        except BackendError as e:
            ctx.record_raw(L, stage, str(e), False)
            wait = min(600.0, 60.0 * (attempt + 1)) if e.kind == "rate" else 5.0
            if e.kind == "json":
                json_fails += 1
                if json_fails >= JSON_FAIL_BAIL:
                    log(ctx.rd, f"{L}: {stage}: {json_fails} consecutive "
                                f"json errors; bailing to substitution")
                    return None
            log(ctx.rd, f"{L}: {stage} {e.kind}: {str(e)[:120]}; "
                        f"sleep {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:  # defensive: never kill the round
            log(ctx.rd, f"{L}: {stage} unexpected {type(e).__name__}: {e}")
            time.sleep(15)
        attempt += 1
    return None


def call_json(ctx: RoundCtx, L: str, stage: str, prompt: str,
              expected_key: str) -> dict | None:
    """Patient call + robust JSON extraction + one strict re-ask, then
    substitution, then (if no substitute left) signal failure upstream."""
    base = prompt
    for phase in range(2):
        p = base if phase == 0 else base + (
            "\n\nIMPORTANT: reply with a single valid JSON object only, "
            "no prose, no markdown fences. Keep each description under 60 "
            "words so the JSON fits your output budget.")
        raw = patient_call(ctx, L, stage, p)
        if raw is None:
            break
        obj = extract_json(raw)
        if isinstance(obj, list):          # model returned the bare array
            obj = {expected_key: obj}
        if isinstance(obj, dict) and expected_key in obj and \
                isinstance(obj[expected_key], list) and obj[expected_key]:
            return obj
        if isinstance(obj, dict) and expected_key in obj and \
                isinstance(obj[expected_key], dict):
            return obj
        log(ctx.rd, f"{L}: {stage} JSON parse fail (phase {phase})")
    if ctx.substitute(L, f"{stage}: no usable JSON after retries"):
        return call_json(ctx, L, stage, prompt, expected_key)
    return None


# ---------------------------------------------------------------------------
# context brief (compact: 2 paragraphs + memory lists + depth push)
# ---------------------------------------------------------------------------

SEPALITH_PARA = """CONTEXT BRIEF (Sepalith project — read carefully before proposing):

Sepalith builds an R "next-edit" coding model (token-level R code
completion/editing). Training data comes from a synthetic-data + RL
program whose eval philosophy is validators > judges > line-F1: ground
truth must be derivable deterministically wherever possible."""

INVENTORY_PARA = """Existing asset families: (1) scenario cases for SFT —
synthetic multi-file repo scenarios with a natural next-edit problem
(rename propagation, base-R<->dplyr pipe refactors, format/lint fixes,
doc_sync where Roxygen must follow a code edit, na_rm idioms, no_op
restraint cases where the RIGHT edit is changing nothing); (2) the
rewrite program — lint-fix and fix-issue rewrites of real CRAN code plus
"verified rewrites"; (3) compounding base-samples + a deterministic
rule registry (real corpus code transformed by invertible rules, ground
truth derived WITHOUT a judge); (4) an RL keystroke environment over R
files rewarded by deterministic validators; (5) a 14,202-package CRAN
corpus incl. vignettes, tests, man pages."""

DEPTH_PARA = """We value DEPTH over breadth and originality over safe
variations of what we already have. The lists above are the
tournament's memory: re-proposing a listed idea (or a cosmetic variant
of it) will be rated 0 for novelty. Where a listed idea was rejected
for being shallow or risky, a proposal that fixes the specific weakness
and goes materially deeper IS welcome — say explicitly what you fix."""


def _v1_memory() -> dict:
    agg = load_json(V1_AGGREGATE) or []
    won = {r.get("expands_pid") for r in (load_json(V1_AGGREGATE3) or [])[:3]}
    build, bank, recycle = [], [], []
    for r in agg:
        c = r.get("composite")
        if r.get("pid") in won:
            build.append((r["title"], "v1 winner, wave-2 spec written"))
        elif isinstance(c, (int, float)) and c >= BAND_BANK:
            bank.append((r["title"], f"v1 composite {c}"))
        else:
            why = ""
            rats = r.get("rationales") or {}
            if rats:
                why = min(rats.values(), key=len)
            recycle.append((r["title"],
                            (why or f"composite {c}")[:110]))
    return {"round": 1, "build": build, "bank": bank, "recycle": recycle}


def prior_rounds_memory(n: int) -> list[dict]:
    mem = []
    for r in range(2, n):
        s = load_json(ROUNDS / f"r{r:03d}" / "round_summary.json")
        if not s or not (ROUNDS / f"r{r:03d}" / "DONE").exists():
            continue
        mem.append({
            "round": r,
            "build": [(b["title"], f"round-{r} composite "
                                   f"{b.get('composite_final')}")
                      for b in s.get("bands", {}).get("BUILD", [])],
            "bank": [(b["title"], f"round-{r} composite {b.get('composite')}")
                     for b in s.get("bands", {}).get("BANK", [])],
            "recycle": [(x["title"], x["why"])
                        for x in s.get("bands", {}).get("RECYCLE", [])]})
    return mem


def build_brief(n: int) -> str:
    memory = ([_v1_memory()] if n >= 2 else []) + prior_rounds_memory(n)
    builds, banks, recycles = [], [], []
    for m in memory:
        builds += m["build"]
        banks += m["bank"]
        recycles += m["recycle"]
    # fold triage outcomes into the memory lists so later casts see them
    ts = load_triage_status().get("titles") or {}
    for info in ts.values():
        if info.get("status") == "promoted":
            builds.append((info["title"],
                           f"promoted from bank by triage (round "
                           f"{info.get('round')})"))
        elif info.get("status") == "demoted":
            banks.append((info["title"],
                          f"triage: {info.get('note', 'deduped')}"))
    parts = [SEPALITH_PARA, "", INVENTORY_PARA, "",
             "TOURNAMENT MEMORY (from earlier rounds — respect it):"]
    if builds:
        parts.append("- Already selected for BUILD / wave-2 specs "
                     "(do NOT re-propose):")
        parts += [f"  * {t} — {w}" for t, w in builds[:12]]
    if banks:
        parts.append("- BANKED, waiting for infrastructure to catch up "
                     "(only re-propose with a materially deeper take):")
        parts += [f"  * {t} ({w})" for t, w in banks[-18:]]
    if recycles:
        parts.append("- Previously proposed and NOT selected — title plus "
                     "the one-line why; do not re-propose:")
        parts += [f"  * {t} — {w}" for t, w in recycles[-24:]]
    parts += ["", DEPTH_PARA]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ambient domain seeds — real-world artifacts from ODD corners of the corpus
# (user idea 2026-08-20): each round's PROPOSE brief carries 2-3 short
# snippets pulled deterministically (seeded by round number) from domains
# OUTSIDE the tidyverse/biostat neighborhood, framed as "do not imitate".
# Cached in results/ideation_tournament/domain_seeds.json so a round's
# seeds are stable and auditable; every snippet is also appended to the
# round's raw_calls.jsonl ledger for provenance.
# ---------------------------------------------------------------------------

NORMALIZED_CORPUS = Path("/mnt/h/sepalith/normalized")
DOMAIN_SEEDS_CACHE = ROOT / "domain_seeds.json"
MARIN_SEEDS_SRC = Path("/tmp/marin_seeds.jsonl")   # volatile; snapshotted
SEED_MAX_LINES = 40          # snippet hard cap
SEED_SCAN_CAP = 600          # DESCRIPTION reads per scan pass (drvfs-polite)

DOMAIN_KEYWORDS = {          # insertion order == the rotation order
    # original 18
    "geotechnics/tunnel": ("geotechn", "rock mechanics", "tunnel",
                           "pile foundation", "slope stability", "soil"),
    "audio/synthesis/music": ("audio", "acoustic", "music", "speech",
                              "synthes", "midi", "pitch detection", "sound"),
    "hardware/microcontroller": ("arduino", "i2c", "spi bus", "gpio",
                                 "sensor", "microcontroller", "serial port",
                                 "embedded"),
    "web/api": ("http", "api client", "url", "rest", "scrap", "curl",
                "websocket"),
    "finance": ("finance", "financial", "trading", "portfolio",
                "option pricing", "stock", "bond", "accounting", "risk"),
    "ecology": ("ecolog", "species distribution", "population dynamics",
                "biodiversity", "habitat", "metabarcoding", "forest"),
    "genomics": ("genom", "gene expression", "transcriptom", "sequencing",
                 "snp", "variant call", "phylogen", "bioinformat"),
    "astronomy": ("astro", "star", "planet", "celestial", "telescope",
                  "light curve", "ephemeris"),
    "game/simulation": ("game theory", "agent-based", "cellular automaton",
                        "simulation of", "chess", "game-playing"),
    "gis/spatial": ("gis", "spatial", "raster", "shapefile", "coordinate",
                    "cartogra", "terrain", "map"),
    "pharma/clinical": ("clinical trial", "drug", "pharmacokinetic",
                        "adverse event", "dose response", "bioequivalence"),
    "epidemiology": ("epidemi", "outbreak", "incidence", "epidemic",
                     "seroprev", "contact tracing"),
    "weather/climate": ("weather", "climate", "meteorolog", "rainfall",
                        "atmospheric", "temperature record"),
    "hydrology": ("hydrolog", "watershed", "streamflow", "groundwater",
                  "river", "rainfall-runoff"),
    "chemometrics/spectroscopy": ("spectrosc", "chromatograph",
                                  "mass spectrometry", "chemomet",
                                  "spectra", "near-infrared"),
    "psychometrics": ("psychometr", "item response", "likert",
                      "cognitive test", "questionnaire scale"),
    "sports/analytics": ("sport", "football", "soccer", "athlete",
                         "basketball", "batting"),
    "ocean/marine": ("ocean", "marine", "fisheries", "sea surface",
                     "tidal", "bathymetr"),
    # marin-harrier K=40-anchored additions (2026-08-20 scout delivery):
    # clusters like Finance/Insurance, Natural Sciences, Consumer Devices,
    # Law/Regulation, Food/Gardening mapped onto CRAN-present veins
    "insurance/actuarial": ("actuar", "insurance", "claim severity",
                            "loss distribution", "premium", "mortality table"),
    "agriculture": ("agricultur", "crop yield", "field trial", "farm survey",
                    "dairy", "plant breeding", "cultivar"),
    "energy/grid": ("energy", "power grid", "photovoltaic", "wind turbine",
                    "electricity", "load forecasting"),
    "transport/traffic": ("traffic", "transit", "transport", "road network",
                          "vehicle", "logistics"),
    "text-mining/nlp": ("text mining", "natural language", "tokeniz",
                        "sentiment", "topic model", "corpus linguist"),
    "imaging/microscopy": ("microscopy", "image analysis", "segmentation",
                           "bioimag", "pixel", "tomograph"),
    "seismic/geophysics": ("seismic", "seismolog", "earthquake",
                           "geophysic", "magnetotelluric"),
    "econometrics/forecasting": ("economet", "forecasting", "seasonal",
                                 "arima", "garch", "causal inference"),
}

# fallback: my domain -> marin harrier K=40 cluster titles that carry
# R/R-adjacent seed docs (snapshot below); used when a domain is thin or
# unmatched in the CRAN corpus
MARIN_CLUSTER_FALLBACK = {
    "hydrology": ("Natural Sciences Research",),
    "astronomy": ("Natural Sciences Research",),
    "ocean/marine": ("Natural Sciences Research",),
    "game/simulation": ("Mathematics Problems and Proofs",),
    "hardware/microcontroller": ("Command-Line Agent Transcripts",
                                 "Home Maintenance and Trades"),
    "geotechnics/tunnel": ("Home Maintenance and Trades",
                           "Natural Sciences Research"),
    "psychometrics": ("Study Skills and Academic Writing",),
    "web/api": ("Software Development and Web Code",
                "Application and Web Development"),
    "finance": ("Business Software and Strategy",),
    "agriculture": ("Pets, Gardening, and Pest Control", "Food and Recipes"),
    "epidemiology": ("Government Notices and Environment",),
    "energy/grid": ("Government Notices and Environment",),
    "text-mining/nlp": ("General Knowledge and Trivia",),
    "pharma/clinical": ("Clinical Treatment and Dentistry",),
    "sports/analytics": ("Racing, Sports, and Entertainment Records",),
    "seismic/geophysics": ("Natural Sciences Research",),
    "imaging/microscopy": ("Natural Sciences Research",),
    "ecology": ("Natural Sciences Research",),
    "chemometrics/spectroscopy": ("Natural Sciences Research",),
}
SEED_FRAME = """AMBIENT DIVERSITY SEEDS — do not imitate, just let them widen
your sense of where R runs. Below are real artifacts from {ndom} CRAN
{domword} far outside the tidyverse/biostat neighborhood (round {n} rotation:
{doms}). They are NOT a template, NOT a topic request and NOT a quality
bar: do not propose anything about these specific packages. Let them remind
you that R users also work in tunnels, clinics, trading desks, forests, at
telescopes and in hobby basements — propose idea families that would train
a next-edit model for THOSE users too."""


def _load_seed_cache() -> dict:
    c = load_json(DOMAIN_SEEDS_CACHE) or {}
    c.setdefault("rounds", {})
    c.setdefault("index", {})        # domain -> [package, ...] (grows lazily)
    c.setdefault("used", {})         # domain -> [package, ...] already seeded
    c.setdefault("missed", [])       # domains with no corpus match (no rescan)
    c.setdefault("marin", {})        # cluster -> [seeddoc, ...] snapshot
    c.setdefault("marin_used", [])   # "cluster:i" ids already seeded
    return c


def _snapshot_marin(cache: dict) -> None:
    """One-time snapshot of the volatile /tmp marin harrier seed docs into
    the cache (28 R/R-adjacent docs across 16 K=40 clusters, delivered
    2026-08-20). Prompt-context provenance only — never training data."""
    if cache["marin"] or not MARIN_SEEDS_SRC.exists():
        return
    try:
        for line in MARIN_SEEDS_SRC.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            cache["marin"].setdefault(d.get("title", "?"), []).append({
                "quality": d.get("quality"), "reason": d.get("reason"),
                "source": d.get("source"), "source_url": d.get("source_url"),
                "license_note": d.get("license_note"),
                "text": (d.get("text") or "")[:4000]})
        log(None, f"domain-seeds: snapshotted {sum(len(v) for v in cache['marin'].values())} "
                  f"marin harrier seed docs into the cache")
    except (OSError, ValueError) as e:
        log(None, f"domain-seeds: marin snapshot failed ({e}); continuing")


def _pkg_list() -> list[str]:
    try:
        return sorted(p.name for p in NORMALIZED_CORPUS.iterdir()
                      if p.is_dir() and re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*",
                                                     p.name))
    except OSError:
        return []


def _pkg_text(pkg: str) -> tuple[str, str]:
    """(title part, full head) of a package's DESCRIPTION, lowercased."""
    try:
        base = NORMALIZED_CORPUS / pkg
        ver = sorted((d.name for d in base.iterdir() if d.is_dir()))[-1]
        desc = base / ver / pkg / "DESCRIPTION"
        with open(desc, encoding="utf-8", errors="replace") as f:
            text = f.read(2000).lower()
        title = text[:text.find("description:")] if "description:" in text \
            else text[:400]
        return title, text
    except (OSError, IndexError):
        return "", ""


def _domain_match(domain_kws: tuple, title: str, text: str) -> bool:
    """A keyword hit on the Title, or TWO distinct word-boundary keyword
    hits in the whole head — plain substring matching lets 'synthes'
    (evidence synthesis) and 'sound' (grounded...) steal packages from
    the wrong domain."""
    pats = [re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])")
            for k in domain_kws]
    if any(p.search(title) for p in pats):
        return True
    return sum(1 for p in pats if p.search(text)) >= 2


def _find_artifact(pkg: str) -> tuple[str, list[str]] | None:
    """(relpath, first lines) of a README or vignette, or None."""
    try:
        base = NORMALIZED_CORPUS / pkg
        ver = sorted((d.name for d in base.iterdir() if d.is_dir()))[-1]
        root = base / ver / pkg
        cands = [root / "README.md"]
        cands += sorted(root.glob("README*"))
        cands += sorted((root / "vignettes").glob("*.Rmd"))
        cands += sorted((root / "vignettes").glob("*.md"))
        for p in cands:
            if p.suffix.lower() in (".pdf", ".html", ".doc") or not p.is_file():
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                lines = f.read(40000).splitlines()
            if len(lines) >= 6:
                rel = str(p.relative_to(NORMALIZED_CORPUS))
                return rel, lines
    except (OSError, IndexError):
        pass
    return None


_JUNK = ("<!--", "[![", "<img", "<a ", "<p>", ")", "[x]", "[ ]",
         "status]", "state]", "developed.]", "https://", "http://")


def _debadge(lns: list[str]) -> list[str]:
    """Trim README badge/comment noise off both ends of a window (a window
    boundary can land mid-badge, so badge CONTINUATION lines count too)."""
    def junk(ln):
        t = ln.lstrip()
        return (not t.strip()) or t.startswith(_JUNK) or \
            (t.startswith("[") and "](" in t)
    a, b = 0, len(lns)
    while a < b and junk(lns[a]):
        a += 1
    while b > a and junk(lns[b - 1]):
        b -= 1
    return lns[a:b]


def _snippet(lines: list[str], max_lines: int = SEED_MAX_LINES) -> list[str]:
    """A <=max_lines window, preferring prose that introduces an R chunk."""
    out = []
    i = 0
    if lines and lines[0].strip() == "---":        # YAML front matter
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    # skip title + blank noise
    while i < len(lines) and (lines[i].startswith("#") or
                              not lines[i].strip()):
        i += 1
    fence = next((j for j in range(i, min(i + 220, len(lines)))
                  if lines[j].lstrip().startswith(("```{r", "```r",
                                                   "```{R"))), None)
    if fence is not None:
        start = max(i, fence - 5)                   # a little prose context
        end = next((j for j in range(fence + 1, len(lines))
                    if lines[j].lstrip().startswith("```")), None)
        end = min(end if end else len(lines), start + max_lines)
        out = lines[start:end]
    else:
        out = [ln for ln in lines[i:i + 3 * max_lines] if ln.strip() or
               (out and out[-1].strip())][:max_lines]
    out = _debadge([ln for ln in out if len(ln) <= 400])
    if len(out) < 4:                                # badge-heavy head
        out = _debadge([ln for ln in lines if len(ln) <= 400
                        and not ln.lstrip().startswith(_JUNK)
                        and ln.strip()])[:max_lines]
    out = [ln[:120] + ("…" if len(ln) > 120 else "") for ln in out]
    return out[:max_lines]


def _scan_domains(need: dict[str, tuple], n: int, cache: dict,
                  per_domain: int = 6) -> None:
    """One drvfs-polite pass over a seeded-shuffle of packages, filling
    cache['index'][domain] for every domain in `need` that is still empty.
    The pass matches ALL still-unindexed domains against each DESCRIPTION
    read (one read amortized over the whole keyword list), so the expensive
    corpus pass effectively happens once per cache lifetime."""
    todo = {d for d, kws in need.items()
            if not cache["index"].get(d) and kws
            and d not in cache.get("missed", [])}
    if not todo:
        return
    log(None, f"domain-seeds: first corpus index scan for {sorted(todo)} "
              f"(cap {SEED_SCAN_CAP} DESCRIPTION reads; /mnt/h can be slow "
              f"under wave load)")
    pkgs = _pkg_list()
    rng = random.Random("sepalith-domain-seeds-scan-v1:"
                        + "|".join(sorted(todo)) + f":{n}")
    rng.shuffle(pkgs)
    for pkg in pkgs[:SEED_SCAN_CAP]:
        if not todo:
            break
        title, text = _pkg_text(pkg)
        if not text:
            continue
        for d in sorted(todo):
            if _domain_match(need[d], title, text):
                cache["index"].setdefault(d, []).append(pkg)
                if len(cache["index"][d]) >= per_domain:
                    todo.discard(d)
    # only domains with ZERO candidates count as missed (partial index
    # entries are usable seeds; they must not be discarded just because
    # the scan cap hit before the domain filled to per_domain)
    empty = {d for d in todo if not cache["index"].get(d)}
    if empty:
        cache.setdefault("missed", []).extend(sorted(empty))
        log(None, f"domain-seeds: no corpus match after {SEED_SCAN_CAP} "
                  f"packages for: {sorted(empty)} (will not rescan)")


def _marin_seed(d: str, cache: dict, rng) -> dict | None:
    """One unused marin harrier seed doc for domain d (fallback when the
    CRAN corpus has no match for the domain)."""
    for title in MARIN_CLUSTER_FALLBACK.get(d, ()):
        for i, doc in enumerate(cache["marin"].get(title, [])):
            key = f"{title}:{i}"
            if key in cache["marin_used"]:
                continue
            snip = _snippet(doc["text"].splitlines())
            if len(snip) < 3:
                continue
            cache["marin_used"].append(key)
            return {"domain": d, "package": f"marin:{title}",
                    "path": doc.get("source_url") or doc.get("source", "?"),
                    "lines": len(snip), "snippet": "\n".join(snip),
                    "source": "marin-harrier-k40",
                    "license_note": doc.get("license_note", "")}
    return None


def domain_seeds_for_round(n: int) -> list[dict]:
    """2-3 seeds from a DIFFERENT domain-set than round n-1; deterministic
    given the round number and the cache state. The outcome — seeds or no
    seeds — is FROZEN in the cache on first sampling, so the brief preview,
    the round itself and every resume see the identical seeds."""
    cache = _load_seed_cache()
    # NB: rounds keys are STRINGS (json round-trips object keys as str);
    # an int lookup silently misses and would re-sample the round.
    frozen = cache["rounds"].get(str(n))
    if isinstance(frozen, dict) and "seeds" in frozen:
        return frozen["seeds"]
    _snapshot_marin(cache)
    rng = random.Random(f"sepalith-domain-seeds-v1:{n}")
    prev = set()
    for s in (cache["rounds"].get(str(n - 1)) or {}).get("seeds", []):
        prev.add(s["domain"])

    def has_seed_path(d):      # corpus candidates OR a marin fallback doc
        return bool(cache["index"].get(d)) or any(
            cache["marin"].get(t) for t in MARIN_CLUSTER_FALLBACK.get(d, ()))

    pool = [d for d in DOMAIN_KEYWORDS if d not in prev and has_seed_path(d)]
    pool = pool or [d for d in DOMAIN_KEYWORDS if has_seed_path(d)] or \
        list(DOMAIN_KEYWORDS)
    want = rng.sample(pool, min(2 + rng.randrange(2), len(pool)))
    _scan_domains(DOMAIN_KEYWORDS, n, cache)   # fill every empty domain once
    seeds, used_all = [], set()
    for d in want:
        idx = [p for p in cache["index"].get(d, [])
               if p not in (cache["used"].get(d) or [])]
        if not idx:
            idx = list(cache["index"].get(d, []))
            cache["used"][d] = []
        rng.shuffle(idx)
        got = None
        for pkg in [p for p in idx if p not in used_all] or idx:
            art = _find_artifact(pkg)
            if not art:
                continue
            rel, lines = art
            snip = _snippet(lines)
            if len(snip) < 4:
                continue
            got = (pkg, rel, snip)
            break                   # first candidate with a usable artifact
        if got is not None:
            pkg, rel, snip = got
            used_all.add(pkg)
            cache["used"].setdefault(d, []).append(pkg)
            seeds.append({"domain": d, "package": pkg, "path": rel,
                          "lines": len(snip), "snippet": "\n".join(snip),
                          "source": "cran-corpus"})
        else:
            m = _marin_seed(d, cache, rng)
            if m:
                seeds.append(m)
    cache["rounds"][str(n)] = {"round": n,
                               "domains": [s["domain"] for s in seeds],
                               "sampled": time.strftime("%F %T"),
                               "seeds": seeds}
    dump_json(DOMAIN_SEEDS_CACHE, cache)
    return seeds


def seeds_paragraph(n: int) -> str:
    seeds = domain_seeds_for_round(n)
    if not seeds:
        return ""
    head = SEED_FRAME.format(n=n, ndom=len(seeds),
                             domword="domain" if len(seeds) == 1 else "domains",
                             doms=", ".join(s["domain"] for s in seeds))
    parts = [head]
    for i, s in enumerate(seeds, 1):
        origin = (" — marin harrier K=40 sample" if
                  s.get("source") == "marin-harrier-k40" else "")
        parts.append(f"\n[seed {i}/{len(seeds)} — domain: {s['domain']} — "
                     f"{s['package']} — {s['path']} "
                     f"({s['lines']} lines{origin})]\n{s['snippet']}")
    return "\n".join(parts)


def attach_domain_seeds(ctx: RoundCtx) -> None:
    """Compute round n's seeds, ledger them into raw_calls.jsonl (once —
    resume-safe), and stash the paragraph on the ctx for the propose stage.
    Never fatal: a corpus hiccup just means an unseeded brief."""
    try:
        para = seeds_paragraph(ctx.n)
        ctx.seeds_para = para
        if not para:
            return
        ledger = ctx.rd / "raw_calls.jsonl"
        done = False
        if ledger.exists():
            with open(ledger, encoding="utf-8", errors="replace") as f:
                done = any('"stage": "domain_seeds"' in ln for ln in f)
        if done:
            return
        for s in domain_seeds_for_round(ctx.n):
            append_jsonl(ledger, {
                "ts": time.strftime("%F %T"), "round": ctx.n, "letter": "-",
                "stage": "domain_seeds",
                "backend": ("marin-harrier-k40 (prompt-context only)"
                            if s.get("source") == "marin-harrier-k40"
                            else "corpus:/mnt/h/sepalith/normalized"),
                "model": f"domain:{s['domain']}/{s['package']}", "ok": True,
                "chars": len(s["snippet"]),
                "raw": f"[{s['domain']}] {s['package']} :: {s['path']}\n"
                       f"{s['snippet']}"[:40000]})
        log(ctx.rd, f"domain-seeds: round {ctx.n} carries "
                    f"{len(domain_seeds_for_round(ctx.n))} ambient seeds")
    except Exception as e:                     # noqa: BLE001
        log(ctx.rd, f"domain-seeds: FAILED ({type(e).__name__}: {e}); "
                    f"round proceeds unseeded")
        ctx.seeds_para = ""


def prompt_propose(n: int, seeds_para: str | None = None) -> str:
    brief = build_brief(n)
    if seeds_para:
        brief += "\n\n" + seeds_para
    return brief + f"""

TASK (tournament round {n}): Propose 8 to 10 DIVERSE ideas (different
kinds, not variations of one idea). At least one should be a
deterministic registry rule or derivation, at least one a whole new
family/generator, at least one an RL environment or reward idea, and at
least one should be something we did NOT ask for (your own wild-card
category). Go DEEP on the 2 you find most promising.

Reply with a single JSON object, no prose:
""" + '{"proposals": [\n' + """
  {"n": 1, "title": "...", "kind": "one of: scenario-family|registry-rule|pretraining-code|no-op-restraint|rl-env|reward-design|wild-card|...",
   "description": "3-8 sentences, concrete mechanics",
   "why_strong": "why this is a strong training signal for a next-edit model",
   "data_shape": "row schema / generator pipeline sketch",
   "yield_cost": "estimated rows and compute cost",
   "determinism": "deterministic | validator-checkable | judge-needed"}
]}"""


REVIEW_AXES = ("signal", "novelty", "buildability", "risk",
               "ready_to_build")


def prompt_review(brief: str, items: list[dict]) -> str:
    lines = []
    for it in items:
        lines.append(
            f"### {it['pid']}\nTitle: {it['title']}\nKind: {it.get('kind','?')}\n"
            f"Description: {it.get('description','')}\n"
            f"Why strong: {it.get('why_strong','')}\n"
            f"Data shape: {it.get('data_shape','')}\n"
            f"Yield/cost: {it.get('yield_cost','')}\n"
            f"Determinism: {it.get('determinism','?')}")
    return brief + """

TASK: Below are anonymized proposals for the Sepalith synthetic-data
program. Rate EACH proposal 0-5 on five axes:
- signal: strength of training signal for a next-edit R model (0=weak,5=excellent)
- novelty: novelty RELATIVE TO what we already have AND the tournament
  memory list above (0=dup of existing/listed,5=new vein)
- buildability: how cheaply/robustly we can build it (0=very hard,5=trivial)
- risk: risk it teaches the wrong thing or is gamed (0=high risk,5=very safe)
- ready_to_build: could a competent engineer start implementing from the
  proposal alone tomorrow? (0=vague,5=fully specified)
Also give a one-line rationale. Be honest and critical — do not flatter.
Reply with a single JSON object, no prose:
{"ratings": [{"pid": "P-001", "signal": 0-5, "novelty": 0-5, "buildability": 0-5, "risk": 0-5, "ready_to_build": 0-5, "rationale": "..."}]}

PROPOSALS:

""" + "\n\n".join(lines)


def prompt_deepen(brief: str, idea: dict) -> str:
    return brief + f"""

TASK: This proposal survived a review tournament. Expand it into a full
implementation-ready spec. Include: exact row schema (field names+types),
validator design (deterministic where possible), source data + generator
pipeline steps, one to three HAND-BUILT example rows (real R code snippets
you write now, showing input context and expected edit), failure modes,
and estimated first-wave size.

Proposal:
Title: {idea['title']}
Kind: {idea.get('kind','?')}
Description: {idea.get('description','')}
Why strong: {idea.get('why_strong','')}
Data shape: {idea.get('data_shape','')}

Reply with a single JSON object, no prose:
""" + '{"title": "...", "spec": {"row_schema": {...}, "validator": "...", "pipeline": [...], "example_rows": [...], "failure_modes": "...", "first_wave": "..."}}'


def prompt_rerate(brief: str, flat: list[dict]) -> str:
    lines = []
    for e in flat:
        lines.append(f"### {e['xid']}\nTitle: {e.get('title')}\n"
                     f"Spec: {json.dumps(e.get('spec'))[:3000]}")
    return brief + """

TASK: These are expanded specs (anonymized) for new synthetic-data ideas.
Rate each 0-5 on: signal, novelty, buildability, risk (5=safe), plus
ready_to_build (0-5: is the spec complete enough to implement tomorrow?)
and a one-line rationale. JSON only:
{"ratings": [{"xid": "X-001", "signal": 0-5, "novelty": 0-5, "buildability": 0-5, "risk": 0-5, "ready_to_build": 0-5, "rationale": "..."}]}

SPECS:

""" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_propose(ctx: RoundCtx) -> dict:
    out = {}
    for L in LETTERS:
        p = ctx.rd / "round1" / f"{L}.json"
        if p.exists():
            out[L] = load_json(p)
            log(ctx.rd, f"round1 {L}: cached "
                        f"({len(out[L].get('proposals', []))} ideas)")
            continue
        log(ctx.rd, f"round1 {L} ({ctx.letters[L]}): proposing...")
        obj = call_json(ctx, L, "propose",
                        prompt_propose(ctx.n,
                                       getattr(ctx, "seeds_para", None)),
                        "proposals")
        if obj is None:
            log(ctx.rd, f"round1 {L}: FAILED (no backend produced JSON)")
            continue
        obj["_backend"] = ctx.letters[L]
        p.parent.mkdir(exist_ok=True)
        dump_json(p, obj)
        out[L] = obj
        log(ctx.rd, f"round1 {L}: got {len(obj.get('proposals', []))} ideas")
    return out


def anonymize(ctx: RoundCtx, r1: dict) -> list[dict]:
    rng = random.Random(1000 * ctx.n + 42)
    items = []
    for L in LETTERS:
        for pr in (r1.get(L) or {}).get("proposals", []):
            if isinstance(pr, dict) and pr.get("title"):
                it = dict(pr)
                for k in ("n", "id"):
                    it.pop(k, None)
                it["author"] = L
                items.append(it)
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["pid"] = f"P-{i:03d}"
    return items


def stage_review(ctx: RoundCtx, brief: str, items: list[dict]) -> dict:
    dump_json(ctx.rd / "anonymized.json", items)
    anon_view = [{k: it.get(k) for k in
                  ("pid", "title", "kind", "description", "why_strong",
                   "data_shape", "yield_cost", "determinism")}
                 for it in items]
    revs = {}
    for L in LETTERS:
        p = ctx.rd / "round2" / f"{L}.json"
        cached = load_json(p)
        if cached and cached.get("ratings"):
            revs[L] = cached
            log(ctx.rd, f"round2 {L}: cached")
            continue
        log(ctx.rd, f"round2 {L} ({ctx.letters[L]}): reviewing "
                    f"{len(items)} proposals...")
        obj = call_json(ctx, L, "review",
                        prompt_review(brief, anon_view), "ratings")
        if obj is None:
            log(ctx.rd, f"round2 {L}: FAILED")
            continue
        ratings = [r for r in obj.get("ratings", [])
                   if isinstance(r, dict) and r.get("pid")]
        rec = {"rater_backend": ctx.letters[L], "ratings": ratings}
        p.parent.mkdir(exist_ok=True)
        dump_json(p, rec)
        revs[L] = rec
        log(ctx.rd, f"round2 {L}: {len(ratings)} ratings")
    return revs


def aggregate_ratings(items: list[dict], revs: dict) -> list[dict]:
    agg: dict[str, dict] = {}
    for L, rv in revs.items():
        for r in (rv or {}).get("ratings", []):
            pid = r.get("pid")
            if not pid:
                continue
            a = agg.setdefault(pid, {k: [] for k in REVIEW_AXES})
            a.setdefault("rationales", {})
            for k in REVIEW_AXES:
                v = r.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg[pid][k].append(max(0.0, min(5.0, float(v))))
            if r.get("rationale"):
                agg[pid]["rationales"][L] = r["rationale"]
    rows = []
    pids = {it["pid"] for it in items}
    for it in items:
        a = agg.get(it["pid"], {})
        means = {k: (round(sum(a[k]) / len(a[k]), 2) if a.get(k) else None)
                 for k in REVIEW_AXES}
        core = [means["signal"], means["novelty"],
                means["buildability"], means["risk"]]
        present = [v for v in core if v is not None]
        composite = round(sum(present) / len(present), 2) \
            if len(present) >= 3 else None
        ready = a.get("ready_to_build") or []
        unanimous_ready = (len(ready) >= 2 and all(v == 5.0 for v in ready))
        rows.append(dict(pid=it["pid"], title=it["title"],
                         kind=it.get("kind"), author=it["author"],
                         **{k: means[k] for k in REVIEW_AXES},
                         n_raters=len(ready), unanimous_ready5=unanimous_ready,
                         composite=composite,
                         rationales=a.get("rationales", {})))
    rows.sort(key=lambda r: -(r["composite"] or 0))
    return [r for r in rows if r["pid"] in pids] or rows


def band_of(row: dict) -> str:
    c = row.get("composite")
    if row.get("unanimous_ready5") or (isinstance(c, (int, float))
                                       and c >= BAND_BUILD):
        return "BUILD"
    if isinstance(c, (int, float)) and c >= BAND_BANK:
        return "BANK"
    return "RECYCLE"


def pick_deepen(rows: list[dict]) -> list[dict]:
    """BUILD band first (capped), filled to DEEPEN_TARGET from top BANK,
    kind-diverse, near-duplicate titles filtered (v1 policy)."""
    build = [r for r in rows if band_of(r) == "BUILD"][:PROPOSALS_MAX]
    picked = list(build)
    for r in rows:
        if len(picked) >= DEEPEN_TARGET:
            break
        if r in picked or band_of(r) != "BANK":
            continue
        if any(r["title"][:25].lower() in p["title"].lower() or
               p["title"][:25].lower() in r["title"].lower()
               for p in picked):
            continue
        picked.append(r)
    if not picked:                      # nothing at/above BANK: deepen top-1
        picked = rows[:1]
    return picked


def stage_deepen(ctx: RoundCtx, brief: str, winners: list[dict]) -> dict:
    full = {it["pid"]: it for it in
            load_json(ctx.rd / "anonymized.json", [])}
    expands: dict[str, list] = {}
    for i, L in enumerate(LETTERS):
        p = ctx.rd / "round3" / f"{L}.json"
        if p.exists():
            expands[L] = load_json(p)
            log(ctx.rd, f"round3 {L}: cached")
            continue
        # each model expands min(2, len) winners; with len==3 every winner
        # gets two independent blind expansions by different models
        mine = [winners[(i + j) % len(winners)]
                for j in range(min(2, len(winners)))]
        got = []
        for w in mine:
            idea = full.get(w["pid"])
            if not idea:
                continue
            obj = call_json(ctx, L, "deepen", prompt_deepen(brief, idea),
                            "spec")
            if obj and isinstance(obj.get("spec"), (dict, list, str)):
                obj["expands_pid"] = w["pid"]
                obj["expands_title"] = w["title"]
                got.append(obj)
        p.parent.mkdir(exist_ok=True)
        dump_json(p, got)
        expands[L] = got
        log(ctx.rd, f"round3 {L} ({ctx.letters[L]}): {len(got)} expansions")
    return expands


def stage_rerate(ctx: RoundCtx, brief: str, expands: dict) -> list[dict]:
    flat = []
    for L in LETTERS:
        for e in expands.get(L) or []:
            flat.append(dict(e, author=L))
    if not flat:
        return []
    rng = random.Random(1000 * ctx.n + 7)
    rng.shuffle(flat)
    for i, e in enumerate(flat, 1):
        e["xid"] = f"X-{i:03d}"
    dump_json(ctx.rd / "round3" / "anonymized.json", flat)
    rev3 = {}
    for L in LETTERS:
        p = ctx.rd / "round3" / f"{L}_reviews.json"
        cached = load_json(p)
        if cached and cached.get("ratings"):
            rev3[L] = cached
            continue
        log(ctx.rd, f"round3 re-rate {L} ({ctx.letters[L]})...")
        obj = call_json(ctx, L, "rerate", prompt_rerate(brief, flat),
                        "ratings")
        if obj is None:
            continue
        ratings = [r for r in obj.get("ratings", [])
                   if isinstance(r, dict) and r.get("xid")]
        rec = {"rater_backend": ctx.letters[L], "ratings": ratings}
        dump_json(p, rec)
        rev3[L] = rec
    agg: dict[str, dict] = {}
    for L, rv in rev3.items():
        for r in (rv or {}).get("ratings", []):
            a = agg.setdefault(r["xid"], {k: [] for k in REVIEW_AXES})
            a["rationales"] = a.get("rationales", {})
            for k in REVIEW_AXES:
                v = r.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    a[k].append(max(0.0, min(5.0, float(v))))
            if r.get("rationale"):
                a["rationales"][L] = r["rationale"]
    rows3 = []
    for e in flat:
        a = agg.get(e["xid"], {})
        means = {k: (round(sum(a[k]) / len(a[k]), 2) if a.get(k) else None)
                 for k in REVIEW_AXES}
        ready = a.get("ready_to_build") or []
        vals = [means[k] for k in REVIEW_AXES]
        present = [v for v in vals if v is not None]
        row = dict(xid=e["xid"], title=e.get("expands_title") or e.get("title"),
                   expands_pid=e.get("expands_pid"), author=e["author"],
                   **{k: means[k] for k in REVIEW_AXES},
                   unanimous_ready5=(len(ready) >= 2 and
                                     all(v == 5.0 for v in ready)),
                   composite3=(round(sum(present) / len(present), 2)
                               if len(present) >= 3 else None),
                   rationales=a.get("rationales", {}))
        rows3.append(row)
    rows3.sort(key=lambda r: -(r["composite3"] or 0))
    dump_json(ctx.rd / "aggregate3.json", rows3)
    log(ctx.rd, "aggregate3 written")
    return rows3


# ---------------------------------------------------------------------------
# selection policy: BUILD / BANK / RECYCLE (nothing discarded)
# ---------------------------------------------------------------------------

def banked_titles() -> set[str]:
    titles = set()
    for p in (BANK, BUILD_JSONL):
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    titles.add(norm_title(json.loads(line).get("title", "")))
                except ValueError:
                    pass
    return titles


def write_build(ctx: RoundCtx, entry: dict) -> None:
    append_jsonl(BUILD_JSONL, entry)
    if not BUILD_MD.exists():
        BUILD_MD.write_text(
            "# Ideation tournament — BUILD band / wave-2 spec doc\n"
            "(composite >= 4.4 or unanimous ready_to_build 5; appended\n"
            "by ideation_tournament.py, never edited by hand)\n")
    with open(BUILD_MD, "a") as f:
        f.write(f"\n## Round {entry['round']}: {entry['title']} "
                f"(composite {entry.get('composite_final')}, "
                f"band BUILD)\n\nkind: {entry.get('kind')} | proposed by "
                f"{entry.get('author_letter')} | "
                f"deepened: {entry.get('origin') == 'deepened'}\n\n")
        spec = entry.get("spec") or entry.get("proposal") or {}
        f.write("```json\n" + json.dumps(spec, indent=1)[:20000] + "\n```\n")
    log(ctx.rd, f"BUILD -> wave-2 spec doc: {entry['title'][:70]}")


def write_bank(ctx: RoundCtx, entry: dict, seen: set) -> bool:
    nt = norm_title(entry["title"])
    if nt in seen:
        log(ctx.rd, f"BANK dup skipped: {entry['title'][:60]}")
        return False
    seen.add(nt)
    append_jsonl(BANK, entry)
    log(ctx.rd, f"BANK <- {entry['title'][:70]} "
                f"(composite {entry.get('composite')})")
    return True


def why_not_selected(row: dict) -> str:
    rats = row.get("rationales") or {}
    if rats:
        return min(rats.values(), key=len)[:110]
    axes = {k: row.get(k) for k in ("signal", "novelty", "buildability",
                                    "risk") if row.get(k) is not None}
    if axes:
        worst = min(axes, key=axes.get)
        return f"weakest axis {worst}={axes[worst]}"
    return "unrated"


def seed_v1_bank_and_build() -> None:
    """One-time: fold v1's results into the new band artifacts so the
    bank is real from day one. v1's top-3 re-rated specs -> BUILD doc;
    deepened runners-up with composite3 >= 3.5 and all proposals in
    [3.5, 4.4) -> spec_bank. Guarded by existence; append-only after."""
    if BUILD_JSONL.exists() and BANK.exists():
        return
    full = {it["pid"]: it for it in load_json(V1_ANONYMIZED, [])}
    specs = {e.get("expands_pid"): e for e in load_json(V1_ROUND3, [])}
    comp3_rows = sorted(
        (load_json(V1_AGGREGATE3, []) or []),
        key=lambda r: -(r.get("composite3", r.get("composite")) or 0))
    comp3 = {r.get("expands_pid"): r for r in comp3_rows}
    won = {r.get("expands_pid") for r in comp3_rows[:3]}   # X-002/004/006

    def _c3(r):
        return r.get("composite3", r.get("composite"))

    deep_banked = {r.get("expands_pid") for r in comp3_rows[3:]
                   if isinstance(_c3(r), (int, float))
                   and _c3(r) >= BAND_BANK}
    if not BUILD_JSONL.exists():
        if not BUILD_MD.exists():
            BUILD_MD.write_text(
                "# Ideation tournament — BUILD band / wave-2 spec doc\n"
                "(composite >= 4.4 or unanimous ready_to_build 5; appended\n"
                "by ideation_tournament.py, never edited by hand)\n")
        for pid in won:
            spec = specs.get(pid, {})
            c3 = comp3.get(pid, {})
            prop = full.get(pid, {})
            append_jsonl(BUILD_JSONL, {
                "round": 1, "pid": pid, "title": prop.get("title"),
                "kind": prop.get("kind"), "author_letter":
                    prop.get("author"),
                "origin": "deepened",
                "composite_final": _c3(c3),
                "scores": {k: c3.get(k) for k in REVIEW_AXES},
                "proposal": prop,
                "spec": spec.get("spec"),
                "note": "seeded from v1 winners (ideation-tournament-v1)"})
            with open(BUILD_MD, "a") as f:
                f.write(f"\n## Round 1 (v1): {prop.get('title')} "
                        f"(composite {_c3(c3)}, band BUILD)\n\n"
                        f"```json\n"
                        f"{json.dumps(spec.get('spec'), indent=1)[:20000]}\n"
                        f"\n```\n")
        print(f"seeded build_specs from v1 winners ({len(won)} specs)")
    if not BANK.exists():
        for pid in deep_banked:      # runners-up, but WITH deepened specs
            spec = specs.get(pid, {})
            c3 = comp3.get(pid, {})
            prop = full.get(pid, {})
            append_jsonl(BANK, {
                "round": 1, "pid": pid, "title": prop.get("title"),
                "kind": prop.get("kind"),
                "author_letter": prop.get("author"),
                "origin": "deepened", "composite": _c3(c3),
                "scores": {k: c3.get(k) for k in REVIEW_AXES},
                "proposal": prop, "spec": spec.get("spec"),
                "note": "seeded from v1 round-3 runners-up "
                        "(composite3 >= 3.5)"})
        for r in load_json(V1_AGGREGATE) or []:
            c = r.get("composite")
            if r.get("pid") in won or r.get("pid") in deep_banked or \
                    not isinstance(c, (int, float)) or \
                    not (BAND_BANK <= c < BAND_BUILD):
                continue
            append_jsonl(BANK, {
                "round": 1, "pid": r["pid"], "title": r["title"],
                "kind": r.get("kind"), "author_letter": r.get("author"),
                "origin": "proposal", "composite": c,
                "scores": {k: r.get(k) for k in
                           ("signal", "novelty", "buildability", "risk")},
                "proposal": full.get(r["pid"]),
                "note": "seeded from v1 aggregate (3.5<=composite<4.4)"})
        print("seeded spec_bank from v1 aggregate")


def finalize(ctx: RoundCtx, items, rows, rows3, expands) -> dict:
    full = {it["pid"]: it for it in items}
    flat3 = load_json(ctx.rd / "round3" / "anonymized.json", []) or []
    flat_by_xid = {e.get("xid"): e for e in flat3}
    # each deepened idea has up to two expansions; its final band follows
    # its BEST re-rated spec (alternates stay preserved in round3/)
    r3_by_pid: dict[str, dict] = {}
    for r in rows3:
        pid = r.get("expands_pid")
        if not pid:
            continue
        cur = r3_by_pid.get(pid)
        if cur is None or (r.get("composite3") or 0) > \
                (cur.get("composite3") or 0):
            r3_by_pid[pid] = r
    spec_by_pid = {
        pid: (flat_by_xid.get(r.get("xid"), {}) or {}).get("spec")
        for pid, r in r3_by_pid.items()}
    seen = banked_titles()
    bands = {"BUILD": [], "BANK": [], "RECYCLE": []}
    for row in rows:
        pid = row["pid"]
        deepened = pid in r3_by_pid
        if deepened:
            r3 = r3_by_pid[pid]
            final_c = r3.get("composite3")
            final_band = band_of({**row, "composite": final_c,
                                  "unanimous_ready5":
                                      r3.get("unanimous_ready5")})
        else:
            final_c = row.get("composite")
            final_band = band_of(row)
        base = dict(round=ctx.n, pid=pid, title=row["title"],
                    kind=row.get("kind"), author_letter=row.get("author"),
                    origin="deepened" if deepened else "proposal",
                    composite=final_c,
                    scores={k: (r3_by_pid[pid].get(k) if deepened
                                else row.get(k))
                            for k in REVIEW_AXES},
                    proposal=full.get(pid))
        if final_band == "BUILD":
            entry = dict(base, composite_final=final_c,
                         spec=spec_by_pid.get(pid))
            write_build(ctx, entry)
            bands["BUILD"].append({k: base[k] for k in
                                   ("pid", "title", "kind", "origin")}
                                  | {"composite_final": final_c})
        elif final_band == "BANK":
            if deepened:
                base["spec"] = spec_by_pid.get(pid)
            write_bank(ctx, base, seen)
            bands["BANK"].append({k: base[k] for k in
                                  ("pid", "title", "kind", "origin",
                                   "composite")})
        else:
            bands["RECYCLE"].append({"pid": pid, "title": row["title"],
                                     "why": why_not_selected(row),
                                     "composite": row.get("composite")})
    dump_json(ctx.rd / "bands.json", bands)
    return bands


# ---------------------------------------------------------------------------
# stage 7 triage — ox-alpha digest over the round's outputs
# (humans read digests, not spec firehoses)
# ---------------------------------------------------------------------------

def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def load_triage_status() -> dict:
    return load_json(TRIAGE_STATUS, {}) or {"titles": {}, "rounds": {}}


def save_triage_status(st: dict) -> None:
    dump_json(TRIAGE_STATUS, st)


def _set_title_status(st: dict, title: str, status: str, note: str = "",
                      round_n: int | None = None) -> None:
    st.setdefault("titles", {})[norm_title(title)] = {
        "title": title, "status": status, "note": note[:200],
        "round": round_n}


def _triage_json(rd: Path, prompt: str) -> tuple[dict | None, str]:
    """Ox call for triage: primary GO ox-alpha-free, fallback openrouter
    stealth/ox-alpha (~50% serve rate — push through). Returns
    (parsed-json-or-None, backend-name)."""
    strict = (prompt + "\n\nIMPORTANT: reply with a single valid JSON "
              "object only, no prose, no markdown fences.")
    for cls in (GoOxBackend, OxStealthBackend):
        b = cls()
        fail_limit = getattr(b, "json_fail_limit", JSON_FAIL_BAIL)
        deadline = time.time() + PATIENT_S
        json_fails, attempt = 0, 0
        while time.time() < deadline:
            try:
                raw = b.complete(prompt if attempt == 0 else strict)
                append_jsonl(rd / "raw_calls.jsonl", {
                    "ts": time.strftime("%F %T"), "stage": "triage",
                    "backend": b.name, "model": b.model,
                    "ok": bool(raw and raw.strip()),
                    "chars": len(raw or ""), "raw": (raw or "")[:40000]})
                obj = extract_json(raw) if raw else None
                if isinstance(obj, dict):
                    return obj, b.name
                json_fails += 1
                if json_fails >= fail_limit:
                    break
                time.sleep(5)
            except BackendError as e:
                wait = min(600.0, 60.0 * (attempt + 1)) \
                    if e.kind == "rate" else 5.0
                log(rd, f"triage {b.name}: {e.kind}: {str(e)[:100]}; "
                        f"sleep {wait:.0f}s")
                time.sleep(wait)
            except Exception as e:
                log(rd, f"triage {b.name}: {type(e).__name__}: {e}")
                time.sleep(15)
            attempt += 1
        log(rd, f"triage: {b.name} exhausted; trying fallback")
    return None, ""


def triage_prompt(n: int, round_build: list[dict], round_bank: list[dict],
                  st: dict) -> str:
    bank_entries = _load_jsonl(BANK)[-40:]
    bank_view = "\n".join(
        f"- {e.get('title')} (r{e.get('round')}, composite "
        f"{e.get('composite')}, {e.get('origin', '?')})"[:130]
        for e in bank_entries) or "(empty)"
    prior_build = [e for e in _load_jsonl(BUILD_JSONL)
                   if e.get("type", "build") == "build"]
    prior_view = "\n".join(f"- {e.get('title')} (r{e.get('round')})"
                           [:130] for e in prior_build) or "(none yet)"
    digest_view = ""
    for r in sorted((st.get("rounds") or {}), key=int)[-2:]:
        d = st["rounds"][r]
        digest_view += f"round {r}: " + "; ".join(
            f"{k}: {', '.join(str(x) for x in v[:4])}"
            for k, v in d.items() if v) + "\n"
    digest_view = digest_view or "(no prior digests)"
    new_specs = "\n\n".join(
        f"### {e.get('title')} (composite {e.get('composite_final')})\n"
        + json.dumps(e.get("spec") or e.get("proposal") or
                     {k: e.get(k) for k in ("kind", "scores")},
                     indent=1)[:3500] for e in round_build) or "(none)"
    bank_adds = "\n".join(f"- {b.get('title')} "
                          f"(composite {b.get('composite')})"
                          for b in round_bank) or "(none)"
    return build_brief(n) + f"""

TASK (triage, round {n}): You are the tournament's triage editor. Below
are (1) this round's NEW build-band specs, (2) this round's bank
additions, (3) the full spec bank (append-only idea bank), (4) all
build-band specs from prior rounds, (5) prior triage digests. Decide:

(a) DEDUPE: is any NEW build spec a near-duplicate BY IDEA (not wording)
    of a banked idea, a prior build spec, or an existing family from the
    inventory above?
(b) MERGE: are two entries (new-new or new-banked) partial overlaps that
    should become ONE spec with both angles noted?
(c) RANK the new build specs 1-10 by build_readiness x novelty x
    distinctness-from-existing-families.
(d) PROMOTE: did this round strengthen an earlier BANKED idea enough to
    build it now? (e.g. the banked wrong-but-passing probe set, if a new
    proposal supplies the missing validator machinery)
(e) DEFER: which of this round's bank additions should explicitly wait,
    and for what?

Reply with a single JSON object, no prose:
{{"dedupes": [{{"title": "<new spec title>", "duplicate_of": "<banked/build/family name>", "kind": "bank|build|family", "basis": "one line"}}],
 "merges": [{{"title": "<merged title>", "components": ["<title1>", "<title2>"], "angle_notes": "one line"}}],
 "ranking": [{{"title": "...", "build_readiness": 0-5, "novelty": 0-5, "distinctness": 0-5, "priority": 1-10, "one_line": "..."}}],
 "promotions": [{{"banked_title": "<banked title>", "strengthened_by": "<this round title>", "reason": "one line"}}],
 "deferred": [{{"title": "<bank addition title>", "reason": "one line"}}]}}

NEW BUILD SPECS THIS ROUND:
{new_specs}

BANK ADDITIONS THIS ROUND:
{bank_adds}

SPEC BANK (latest 40):
{bank_view}

PRIOR BUILD SPECS:
{prior_view}

PRIOR TRIAGE DIGESTS:
{digest_view}"""


def _demote_to_bank(rd: Path, entry: dict, dup_of: str, basis: str,
                    st: dict, round_n: int) -> None:
    append_jsonl(BUILD_JSONL, {
        "type": "demotion", "round": round_n, "title": entry["title"],
        "duplicate_of": dup_of, "basis": basis})
    append_jsonl(BANK, dict(entry, origin="demoted-from-build",
                            composite=entry.get("composite_final")
                            or entry.get("composite"),
                            note=f"triage dedupe: duplicate of {dup_of} "
                                 f"({basis})"))
    with open(BUILD_MD, "a") as f:
        f.write(f"- TRIAGE r{round_n} DEMOTE: {entry['title']} — "
                f"duplicate of {dup_of} ({basis})\n")
    _set_title_status(st, entry["title"], "demoted",
                      f"duplicate of {dup_of}", round_n)
    log(rd, f"triage demote: {entry['title'][:50]} = {dup_of[:50]}")


def _promote_from_bank(rd: Path, banked: dict, strengthened_by: str,
                       reason: str, st: dict, round_n: int) -> None:
    append_jsonl(BUILD_JSONL, {
        "type": "promotion", "round": round_n,
        "title": banked.get("title"), "strengthened_by": strengthened_by,
        "reason": reason, "spec": banked.get("spec"),
        "proposal": banked.get("proposal"),
        "scores": banked.get("scores"),
        "banked_composite": banked.get("composite")})
    with open(BUILD_MD, "a") as f:
        f.write(f"\n### TRIAGE r{round_n} PROMOTION: {banked.get('title')}\n"
                f"strengthened by {strengthened_by} — {reason}\n\n")
    _set_title_status(st, banked.get("title", "?"), "promoted",
                      f"strengthened by {strengthened_by}", round_n)
    log(rd, f"triage promote: {banked.get('title', '?')[:50]}")


def triage_round(n: int) -> bool:
    """Stage 7: ox-powered triage of a COMPLETED round. Idempotent
    (TRIAGE_DONE marker); returns False if ox could not serve (the
    supervisor retries on a later cycle)."""
    rd = ROUNDS / f"r{n:03d}"
    if not (rd / "DONE").exists():
        log(rd, f"triage r{n}: round not DONE; skipping")
        return False
    if (rd / "TRIAGE_DONE").exists():
        log(rd, f"triage r{n}: cached")
        return True
    log(rd, f"=== triage round {n} start ===")
    (rd / "triage").mkdir(exist_ok=True)
    st = load_triage_status()
    round_build = [e for e in _load_jsonl(BUILD_JSONL)
                   if e.get("round") == n and
                   e.get("type", "build") == "build"]
    summary = load_json(rd / "round_summary.json", {}) or {}
    bank_adds = [dict(title=b["title"], composite=b.get("composite"))
                 for b in summary.get("bands", {}).get("BANK", [])]
    if not round_build and not bank_adds:
        (rd / "triage_digest.md").write_text(
            f"# Triage digest — round {n}\n\nNothing build-band or banked "
            f"this round; nothing to triage.\n")
        (rd / "TRIAGE_DONE").write_text("empty " + time.strftime("%F %T"))
        _prepend_digest(rd, n)
        return True
    prompt = triage_prompt(n, round_build, bank_adds, st)
    (rd / "triage" / "prompt.txt").write_text(prompt)
    obj, served_by = _triage_json(rd, prompt)
    if obj is None:
        log(rd, f"triage r{n}: FAILED (no ox backend served); "
                f"will retry next cycle")
        return False
    dump_json(rd / "triage" / "triage.json", obj)
    banked_by_title = {norm_title(e.get("title", "")): e
                       for e in _load_jsonl(BANK)
                       if e.get("origin") != "demoted-from-build"}
    titles_st = st.setdefault("titles", {})
    # (a) dedupes -> demote from build back into the bank (append-only)
    for d in obj.get("dedupes") or []:
        t = norm_title(d.get("title", ""))
        entry = next((e for e in round_build
                      if norm_title(e.get("title", "")) == t), None)
        if entry is None:
            continue
        if (titles_st.get(t, {}).get("status") == "promoted"):
            continue
        _demote_to_bank(rd, entry, d.get("duplicate_of", "?"),
                        d.get("basis", ""), st, n)
    # (b) merges -> one entry with both angles. Components may span
    # rounds (build entries of THIS round and/or banked ideas); the
    # merged entry lands in build only when every component was
    # build-band this round, otherwise it ripens in the bank.
    for m in obj.get("merges") or []:
        comps = [c for c in m.get("components") or [] if c]
        if len(comps) < 2:
            continue
        comp_keys = {norm_title(c) for c in comps}
        matched_build = [e for e in round_build
                         if norm_title(e.get("title", "")) in comp_keys]
        matched_bank = [e for e in _load_jsonl(BANK)
                        if norm_title(e.get("title", "")) in comp_keys and
                        e.get("origin") not in ("demoted-from-build",
                                                "merged")]
        matched = matched_build + matched_bank
        if len(matched) < 2:
            continue
        all_build = len(matched_build) == len(matched)
        best = max(
            matched,
            key=lambda e: (e.get("composite_final") or
                           e.get("composite") or 0))
        merged = {
            "type": "merge", "round": n,
            "title": m.get("title") or best.get("title"),
            "components": comps, "angle_notes": m.get("angle_notes", ""),
            "spec": best.get("spec"), "proposal": best.get("proposal"),
            "scores": best.get("scores")}
        if all_build:
            merged["composite_final"] = best.get("composite_final")
            append_jsonl(BUILD_JSONL, merged)
            with open(BUILD_MD, "a") as f:
                f.write(f"\n### TRIAGE r{n} MERGE: "
                        f"{merged['title']}\ncomponents: "
                        f"{', '.join(comps)} — "
                        f"{m.get('angle_notes', '')}\n\n")
            _set_title_status(st, merged["title"], "build",
                              f"merged {', '.join(comps)}", n)
            for e in matched:
                _demote_to_bank(rd, e, f"merge -> {merged['title']}",
                                "partial overlap, angles folded into "
                                "merge", st, n)
        else:
            append_jsonl(BANK, dict(
                merged, round=n, origin="merged",
                composite=best.get("composite"),
                note=f"triage merge of {', '.join(comps)}: "
                     f"{m.get('angle_notes', '')}"))
            _set_title_status(st, merged["title"], "banked",
                              f"merged {', '.join(comps)}", n)
            log(rd, f"triage bank-merge: {merged['title'][:50]}")
    # (d) promotions bank -> build
    for p in obj.get("promotions") or []:
        banked = banked_by_title.get(norm_title(p.get("banked_title", "")))
        if banked is None:
            continue
        _promote_from_bank(rd, banked, p.get("strengthened_by", "?"),
                           p.get("reason", ""), st, n)
    # digest
    ranking = sorted(obj.get("ranking") or [],
                     key=lambda r: -(r.get("priority") or 0))
    lines = [f"# Triage digest — round {n} (served by {served_by}, "
             f"{time.strftime('%F %T')})", ""]
    lines.append("## New build-ready (ranked)")
    if ranking:
        for i, r in enumerate(ranking, 1):
            lines.append(
                f"{i}. **{r.get('title')}** — priority "
                f"{r.get('priority')}/10 (build {r.get('build_readiness')}"
                f", novelty {r.get('novelty')}, distinctness "
                f"{r.get('distinctness')}) — {r.get('one_line')}")
    else:
        lines.append("(none this round)")
    lines.append("\n## Promoted from bank")
    promo_lines = [f"- **{p.get('banked_title')}** — strengthened by "
                   f"{p.get('strengthened_by')} — {p.get('reason')}"
                   for p in obj.get("promotions") or []]
    lines.extend(promo_lines or ["(none)"])
    lines.append("\n## Deferred (with reason)")
    d_lines = [f"- {d.get('title')} — {d.get('reason')}"
               for d in obj.get("deferred") or []]
    lines.extend(d_lines or ["(none)"])
    lines.append("\n## Deduped (against what)")
    dd = [f"- {d.get('title')} = {d.get('duplicate_of')} "
          f"[{d.get('kind')}] ({d.get('basis')})"
          for d in obj.get("dedupes") or []]
    lines.extend(dd or ["(none)"])
    mrg = [f"- {m.get('title')} <- {', '.join(m.get('components') or [])}"
           f" ({m.get('angle_notes')})" for m in obj.get("merges") or []]
    if mrg:
        lines.append("\n## Merged")
        lines.extend(mrg)
    digest = "\n".join(lines) + "\n"
    (rd / "triage_digest.md").write_text(digest)
    st.setdefault("rounds", {})[str(n)] = {
        "new_build_ready": [r.get("title") for r in ranking[:4]],
        "promoted": [p.get("banked_title")
                     for p in obj.get("promotions") or []],
        "deferred": [d.get("title") for d in obj.get("deferred") or []],
        "deduped": [d.get("title") for d in obj.get("dedupes") or []]}
    save_triage_status(st)
    _prepend_digest(rd, n)
    (rd / "TRIAGE_DONE").write_text("done " + time.strftime("%F %T"))
    log(rd, f"=== triage round {n} COMPLETE (served by {served_by}) ===")
    return True


def _prepend_digest(rd: Path, n: int) -> None:
    src = rd / "triage_digest.md"
    if not src.exists():
        return
    body = src.read_text()
    if TRIAGE_DIGESTS.exists():
        body += "\n---\n\n" + TRIAGE_DIGESTS.read_text()
    TRIAGE_DIGESTS.write_text(body)


# ---------------------------------------------------------------------------
# round driver
# ---------------------------------------------------------------------------

def next_round_number() -> int:
    done = [int(m.group(1)) for p in ROUNDS.glob("r[0-9][0-9][0-9]")
            if (m := re.match(r"r(\d{3})$", p.name)) and
            (p / "DONE").exists()]
    return (max(done) + 1) if done else 2


def run_round(n: int, cast_override: list[str] | None = None) -> bool:
    rd = ROUNDS / f"r{n:03d}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "round1").mkdir(exist_ok=True)
    (rd / "round2").mkdir(exist_ok=True)
    (rd / "round3").mkdir(exist_ok=True)
    if (rd / "DONE").exists():
        log(rd, f"round {n} already DONE")
        return True
    if STOP_FILE.exists():
        log(rd, f"STOP file present; round {n} not started")
        return False
    cast_names = cast_override or cast_for_round(n)
    log(rd, f"=== round {n} start; cast rotation: {cast_names} ===")
    ctx = RoundCtx(n, rd, cast_names)
    attach_domain_seeds(ctx)          # ambient diversity seeds (user idea)
    brief = build_brief(n)

    while ctx.left() > 300:
        r1 = stage_propose(ctx)
        if all((r1.get(L) or {}).get("proposals") for L in LETTERS):
            break
        log(rd, "propose incomplete; retrying in 120s")
        time.sleep(120)
    else:
        log(rd, f"round {n}: propose deadline hit; will resume next cycle")
        return False
    # ^ note: while/else fires only if the loop exited via condition

    items = anonymize(ctx, r1)
    log(rd, f"anonymized {len(items)} proposals")
    revs = {}
    while ctx.left() > 300:
        revs = stage_review(ctx, brief, items)
        if sum(1 for L in LETTERS if (revs.get(L) or {}).get("ratings")) >= 2:
            break
        log(rd, "review incomplete (<2 raters); retrying in 120s")
        time.sleep(120)
    else:
        log(rd, f"round {n}: review deadline hit; will resume next cycle")
        return False

    rows = aggregate_ratings(items, revs)
    dump_json(rd / "aggregate.json", rows)
    for r in rows[:10]:
        log(rd, f"  {r['pid']} {r.get('composite')} "
                f"{band_of(r):7s} {r['title'][:60]}")

    winners = pick_deepen(rows)
    log(rd, f"deepen set: {[w['pid'] for w in winners]}")
    while ctx.left() > 300:
        expands = stage_deepen(ctx, brief, winners)
        if sum(len(v) for v in expands.values()) >= 1:
            break
        log(rd, "deepen incomplete; retrying in 120s")
        time.sleep(120)
    else:
        log(rd, f"round {n}: deepen deadline hit; will resume next cycle")
        return False

    rows3 = stage_rerate(ctx, brief, expands)
    bands = finalize(ctx, items, rows, rows3, expands)

    summary = {
        "round": n, "started": time.strftime(
            "%F %T", time.localtime(ctx.t0)),
        "finished": time.strftime("%F %T"),
        "cast": dict(ctx.letters), "substitutions": ctx.history,
        "n_proposals": len(items),
        "n_raters": sum(1 for L in LETTERS
                        if (revs.get(L) or {}).get("ratings")),
        "bands": bands,
        "deepened": [{"pid": r.get("expands_pid"),
                      "title": r.get("title"),
                      "composite3": r.get("composite3")}
                     for r in rows3],
        "backend_stats": ctx.stats(),
        "policy": {"BUILD": f"composite >= {BAND_BUILD} OR unanimous "
                            f"ready_to_build 5 -> build_specs (wave-2 doc)",
                   "BANK": f"{BAND_BANK} <= composite < {BAND_BUILD} -> "
                           f"spec_bank.jsonl (append-only)",
                   "RECYCLE": f"composite < {BAND_BANK} -> kept in raw "
                              f"JSONL + next round's brief as "
                              f"'previously proposed and not selected'"}}
    dump_json(rd / "round_summary.json", summary)
    (rd / "DONE").write_text(f"done {time.strftime('%F %T')}\n")
    log(rd, f"=== ROUND {n} COMPLETE: "
            f"BUILD={len(bands['BUILD'])} BANK={len(bands['BANK'])} "
            f"RECYCLE={len(bands['RECYCLE'])} ===")
    return True


def cmd_status() -> None:
    print(f"next round: {next_round_number()}")
    print(f"cast rotation: {cast_for_round(next_round_number())}")
    print(f"bank entries: "
          f"{len(BANK.read_text().splitlines()) if BANK.exists() else 0}")
    print(f"build entries: "
          f"{len(BUILD_JSONL.read_text().splitlines()) if BUILD_JSONL.exists() else 0}")
    for p in sorted(ROUNDS.glob("r[0-9][0-9][0-9]")):
        done = (p / "DONE").exists()
        tri = (p / "TRIAGE_DONE").exists()
        s = load_json(p / "round_summary.json") or {}
        b = s.get("bands") or {}
        print(f"  {p.name}: {'DONE' if done else 'incomplete'}"
              f"{'+triaged' if tri else ''} "
              f"cast={s.get('cast') or '?'} "
              f"BUILD={len(b.get('BUILD', []))} "
              f"BANK={len(b.get('BANK', []))} "
              f"RECYCLE={len(b.get('RECYCLE', []))}")


def triage_pending() -> list[int]:
    out = []
    for p in sorted(ROUNDS.glob("r[0-9][0-9][0-9]")):
        n = int(p.name[1:])
        if (p / "DONE").exists() and not (p / "TRIAGE_DONE").exists():
            out.append(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("round", help="run/resume one tournament round "
                                      "(then triage it on success)")
    pr.add_argument("--n", type=int, default=None)
    pr.add_argument("--cast", type=str, default=None,
                    help="comma-separated backend names (overrides "
                         "rotation; recorded in cast.json)")
    pt = sub.add_parser("triage", help="ox triage of completed but "
                                       "untriaged rounds (catch-up)")
    pt.add_argument("--n", type=int, default=None)
    br = sub.add_parser("brief", help="print the context brief for a round")
    br.add_argument("--n", type=int, default=next_round_number())
    sub.add_parser("status", help="tournament state overview")
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    ROUNDS.mkdir(exist_ok=True)
    if args.cmd == "status":
        cmd_status()
        return
    if args.cmd == "brief":
        print(build_brief(args.n))
        para = seeds_paragraph(args.n)
        if para:
            print("\n" + "-" * 70 +
                  "\nPROPOSE-stage ambient domain seeds for this round:\n" +
                  "-" * 70 + "\n" + para)
        return
    if args.cmd == "triage":
        targets = [args.n] if args.n else triage_pending()
        if not targets:
            print("no rounds pending triage")
            return
        ok = all(triage_round(t) for t in targets)
        sys.exit(0 if ok else 1)
    seed_v1_bank_and_build()
    n = args.n or next_round_number()
    cast = [c.strip() for c in args.cast.split(",")] if args.cast else None
    ok = run_round(n, cast)
    if ok:
        triage_round(n)          # stage 7 immediately after completion
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
