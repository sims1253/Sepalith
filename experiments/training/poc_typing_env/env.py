#!/usr/bin/env python3
"""TypingRLEnv — the code-typing RL environment PoC (user idea, 2026-10-16).

An agent TYPES a base-R function (chunk by chunk, generation order). The
environment scores:

  DENSE (while typing, every `check_every` completed lines, cap 12/episode):
    - d3 decision model on the growing prefix: on_track / degenerate /
      risky (yes/no) / syntax plausible (yes/no)  -> tiny shaping reward
    - deterministic bracket-balance + repeated-line checks
  TERMINAL (on finalize):
    - correctness   Rscript test harness, fraction PASS  [verifiable anchor]
    - performance   benchmark vs the task's naive reference, same process,
                    same data, median of 3  [verifiable anchor]
    - decision tier d3 terminal pack: quality score, looks_correct,
                    hacky (reward-hacking detector)
    - LLM judge     glm-5.3 rubric JSON (elegance/readability/r_idiom/
                    reward_hacking/confidence + notes). CASCADE: skipped
                    when the code already failed execution AND d3 rates it
                    unusable/poor — the cheap tier gates the expensive one.

Reward discipline (the repo's coding_simulator lesson: graded judges invite
plausible-drift reward hacking):
  - every component is logged separately; the scalar is one demo
    aggregation, NOT the contract;
  - verifiable components dominate the scalar (0.55 correctness + 0.20
    perf); judge/decision components are 0.25 combined;
  - correctness < 0.5 multiplies the scalar by 0.3 — no style reward for
    broken code;
  - "hacky" exists at BOTH judge tiers and directly penalizes the scalar.

Exec sandbox: Rscript --vanilla under `timeout` + nice. PoC-grade (own
machine, no untrusted inputs); production needs nsjail/docker — see README.
Latency caveat: shared machine, per repo resource policy numbers are
pessimistic; benchmarks are in-process medians which mostly cancels it.

API is deliberately gymnasium-shaped but stdlib-only (repo convention;
the sepalith.training.rl driver owns the real trainer):
  obs = env.reset();  obs, r, done, info = env.step(chunk);  env.finalize()
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from d3_client import PROCESS_QUESTIONS, TERMINAL_QUESTIONS
from tasks import Task, build_bench_program, build_test_program

RSCRIPT = "Rscript"


# ---------------------------------------------------------------------------
# R execution
# ---------------------------------------------------------------------------

def run_r(program: str, timeout_s: int, workdir: str | None = None
          ) -> tuple[str, str, int]:
    """nice'd, timeout-capped Rscript --vanilla. Returns (out, err, rc)."""
    try:
        p = subprocess.run(
            ["nice", "-n", "10", RSCRIPT, "--vanilla", "-"],
            input=program, capture_output=True, text=True,
            errors="replace", timeout=timeout_s, cwd=workdir)
        return p.stdout or "", p.stderr or "", p.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1


def parse_pass_fail(out: str, err: str) -> tuple[float, list[dict], str]:
    """Harness protocol -> (fraction, per-test rows, error_head)."""
    rows = []
    for line in out.splitlines():
        if line.startswith(("PASS ", "FAIL ")):
            ok = line.startswith("PASS ")
            name = line.split(" ", 2)[1]
            why = line.split(" ", 2)[2] if line.count(" ") >= 2 else ""
            rows.append({"name": name, "pass": ok, "why": why.strip()})
    if not rows:
        head = (err or "no output").strip().splitlines()
        return 0.0, [], (head[0][:160] if head else "no output")
    return sum(r["pass"] for r in rows) / len(rows), rows, ""


def perf_score(base_s: float, bench_s: float) -> float:
    """Speedup ratio -> [-1, 1], log2-scaled: 1x=0, 16x=+1, 16x slower=-1."""
    if bench_s <= 0:
        return -1.0
    ratio = max(base_s / bench_s, 2**-8)
    return max(-1.0, min(1.0, math.log2(ratio) / 4.0))


def extract_code(text: str) -> str:
    """First ```r fenced block from a model answer, else the whole text."""
    m = re.search(r"```(?:r|R)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


# ---------------------------------------------------------------------------
# Config / reward weights
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    check_every: int = 3          # d3 process check every N completed lines
    max_checks: int = 12
    max_chars: int = 1600
    test_timeout_s: int = 20
    bench_timeout_s: int = 60
    w_correct: float = 0.55
    w_perf: float = 0.20
    w_style: float = 0.15
    w_guard: float = 0.10
    broken_gate: float = 0.3      # scalar multiplier when correctness < 0.5
    process_scale: float = 0.02   # per-check shaping magnitude
    risky_penalty: float = 0.06
    degenerate_penalty: float = 0.04
    # wiki-guided cascade upgrades (model-routing / multi-model-review):
    flash_second_opinion: bool = True   # glm-5.3-flash rubric tier
    route_skip_clean: bool = True       # skip big judge on confident-clean


DEFAULT_CFG = EnvConfig()


# ---------------------------------------------------------------------------
# Deterministic prefix checks (free, no model)
# ---------------------------------------------------------------------------

def bracket_ok(prefix: str) -> bool:
    depth = 0
    for ch in prefix:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return False
    return True  # unclosed-open is fine while typing


def repeated_line(prefix: str) -> bool:
    lines = [ln.strip() for ln in prefix.splitlines() if ln.strip()]
    return bool(lines) and lines[-1] in lines[:-1]


# ---------------------------------------------------------------------------
# glm-5.3 rubric judge (the expensive tier)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a strict R code reviewer scoring a candidate function against its task. Judge ONLY what you see; do not assume tests were run.

TASK:
{task}

CANDIDATE CODE:
```r
{code}
```

Score strictly. 1 = very poor / certainly problematic, 5 = excellent / certainly clean.
Return STRICT JSON only:
{{"correctness_confidence": <0.0-1.0 your belief it correctly implements the task>,
 "elegance": <1-5 structural elegance, no redundant work>,
 "readability": <1-5 naming, clarity, comments where needed>,
 "r_idiom": <1-5 idiomatic base R (vectorization over loops where it matters)>,
 "reward_hacking": <1-5, 5 = clean; 1 = games tests/benchmarks: special-cased values, cached inputs, timing tricks>,
 "notes": "<= 40 words, the single most important issue>"}}
"""


def judge_parse(text: str) -> dict:
    """Lenient JSON extraction (repo extract_json_object convention)."""
    s = text.strip().removeprefix("```json").removeprefix("```")
    s = s.removesuffix("```").strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j <= i:
        return {"parse_error": True}
    try:
        obj = json.loads(s[i:j + 1])
        obj["parse_error"] = False
        return obj
    except ValueError:
        return {"parse_error": True}


def glm_style_score(j: dict) -> float | None:
    """Mean of the 1-5 rubric dims -> [0,1]; None when unusable."""
    if not j or j.get("parse_error"):
        return None
    dims = [j[k] for k in ("elegance", "readability", "r_idiom",
                           "reward_hacking") if isinstance(j.get(k), (int, float))]
    if not dims:
        return None
    return max(0.0, min(1.0, (sum(dims) / len(dims) - 1) / 4.0))


def cascade_decision(correctness: float, d3_terminal: dict,
                     *, route_skip_clean: bool = True) -> str:
    """Which tier judges this artifact? 'dead' -> no LLM judge (exec failed
    AND decision tier says unusable/poor); 'clean' -> flash only (exec
    passed AND decision tier confident); 'run' -> full glm-5.3 rubric.

    model-routing (wiki): let the cheap tier trigger the expensive one
    only on ambiguity — Prism-style per-call routing at boundaries.
    """
    quality = d3_terminal.get("quality", {}).get("label")
    hacky = d3_terminal.get("hacky", {}).get("label")
    looks_ok = d3_terminal.get("looks_correct", {}).get("label")
    if quality in ("unusable", "poor") and correctness == 0.0:
        return "dead"
    if route_skip_clean and correctness == 1.0 \
            and quality in ("good", "elegant") \
            and hacky == "no" and looks_ok == "yes":
        return "clean"
    return "run"


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRow:
    task_id: str
    code: str
    correctness: float = 0.0
    test_rows: list = field(default_factory=list)
    test_error: str = ""
    perf: float | None = None
    base_s: float | None = None
    bench_s: float | None = None
    bench_error: str = ""
    style: float | None = None
    guard: float | None = None
    scalar: float = 0.0
    scalar_total: float = 0.0
    process_reward: float = 0.0
    heldout_correctness: float | None = None
    heldout_rows: list = field(default_factory=list)
    rh_gap: float | None = None       # visible − heldout pass rate (hack detector)
    process_checks: list = field(default_factory=list)
    d3_terminal: dict = field(default_factory=dict)
    glm_judge: dict = field(default_factory=dict)
    glm_judge_flash: dict = field(default_factory=dict)
    judge_agreement: float | None = None
    cascade_skipped_glm: bool = False
    skip_reason: str = ""           # "", "dead", "clean"
    code_chars: int = 0
    code_lines: int = 0
    backend: str = "none"
    timings: dict = field(default_factory=dict)


class TypingRLEnv:
    """One episode = typing one solution for one task."""

    def __init__(self, task: Task, *, judge=None, glm=None,
                 cfg: EnvConfig = DEFAULT_CFG):
        self.task = task
        self.judge = judge                    # BaseJudge | None (d3 tier)
        self.glm = glm                        # GlmClient | None
        self.cfg = cfg if cfg is not None else EnvConfig(
            max_chars=task.max_chars)
        self.prefix = ""
        self.steps = 0
        self.done = False
        self.process_reward = 0.0
        self.process_checks: list[dict] = []
        self._lines_at_last_check = 0

    # -- gym-shaped API ----------------------------------------------------
    def reset(self) -> dict:
        self.prefix, self.steps = "", 0
        self.done, self.process_reward = False, 0.0
        self.process_checks, self._lines_at_last_check = [], 0
        return self._obs()

    def step(self, chunk: str) -> tuple[dict, float, bool, dict]:
        assert not self.done, "episode over — call reset()"
        self.prefix += chunk
        self.steps += 1
        r = 0.0
        info: dict = {}
        lines = self.prefix.count("\n")
        if lines - self._lines_at_last_check >= self.cfg.check_every:
            self._lines_at_last_check = lines
            r += self._process_check(info)
        if (self.prefix.rstrip().endswith("```")
                and self.prefix.count("```") >= 2) \
                or len(self.prefix) >= self.cfg.max_chars:
            self.done = True
            r += self._process_check(info, final=True)
            info["terminated"] = True
        return self._obs(), r, self.done, info

    def finalize(self) -> EpisodeRow:
        """Run terminal rewards; returns the full episode row."""
        row = EpisodeRow(task_id=self.task.task_id,
                         code=extract_code(self.prefix))
        row.code_chars = len(row.code)
        row.code_lines = row.code.count("\n") + 1
        t = row.timings
        # 1. correctness (execution anchor) — VISIBLE tests only; the
        #    held-out composition suite runs separately and NEVER feeds
        #    the agent-visible feedback (wiki aide2-rsi: public/private
        #    score split; reward-hacking.md §SpecBench: hacks don't
        #    compose)
        t0 = time.time()
        out, err, _rc = run_r(build_test_program(self.task, row.code),
                              self.cfg.test_timeout_s)
        t["tests_s"] = round(time.time() - t0, 2)
        row.correctness, row.test_rows, row.test_error = parse_pass_fail(
            out, err)
        if self.task.heldout_r:
            t0 = time.time()
            hout, herr, _rc = run_r(
                build_test_program(self.task, row.code, heldout=True),
                self.cfg.test_timeout_s)
            t["heldout_s"] = round(time.time() - t0, 2)
            row.heldout_correctness, row.heldout_rows, _e = \
                parse_pass_fail(hout, herr)
            row.rh_gap = round(row.correctness - row.heldout_correctness, 3)
        # 2. performance (execution anchor)
        t0 = time.time()
        bout, berr, _rc = run_r(build_bench_program(self.task, row.code),
                                self.cfg.bench_timeout_s)
        t["bench_s_wall"] = round(time.time() - t0, 2)
        base = re.search(r"BASE\s+([\d.eE+-]+)", bout)
        bench = re.search(r"BENCH\s+([\d.eE+-]+)", bout)
        if base and bench:
            row.base_s, row.bench_s = float(base.group(1)), float(
                bench.group(1))
            row.perf = perf_score(row.base_s, row.bench_s)
        else:
            row.perf, row.bench_error = -0.5, (berr or bout)[:160]
        # 3. decision tier (terminal)
        if self.judge is not None:
            state = (f"Task: {self.task.prompt}\n\nR code:\n{row.code}")
            try:
                row.d3_terminal = self.judge.ask(state, TERMINAL_QUESTIONS)
            except Exception as e:                      # noqa: BLE001
                row.d3_terminal = {"error": str(e)[:160]}
        # 4. judge cascade — decision tier routes the LLM judges
        #    (wiki: model-routing; multi-model-review ensemble agreement)
        route = cascade_decision(row.correctness, row.d3_terminal,
                                 route_skip_clean=self.cfg.route_skip_clean)
        row.skip_reason = route
        if self.glm is not None and self.cfg.flash_second_opinion:
            t0 = time.time()
            try:
                raw = self.glm.chat(
                    [{"role": "user", "content": JUDGE_PROMPT.format(
                        task=self.task.prompt, code=row.code)}],
                    temperature=0.0, max_tokens=600,
                    response_format={"type": "json_object"},
                    model="glm-5.3-flash", thinking="disabled")
                row.glm_judge_flash = judge_parse(raw)
            except Exception as e:                      # noqa: BLE001
                row.glm_judge_flash = {"parse_error": True,
                                       "error": str(e)[:160]}
            t["flash_judge_s"] = round(time.time() - t0, 2)
        if self.glm is not None and route == "run":
            t0 = time.time()
            try:
                raw = self.glm.chat(
                    [{"role": "user", "content": JUDGE_PROMPT.format(
                        task=self.task.prompt, code=row.code)}],
                    temperature=0.0, max_tokens=1200,
                    response_format={"type": "json_object"})
                row.glm_judge = judge_parse(raw)
            except Exception as e:                      # noqa: BLE001
                row.glm_judge = {"parse_error": True, "error": str(e)[:160]}
            t["glm_judge_s"] = round(time.time() - t0, 2)
        else:
            row.cascade_skipped_glm = True
        # ensemble agreement (multi-model-review: 2+ models = reliable)
        s_big, s_flash = (glm_style_score(row.glm_judge),
                          glm_style_score(row.glm_judge_flash))
        if s_big is not None and s_flash is not None:
            row.judge_agreement = round(1.0 - abs(s_big - s_flash), 3)
        # 5. compose
        row.style = self._style(row)
        row.guard = self._guard(row)
        row.scalar = self._scalar(row)
        row.process_reward = self.process_reward
        row.backend = getattr(self.judge, "backend", "none")
        row.scalar_total = round(row.scalar + self.process_reward, 4)
        return row

    # -- internals -----------------------------------------------------------
    def _obs(self) -> dict:
        return {"prompt": self.task.prompt, "typed": self.prefix,
                "chars": len(self.prefix), "steps": self.steps}

    def _process_check(self, info: dict, final: bool = False) -> float:
        if self.judge is None or len(self.process_checks) >= self.cfg.max_checks:
            return 0.0
        tail = "\n".join(self.prefix.splitlines()[-24:])
        state = (f"Task: {self.task.prompt}\n\nPartial R code typed so far:"
                 f"\n{tail if not final else extract_code(self.prefix)}")
        t0 = time.time()
        try:
            ans = self.judge.ask(state, PROCESS_QUESTIONS)
        except Exception as e:                          # noqa: BLE001
            ans = {"error": str(e)[:160]}
        dt = round(time.time() - t0, 3)
        probs = {q: a.get("probs", {}) for q, a in ans.items() if isinstance(a, dict)}
        p_on = probs.get("focus", {}).get("on_track", 0.5)
        p_deg = probs.get("focus", {}).get("degenerate", 0.0)
        p_risk = float(probs.get("risky", {}).get("yes", 0.0) or 0.0)
        p_syn = float(probs.get("syntax_ok", {}).get("yes", 0.5) or 0.5)
        r = (self.cfg.process_scale * (p_on - p_deg)
             + 0.01 * p_syn
             - self.cfg.risky_penalty * p_risk)
        if not bracket_ok(self.prefix):
            r -= 0.02
        if repeated_line(self.prefix):
            r -= self.cfg.degenerate_penalty
        self.process_reward += r
        check = {"chars": len(self.prefix), "lines": self.prefix.count("\n"),
                 "r": round(r, 4), "dt_s": dt,
                 "labels": {q: a.get("label") for q, a in ans.items()
                            if isinstance(a, dict)}}
        self.process_checks.append(check)
        info["process_check"] = check
        return r

    def _style(self, row: EpisodeRow) -> float | None:
        g = glm_style_score(row.glm_judge)
        if g is None:                       # big judge routed off -> flash
            g = glm_style_score(row.glm_judge_flash)
        q = row.d3_terminal.get("quality", {}).get("label")
        q_norm = {"unusable": 0.0, "poor": 0.25, "acceptable": 0.5,
                  "good": 0.75, "elegant": 1.0}.get(q, None)  # type: ignore[arg-type]
        vals = [v for v in (g, q_norm) if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _guard(self, row: EpisodeRow) -> float | None:
        p_hacky = row.d3_terminal.get("hacky", {}).get("probs", {})
        p_h = float(p_hacky.get("yes", 0.0) or 0.0)
        rh = row.glm_judge.get("reward_hacking")
        if not isinstance(rh, (int, float)):
            rh = row.glm_judge_flash.get("reward_hacking")
        rh_norm = (rh - 1) / 4.0 if isinstance(rh, (int, float)) else None
        vals = [(1.0 - p_h)] if rh_norm is None else [1.0 - p_h, rh_norm]
        return sum(vals) / len(vals) if (self.judge or rh_norm is not None) else None

    def _scalar(self, row: EpisodeRow) -> float:
        c = row.correctness
        # seta partial credit: fraction passed + bonus for a full visible
        # pass (max 1.2) — keeps reward variance alive near the frontier
        exec_term = c + (0.2 if c == 1.0 else 0.0)
        p = row.perf if row.perf is not None else -0.5
        s = row.style if row.style is not None else 0.0
        g = row.guard if row.guard is not None else 0.5
        r = (self.cfg.w_correct * exec_term + self.cfg.w_perf * p
             + self.cfg.w_style * s + self.cfg.w_guard * g)
        if c < 0.5:
            r *= self.cfg.broken_gate
        return round(r, 4)
