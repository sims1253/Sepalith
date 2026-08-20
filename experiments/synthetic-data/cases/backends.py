"""Backend adapters: one .complete(prompt) -> raw-text interface over every
LLM source the project uses, with per-backend pacing, retry and stats.

  agy         the agy CLI on the free Google AI Pro quota. The CLI is
              stateful (shares language-server session context), so every
              call uses --new-project, and the prompt MUST go through the
              --prompt flag — a positional prompt silently fails.
  zai         the glm-5.3 chat endpoint (judge_validation.py pattern),
              ZAI_API_KEY env, json_object response_format, low effort.
  opencode    opencode.ai zen free tier (comment_to_code.py pattern),
              OPENCODE_API_KEY; long cooldown on quota 429s.
  openrouter  openrouter free models (comment_to_code.py pattern),
              OPENROUTER_API_KEY; patient 6.5s pacing.
  mock        deterministic test backend (no network): env knobs
              CASES_MOCK_FAIL_EVERY / CASES_MOCK_INVALID / CASES_MOCK_CONSTANT
              exercise the gate, dedup and resume paths in tests.

Retry taxonomy mirrors comment_to_code._Retryable: kind "rate" (429/5xx/
provider) backs off (i+1)*5s; "json"/"net" back off 1s. Stats mirror
API_STATS (attempts/ok/err_429/err_provider/err_other/err_timeout/err_json,
mean latency).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

ZAI_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
OPENCODE_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENCODE_MODEL = "deepseek-v4-flash-free"
OPENROUTER_MODEL = "dots-studio/dots-3-note-preview:free"

AGY_CMD = ["agy", "--print", "--new-project", "--model",
           "gemini-3.7-flash-low", "--effort", "low"]


class BackendError(Exception):
    """kind: 'rate' (429/5xx/provider) -> real backoff; 'json'/'net' -> 1s."""

    def __init__(self, msg: str, kind: str = "rate"):
        super().__init__(msg)
        self.kind = kind


def _new_stats() -> dict:
    return dict(attempts=0, ok=0, err_429=0, err_provider=0, err_other=0,
                err_timeout=0, err_json=0, lat_s=0.0)


def _backoff(kind: str, attempt: int) -> float:
    return (attempt + 1) * 5.0 if kind == "rate" else 1.0


def extract_json_object(text: str | None) -> dict | None:
    """Layer 1: model text -> JSON dict (fence stripping + brace slicing,
    the agy_generators.parse_json convention)."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip().removeprefix("```json").removeprefix("```")
    s = s.removesuffix("```").strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


class Backend:
    name = "?"
    model = "?"

    def __init__(self):
        self.stats = _new_stats()
        self._lock = threading.Lock()
        self._next_start = 0.0

    # -- pacing -----------------------------------------------------------
    def _pace(self):
        gap = getattr(self, "pace_gap_s", 0.0)
        with self._lock:
            now = time.time()
            start = max(now, self._next_start)
            self._next_start = start + gap
        if start - now > 0:
            time.sleep(start - now)

    # -- stats ------------------------------------------------------------
    def _bump(self, key: str, dt: float = 0.0):
        with self._lock:
            d = self.stats
            d["attempts"] += 1
            d[key] = d.get(key, 0) + 1
            d["lat_s"] += dt

    def stats_summary(self) -> dict:
        d = dict(self.stats)
        ok = d.get("ok", 0)
        d["mean_latency_s"] = round(d.pop("lat_s") / ok, 2) if ok else None
        return d

    # -- request ----------------------------------------------------------
    def complete(self, prompt: str) -> str:
        retries = getattr(self, "retries", 3)
        last = None
        for a in range(retries):
            try:
                return self._complete_once(prompt)
            except BackendError as e:
                last = e
                if a < retries - 1:
                    time.sleep(_backoff(e.kind, a))
        raise last

    def _complete_once(self, prompt: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# agy CLI
# ---------------------------------------------------------------------------

class AgyBackend(Backend):
    name = "agy"
    model = "gemini-3.7-flash-low"
    pace_gap_s = 0.5
    retries = 2

    def _complete_once(self, prompt: str) -> str:
        self._pace()
        t0 = time.time()
        try:
            # NEVER a positional prompt: the CLI drops it silently.
            r = subprocess.run(AGY_CMD + ["--prompt", prompt],
                               capture_output=True, text=True, timeout=150)
        except subprocess.TimeoutExpired:
            self._bump("err_timeout", time.time() - t0)
            raise BackendError("agy timeout", kind="net")
        except OSError as e:
            self._bump("err_timeout", time.time() - t0)
            raise BackendError(f"agy spawn failed: {e}", kind="net")
        dt = time.time() - t0
        txt = (r.stdout or "").strip()
        if r.returncode != 0 and not txt:
            self._bump("err_other", dt)
            raise BackendError(f"agy rc={r.returncode}: "
                               f"{(r.stderr or '')[:200]}", kind="rate")
        if not txt:
            self._bump("err_json", dt)
            raise BackendError("agy empty output", kind="json")
        self._bump("ok", dt)
        return txt


# ---------------------------------------------------------------------------
# HTTP chat backends (zai / opencode / openrouter)
# ---------------------------------------------------------------------------

class _HttpBackend(Backend):
    url = ""
    env_key = ""
    timeout_s = 75.0
    user_agent = "curl/8.5.0"   # plain urllib UA gets 403 on the free tiers

    def _api_key(self) -> str:
        key = os.environ.get(self.env_key, "")
        if not key:
            raise RuntimeError(f"{self.name}: ${self.env_key} not set")
        return key

    def _payload(self, prompt: str) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _complete_once(self, prompt: str) -> str:
        self._pace()
        t0 = time.time()
        body = json.dumps(self._payload(prompt)).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            })
        try:
            with urllib.request.urlopen(req, timeout=int(self.timeout_s)) as r:
                payload = json.loads(r.read())
                content = payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            dt = time.time() - t0
            detail = ""
            try:
                detail = e.read()[:200].decode("utf-8", "replace")
            except Exception:
                pass
            if e.code == 429:
                self._bump("err_429", dt)
                kind = "rate"
            elif e.code >= 500 or "provider" in detail.lower():
                self._bump("err_provider", dt)
                kind = "rate"
            else:
                self._bump("err_other", dt)
                kind = "rate" if e.code in (402, 408, 409) else "json"
            raise BackendError(f"{self.name} http {e.code}: {detail}", kind)
        except urllib.error.URLError as e:
            self._bump("err_timeout", time.time() - t0)
            raise BackendError(f"{self.name} net: {e}", kind="net")
        except (KeyError, IndexError, ValueError) as e:
            self._bump("err_json", time.time() - t0)
            raise BackendError(f"{self.name} json: {e}", kind="json")
        dt = time.time() - t0
        if not isinstance(content, str) or not content.strip():
            self._bump("err_json", dt)
            raise BackendError(f"{self.name} empty/null content", kind="json")
        self._bump("ok", dt)
        return content


class ZaiBackend(_HttpBackend):
    """glm-5.3 endpoint exactly as judge_validation.py / call_zai use it."""
    name = "zai"
    model = "glm-5.3"
    url = ZAI_URL
    env_key = "ZAI_API_KEY"
    timeout_s = 180.0
    pace_gap_s = 0.3
    retries = 3

    def _payload(self, prompt: str) -> dict:
        return {
            "model": "glm-5.3",
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": 1500,
            "temperature": 0.95,
        }


class OpencodeBackend(_HttpBackend):
    name = "opencode"
    model = OPENCODE_MODEL
    url = OPENCODE_URL
    env_key = "OPENCODE_API_KEY"
    timeout_s = 60.0
    pace_gap_s = 0.0            # quota-based 429s, fast responses
    retries = 2
    cooldown_s = 300.0          # free-tier quota resets slowly

    def __init__(self):
        super().__init__()
        self._cooldown_until = 0.0

    def _complete_once(self, prompt: str) -> str:
        if time.time() < self._cooldown_until:
            raise BackendError("opencode in 429 cooldown", kind="rate")
        try:
            return super()._complete_once(prompt)
        except BackendError as e:
            if "http 429" in str(e):
                self._cooldown_until = time.time() + self.cooldown_s
            raise

    def _payload(self, prompt: str) -> dict:
        return {"model": OPENCODE_MODEL, "max_tokens": 300,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}


class OpenrouterBackend(_HttpBackend):
    name = "openrouter"
    model = OPENROUTER_MODEL
    url = OPENROUTER_URL
    env_key = "OPENROUTER_API_KEY"
    timeout_s = 75.0
    pace_gap_s = 6.5            # shared free-model cap + provider congestion
    retries = 3

    def _payload(self, prompt: str) -> dict:
        return {"model": OPENROUTER_MODEL, "max_tokens": 3000,
                "reasoning": {"effort": "low"},
                "messages": [{"role": "user", "content": prompt}]}


# ---------------------------------------------------------------------------
# mock (tests; no network)
# ---------------------------------------------------------------------------

class MockBackend(Backend):
    """Deterministic pseudo-LLM driven by env knobs (all default off):
      CASES_MOCK_FAIL_EVERY=N   every Nth request returns unparseable text
      CASES_MOCK_INVALID=1      well-formed JSON whose target fails the gate
      CASES_MOCK_CONSTANT=1     always the same target (exercises dedup)
    The target itself is derived from a seeded hash of the prompt, so runs
    are reproducible and resume-safe."""

    name = "mock"
    model = "mock-0"

    def __init__(self, target_key: str = "comment", seed: int = 0):
        super().__init__()
        self.target_key = target_key
        self.fail_every = int(os.environ.get("CASES_MOCK_FAIL_EVERY", "0") or 0)
        self.invalid = bool(os.environ.get("CASES_MOCK_INVALID", ""))
        self.constant = bool(os.environ.get("CASES_MOCK_CONSTANT", ""))
        self._n = 0
        self._seed = seed

    def _mock_target(self, prompt: str) -> str:
        if self.constant:
            return "a mock comment about the block"
        h = int(hashlib.sha1(
            f"{self._seed}\x00{prompt}".encode()).hexdigest()[:8], 16)
        if self.target_key == "completion":
            helpers = ('starts_with("PARAM")', 'where(is.numeric)',
                       'contains("DT")', 'all_of(c("AGE", "SEX"))',
                       'matches("^[A-Z]{2,}[0-9]+$")')
            return helpers[h % len(helpers)]
        words = ("normalises", "filters", "aggregates", "casts", "sorts",
                 "merges", "recodes", "summarises")
        return f"mock {words[h % len(words)]} the block {h % 997}"

    def _complete_once(self, prompt: str) -> str:
        self._pace()
        self._n += 1
        self._bump("ok", 0.0)
        if self.fail_every and self._n % self.fail_every == 0:
            return "not json at all, sorry"
        target = self._mock_target(prompt)
        if self.invalid:
            target = "bad <- target; length(x)"   # passes JSON, fails gate
        return json.dumps({self.target_key: target})


BACKENDS = {"agy": AgyBackend, "zai": ZaiBackend, "opencode": OpencodeBackend,
            "openrouter": OpenrouterBackend, "mock": MockBackend}


def make_backend(name: str, target_key: str = "comment", seed: int = 0) -> Backend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    cls = BACKENDS[name]
    if cls is MockBackend:
        return MockBackend(target_key=target_key, seed=seed)
    return cls()


# ---------------------------------------------------------------------------
# prompt/response normalization shared by the harness
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$")


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()
