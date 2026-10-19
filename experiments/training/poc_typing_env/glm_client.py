#!/usr/bin/env python3
"""glm-5.3 client for the typing-RL PoC.

Same endpoint / headers / retry taxonomy as the project's proven
`experiments/synthetic-data/cases/backends.py::ZaiBackend`
(api.z.ai coding endpoint, curl UA, 429/5xx backoff, pacing), with two
additions the typing environment needs:

  - `stream=True` chat: the typing agent consumes the completion AS IT IS
    PRODUCED. Each yielded delta is one "keystroke burst" — the env scores
    the growing prefix, so process rewards see genuine generation states.
  - usage accounting (prompt/completion tokens) for cost reporting.

`ZAI_API_KEY` from the environment (repo convention).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

ZAI_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"


class GlmError(Exception):
    def __init__(self, msg: str, kind: str = "rate"):
        super().__init__(msg)
        self.kind = kind


class GlmClient:
    def __init__(self, *, pace_gap_s: float = 0.3, retries: int = 3,
                 timeout_s: int = 300):
        self.pace_gap_s = pace_gap_s
        self.retries = retries
        self.timeout_s = timeout_s
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                      "calls": 0}
        self._lock = threading.Lock()
        self._next_start = 0.0

    # -- api ---------------------------------------------------------------
    def _key(self) -> str:
        key = os.environ.get("ZAI_API_KEY", "")
        if not key:
            raise RuntimeError("ZAI_API_KEY not set")
        return key

    def _pace(self):
        with self._lock:
            now = time.time()
            start = max(now, self._next_start)
            self._next_start = start + self.pace_gap_s
        if start - now > 0:
            time.sleep(start - now)

    def _bump_usage(self, usage: dict | None):
        if usage:
            with self._lock:
                self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += usage.get(
                    "completion_tokens", 0)
                self.usage["calls"] += 1

    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_tokens: int = 2048, stream: bool = False,
             thinking: str = "enabled", reasoning_effort: str = "low",
             response_format: dict | None = None,
             model: str = MODEL):
        """Chat completion. stream=True -> generator of str deltas (usage
        lands on the client when the stream ends). model override enables
        the cheap flash tier (glm-5.3-flash, thinking disabled)."""
        payload: dict = {
            "model": model,
            "thinking": {"type": thinking},
            "reasoning_effort": reasoning_effort,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                if stream:
                    return self._stream_once(payload)
                return self._once(payload)
            except GlmError as e:
                last = e
                if attempt < self.retries - 1:
                    time.sleep((attempt + 1) * 5.0
                               if e.kind == "rate" else 1.0)
        raise last  # type: ignore[misc]

    # -- one request ---------------------------------------------------------
    def _request(self, payload: dict, *, stream: bool):
        body = json.dumps({**payload, "stream": stream}).encode()
        req = urllib.request.Request(
            ZAI_URL, data=body, headers={
                "Authorization": f"Bearer {self._key()}",
                "Content-Type": "application/json",
                "User-Agent": "curl/8.5.0",
            })
        return urllib.request.urlopen(req, timeout=self.timeout_s)

    def _once(self, payload: dict) -> str:
        self._pace()
        with self._request(payload, stream=False) as r:
            d = json.loads(r.read())
        self._bump_usage(d.get("usage"))
        return d["choices"][0]["message"]["content"]

    def _stream_once(self, payload: dict):
        self._pace()
        gen = self._deltas(payload)
        return gen

    def _deltas(self, payload: dict):
        with self._request(payload, stream=True) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    d = json.loads(data)
                except ValueError:
                    continue
                self._bump_usage(d.get("usage"))   # final chunk carries it
                for ch in d.get("choices", []):
                    delta = ch.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece


# shared singleton importers can reuse
_client: GlmClient | None = None


def shared_client() -> GlmClient:
    global _client
    if _client is None:
        _client = GlmClient()
    return _client
