#!/usr/bin/env python3
"""Decision-model judges for the typing-RL PoC (the cheap tier of the
reward cascade; user-linked collections, 2026-10-16).

The collections and their roles in this design:

  vllm-sr Decision 3.0 (d3-edge/d3-lite/...): foundation DECISION models —
      zero-shot choice / yes-no / score questions with per-answer
      probabilities, one forward per question, no text generation. This is
      the dense-process tier: called on the growing prefix while the agent
      types, and as a cheap terminal gate before the expensive LLM judge.
  Cloudflare CLEF (clef 27B / clef-flash 9B): decision models, larger —
      same slot, needs GPU-class serving; not run in this PoC.
  convaiinnovations/laya (MoE small LLMs): generative cheap-judge slot.
  LiquidAI liquid-nanos: mixed family (incl. the d1-omni-600M decision
      model that appears on the same decision board as d3) — alternate
      decision backend.

Interface: one `ask(state, questions) -> {qname: {"label": .., "probs": ..}}`
so any of the above can be swapped in behind the same env code.

Backends here:
  d3:<model_id>   in-process transformers AutoModel .system_one() (local,
                  CPU-fine for 0.6B; ~sub-second per question pack on this
                  box; GPU-class speed documented on the model card)
  stub            deterministic offline judge for tests / --glm-off runs.
                  NEVER used silently: rows record backend="stub".

Setup (once, in this repo — .venv/ is gitignored):
  uv venv experiments/training/poc_typing_env/.venv --python 3.12
  ... see README.md "Decision-model runtime"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading


# ---------------------------------------------------------------------------
# Question packs (d3 system_one schema: choice / noul (yes-no) / score)
# ---------------------------------------------------------------------------

PROCESS_QUESTIONS = {
    "focus": {
        "type": "choice",
        "instructions": ("A developer is typing an R function for a given "
                         "task. Is the partial code on track?"),
        "criteria": {
            "on_track": "Consistent progress toward the assigned task",
            "off_track": "Contradicts or ignores the task requirements",
            "degenerate": "Repetition, filler, or tokens instead of progress",
        },
    },
    "syntax_ok": {
        "type": "noul",
        "instructions": "Does this look like syntactically plausible R code?",
    },
    "risky": {
        "type": "noul",
        "instructions": ("Does the code attempt filesystem writes, network "
                         "access, system() calls, or eval/parse of dynamic "
                         "strings?"),
    },
}

TERMINAL_QUESTIONS = {
    "quality": {
        "type": "score",
        "instructions": "Rate this complete R function against its task.",
        "criteria": ["unusable", "poor", "acceptable", "good", "elegant"],
    },
    "looks_correct": {
        "type": "noul",
        "instructions": ("Ignoring style: does this look like a correct "
                         "implementation of the task?"),
    },
    "hacky": {
        "type": "noul",
        "instructions": ("Does it try to game the tests or benchmark instead "
                         "of genuinely solving the task (special-casing "
                         "test values, caching benchmark inputs, timing "
                         "tricks)?"),
    },
}


def _normalize_answer(ans, question: dict | None = None) -> dict:
    """d3 raw answers -> {"label", "probs"}.

    Observed d3 shapes (probed on d3-edge 2026-10-16):
      choice: {"type": "choice", "choice": <label>, "probs": {...}}
      noul:   {"type": "noul", "noul": <P(yes)>}          (yes/no prob)
      score:  {"type": "score", "score": <idx>, "probs": {"0".."n-1": p}}
    score indices are mapped onto the question's criteria order.
    """
    if not isinstance(ans, dict):
        return {"label": ans, "probs": {}}
    # noul: direct P(yes)
    if isinstance(ans.get("noul"), (int, float)):
        p = float(ans["noul"])
        return {"label": "yes" if p >= 0.5 else "no",
                "probs": {"yes": p, "no": 1.0 - p}}
    crit = question.get("criteria") if question else None
    crit_seq = (list(crit) if isinstance(crit, (dict, list)) else None)

    def _map(k) -> str:
        if crit_seq and isinstance(k, (str, int)) and str(k).isdigit() \
                and int(k) < len(crit_seq):
            return str(crit_seq[int(k)])
        return str(k)

    label = None
    for k in ("decision", "answer", "label", "choice", "top", "value",
              "result", "score"):
        if k in ans and not isinstance(ans[k], dict):
            label = ans[k]
            break
    probs = {}
    for k in ("probs", "probabilities", "scores", "p"):
        if isinstance(ans.get(k), dict):
            probs = {_map(a): float(b) for a, b in ans[k].items()}
            break
    if label is not None:
        label = _map(label)
    if isinstance(label, bool):
        label = "yes" if label else "no"
    probs = {("yes" if k in (True, "True", "true") else
              "no" if k in (False, "False", "false") else k): v
             for k, v in probs.items()}
    # score answers carry a numeric expectation (possibly as a string);
    # label by the criteria level it rounds to, keep the expectation
    expectation = None
    if probs:
        try:
            expectation = float(label)
            label = None
        except (TypeError, ValueError):
            pass
    if label is None and probs:
        if expectation is not None and crit_seq:
            label = str(
                crit_seq[max(0, min(len(crit_seq) - 1,
                                    int(round(expectation))))])
        else:
            label = max(probs, key=probs.get)  # type: ignore[arg-type]
    if isinstance(label, str):
        low = {k.lower(): k for k in probs}
        label = low.get(label.lower(), label)
    out = {"label": label, "probs": probs}
    if expectation is not None:
        out["expectation"] = expectation
    return out


class BaseJudge:
    backend = "?"
    calls = 0

    def ask(self, state: str, questions: dict) -> dict:
        self.calls += 1
        out = self._ask(state, questions)
        return {q: _normalize_answer(a, questions.get(q))
                for q, a in out.items()}

    def _ask(self, state: str, questions: dict) -> dict:
        raise NotImplementedError


class D3LocalJudge(BaseJudge):
    """vllm-sr d3 via in-process transformers (AutoModel.system_one)."""

    backend = "d3"

    def __init__(self, model_id: str = "vllm-sr/d3-edge",
                 threads: int = 8):
        self.model_id = model_id
        import torch  # noqa: F401  (fail fast if runtime missing)
        torch.set_num_threads(threads)
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True)

    def _ask(self, state: str, questions: dict) -> dict:
        r = self._model.system_one(state=state, questions=questions)
        return r["answers"]


class StubJudge(BaseJudge):
    """Deterministic offline decision judge (tests / --glm-off runs only).

    sha1(state) -> uniform in [0,1) per question; label = argmax of a
    fixed pseudo-distribution. Marked backend="stub" in every row.
    """

    backend = "stub"

    def _ask(self, state: str, questions: dict) -> dict:
        out = {}
        for name, q in questions.items():
            h = hashlib.sha1(
                f"{name}|{state}".encode()).hexdigest()
            u = int(h[:8], 16) / 2**32
            crit = q.get("criteria")
            if isinstance(crit, dict):
                labels = list(crit)
            elif isinstance(crit, list):
                labels = list(crit)
            elif q.get("type") == "noul":
                labels = [True, False]
            else:
                labels = ["yes", "no"]
            if q.get("type") == "noul":
                # biased toward yes/plausible: stub must be benign-ish
                label = "yes" if u < 0.8 else "no"
                probs = {"yes": 0.5 + 0.3 * (u < 0.8),
                         "no": 0.5 - 0.3 * (u < 0.8)}
            else:
                i = min(int(u * len(labels)), len(labels) - 1)
                label = labels[i]
                w = [0.5 if j == i else 0.5 / max(1, len(labels) - 1)
                     for j in range(len(labels))]
                probs = dict(zip(map(str, labels), w))
            out[name] = {"decision": label, "probs": probs}
        return out


_LOCK = threading.Lock()
_LOADED: BaseJudge | None = None


def make_judge(spec: str | None = None) -> BaseJudge | None:
    """spec: 'd3', 'd3:vllm-sr/d3-lite', 'stub', 'off'/None -> None."""
    spec = (spec if spec is not None
            else os.environ.get("TYPING_ENV_D3", "auto")).lower()
    if spec in ("off", "none", ""):
        return None
    if spec == "auto":
        spec = "d3"
    global _LOADED
    with _LOCK:
        if _LOADED is not None:
            return _LOADED
        if spec == "stub":
            j: BaseJudge = StubJudge()
        elif spec.startswith("d3"):
            model_id = (spec.split(":", 1)[1] if ":" in spec
                        else "vllm-sr/d3-edge")
            j = D3LocalJudge(model_id=model_id)
        else:
            raise ValueError(f"unknown judge spec {spec!r}")
        _LOADED = j
        return j


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=None,
                    help="d3 | d3:<model_id> | stub | off")
    ap.add_argument("--state", default=(
        "An R developer typed: roll <- function(x, w) { s <- cumsum(c(0, x))"
        "; (s[-c(1:w)] - s[1:(length(x)-w+1)]) / w }"))
    args = ap.parse_args()
    j = make_judge(args.spec)
    if j is None:
        print("judge disabled")
        raise SystemExit(0)
    for pack_name, pack in (("process", PROCESS_QUESTIONS),
                            ("terminal", TERMINAL_QUESTIONS)):
        r = j.ask(args.state, pack)
        print(pack_name, json.dumps(r, indent=1, default=str))
