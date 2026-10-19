#!/usr/bin/env python3
"""The typing agent: streams a solution chunk-by-chunk via glm-5.3.

`type_solution` yields text chunks in GENERATION ORDER (streamed deltas
re-chunked to line boundaries) — the environment scores the growing prefix,
so process rewards observe genuine partial states of the model's own
composition, not a post-hoc slicing.

A `MockTypist` replays fixed text the same way (tests / --glm-off runs).
"""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM = (
    "You write small, fast, correct base-R functions. Output EXACTLY one "
    "fenced code block (```r ... ```) containing ONLY the function — no "
    "prose before or after, no packages, library() calls, or cat/print "
    "debugging. Respect NA and empty-input semantics exactly as the task "
    "states. Prefer vectorized base-R idioms over loops when they matter "
    "for speed."
)


def build_prompt(task_prompt: str, feedback: str | None = None) -> str:
    p = f"TASK:\n{task_prompt}\n"
    if feedback:
        p += f"\nPREVIOUS ATTEMPT FEEDBACK (improve on it):\n{feedback}\n"
    p += "\nNow output the function."
    return p


@dataclass
class Feedback:
    text: str | None


class GlmTypist:
    def __init__(self, glm, *, temperature: float = 0.8,
                 max_tokens: int = 1600):
        self.glm = glm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def type_solution(self, task_prompt: str, feedback: str | None = None,
                      temperature: float | None = None):
        """Yields chunks; returns full text via StopIteration value.

        Stream failures before the first delta are retried by the client;
        a failure MID-STREAM propagates (the caller finalizes the partial
        prefix as a failed episode — honest RL, no silent regeneration
        over already-typed text)."""
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(task_prompt, feedback)},
        ]
        stream = self.glm.chat(
            messages, temperature=self.temperature if temperature is None
            else temperature, max_tokens=self.max_tokens, stream=True)
        buf, full = "", []
        got_any = False
        for delta in stream:
            got_any = True
            buf += delta
            full.append(delta)
            *lines, buf = buf.split("\n")
            for ln in lines:
                yield ln + "\n"
        if buf:
            full.append("")
            yield buf
        return "".join(full)


class MockTypist:
    """Replays fixed text as line chunks (same contract as GlmTypist)."""

    def __init__(self, text: str):
        self.text = text

    def type_solution(self, task_prompt: str, feedback: str | None = None,
                      temperature: float | None = None):
        lines = self.text.splitlines(keepends=True)
        for ln in lines:
            yield ln
        return self.text
