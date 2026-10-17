#!/usr/bin/env python3
"""Offline drafter simulator: exact ports of the b10453 n-gram draft
algorithms plus a region-copy oracle, evaluated over the frozen reference
outputs (greedy target outputs of the baseline arm).

Greedy verification model: at each step the drafted tokens are accepted while
they equal the target's next tokens (llama.cpp greedy accept). ``tok/step``
(the step count = target-model forward passes) is the ceiling of the decode
speedup; real speedup is measured on the server.

Usage:
  uv run --with tokenizers python sim_draft.py [--depths 4,8,16]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Protocol

HERE = Path(__file__).resolve().parent
TOKENIZER = Path(
    "/mnt/e/sepalith/campaign-20260915/checkpoints/SFT-v2-r4-lr1e-5/"
    "full/checkpoint-536/tokenizer.json"
)
# marker token ids in prompt space (verified against traces)
CUR_MARKER = (99140, 16053, 49, 102725, 220)   # '<<<<','<<','<',' CURRENT','\n'
EQ_MARKER = (74420, 220)                        # '=======','\n'


def find_sub(hay: list[int], needle: tuple[int, ...], start: int = 0) -> int:
    n = len(needle)
    for i in range(start, len(hay) - n + 1):
        if tuple(hay[i:i + n]) == needle:
            return i
    return -1


def region_span(prompt_ids: list[int]) -> tuple[int, int] | None:
    """Token span [beg, end) of the CURRENT region in the prompt."""
    c = find_sub(prompt_ids, CUR_MARKER)
    if c < 0:
        return None
    e = find_sub(prompt_ids, EQ_MARKER, c + len(CUR_MARKER))
    if e < 0:
        return None
    # region is between 'CURRENT\n' and the '\n======='; strip one leading \n
    beg = c + len(CUR_MARKER)
    if prompt_ids[beg] == 220:  # '\n'
        beg += 1
    return beg, e


# ---------------------------------------------------------------------------
# exact b10453 ports


def ngram_simple_draft(tokens: list[int], sampled: int, size_n: int, size_m: int) -> list[int]:
    """Port of common_ngram_simple_draft (src/common/ngram-map.cpp @ b10453)."""
    cur_len = len(tokens)
    if cur_len <= size_n + size_m + 1:
        return []
    pattern = list(tokens[cur_len - size_n + 1:cur_len]) + [sampled]
    match_pos = 0
    j = cur_len - size_n - 1
    while j > 0:
        if tokens[j:j + size_n] == pattern:
            match_pos = j
            break
        j -= 1
    if match_pos == 0:
        return []
    copy_max = min(size_m, cur_len - (match_pos + size_n))
    if copy_max < size_n:
        return []
    return list(tokens[match_pos + size_n: match_pos + size_n + copy_max])


class NgramMod:
    """Port of common_speculative_impl_ngram_mod (speculative.cpp @ b10453).

    Collision-free dict stands in for the 4M-entry table; occupancy reset and
    chunked adds are replicated.
    """

    def __init__(self, n_match: int, n_min: int, n_max: int) -> None:
        self.n = n_match
        self.n_min, self.n_max = n_min, n_max
        self.table: dict[tuple[int, ...], int] = {}
        self.i_last = 0
        self.n_low = 0

    def begin(self, prompt: list[int]) -> None:
        self.i_last = 0
        self.n_low = 0
        if len(prompt) < self.n:
            return
        for i in range(len(prompt) - self.n):
            self.table[tuple(prompt[i:i + self.n])] = prompt[i + self.n]
        self.i_last = len(prompt) - self.n

    def _add_chunk(self, tokens: list[int]) -> None:
        cur_len = len(tokens)
        if self.i_last + 32 < cur_len:
            for i in range(self.i_last, cur_len - self.n):
                self.table[tuple(tokens[i:i + self.n])] = tokens[i + self.n]
            self.i_last = cur_len - self.n

    def draft(self, tokens: list[int], sampled: int) -> list[int]:
        cur_len = len(tokens)
        if cur_len < self.n:
            return []
        self._add_chunk(tokens)
        # window = last n-1 context tokens + sampled
        window = list(tokens[cur_len - self.n + 1:cur_len]) + [sampled]
        out: list[int] = []
        for _ in range(self.n_max):
            nxt = self.table.get(tuple(window))
            if nxt is None:
                if len(out) < self.n_min:
                    return []
                break
            out.append(nxt)
            window = window[1:] + [nxt]
        return out

    def accept(self, n_accepted: int, drafted: int) -> None:
        if drafted > 0 and n_accepted / drafted < 0.25:
            self.n_low += 1
            if self.n_low >= 5:
                self.table.clear()
                self.n_low = 0
                self.i_last = 0
        else:
            self.n_low = 0


class RegionCopy:
    """Oracle: draft from the CURRENT region, re-aligning after the current
    output position by longest-suffix match (suffix-decoding style)."""

    def __init__(self, region: list[int], depth: int, resync: int = 8) -> None:
        self.region = region
        self.depth = depth
        self.resync = resync
        self.p = 0

    def realign(self, ctx: list[int]) -> int | None:
        """Region position right after the most recent occurrence of the
        longest (<= resync) suffix of ctx; None if no anchor found."""
        for k in range(min(self.resync, len(ctx)), 0, -1):
            key = ctx[-k:]
            for i in range(len(self.region) - k, -1, -1):
                if self.region[i:i + k] == key:
                    return i + k
        return None

    def draft(self, ctx: list[int]) -> list[int]:
        p = self.realign(ctx)
        if p is None:
            return []
        self.p = p
        if self.p >= len(self.region):
            return []
        return self.region[self.p:self.p + self.depth]

    def accepted(self, n: int) -> None:
        self.p += n


# ---------------------------------------------------------------------------
# verification


class Drafter(Protocol):
    def step(self, tokens: list[int], sampled: int, out_so_far: list[int]) -> list[int]: ...

    def on_accept(self, n_accepted: int) -> None: ...


def simulate(prompt: list[int], out: list[int], drafter: Drafter) -> tuple[int, int, int]:
    """Returns (steps, drafted_total, accepted_total) for a drafter."""
    tokens = list(prompt)
    steps = drafted_total = accepted_total = 0
    pos = 0
    while pos < len(out):
        sampled = out[pos]
        draft = drafter.step(tokens, sampled, out[:pos])
        i = 0
        while i < len(draft) and pos + 1 + i < len(out) and draft[i] == out[pos + 1 + i]:
            i += 1
        steps += 1
        drafted_total += len(draft)
        accepted_total += i
        pos += 1 + i
        tokens.extend(out[pos - 1 - i:pos])
        drafter.on_accept(i)
    return steps, drafted_total, accepted_total


def make_simple(n: int, m: int) -> Drafter:
    class D:
        def step(self, tokens: list[int], sampled: int, out_so_far: list[int]) -> list[int]:
            return ngram_simple_draft(tokens, sampled, n, m)

        def on_accept(self, n_accepted: int) -> None:
            pass
    return D()


def make_mod_full(n_match: int, n_min: int, n_max: int) -> Drafter:
    impl = NgramMod(n_match, n_min, n_max)
    begun = [False]
    last_len = [0]

    class D:
        def step(self, tokens: list[int], sampled: int, out_so_far: list[int]) -> list[int]:
            if not begun[0]:
                impl.begin(tokens)
                begun[0] = True
            d = impl.draft(tokens, sampled)
            last_len[0] = len(d)
            return d

        def on_accept(self, n_accepted: int) -> None:
            impl.accept(n_accepted, last_len[0])
    return D()


def make_region(region: list[int], depth: int) -> Drafter:
    rc = RegionCopy(region, depth)

    class D:
        def step(self, tokens: list[int], sampled: int, out_so_far: list[int]) -> list[int]:
            # draft continues AFTER the sampled token
            return rc.draft(list(out_so_far) + [sampled])

        def on_accept(self, n_accepted: int) -> None:
            rc.accepted(n_accepted)
    return D()


def evaluate(traces: list[dict], refs: dict, builders: dict[str, Any]) -> list[dict]:
    rows = []
    for name, build in builders.items():
        per: list[dict] = []
        for tr in traces:
            out = refs.get(tr["trace_id"], {}).get("tokens")
            if not out:
                continue
            if "region" in name:
                span = region_span(tr["prompt_ids"])
                if span is None:
                    continue
                beg, end = span
                drafter = build(tr["prompt_ids"][beg:end])
            else:
                drafter = build()
            steps, drafted, accepted = simulate(tr["prompt_ids"], out, drafter)
            per.append({
                "trace_id": tr["trace_id"], "family": tr["family"],
                "is_noop": tr["is_noop"], "band": tr["band"],
                "out_len": len(out), "steps": steps,
                "drafted": drafted, "accepted": accepted,
            })
        if not per:
            continue
        n_out = sum(p["out_len"] for p in per)
        n_steps = sum(p["steps"] for p in per)
        n_draft = sum(p["drafted"] for p in per)
        n_acc = sum(p["accepted"] for p in per)
        rows.append({
            "config": name, "n": len(per),
            "tok_per_step": round(n_out / n_steps, 3),
            "tok_per_step_med": round(statistics.median(
                [p["out_len"] / p["steps"] for p in per]), 3),
            "accept_rate": round(n_acc / n_draft, 3) if n_draft else 0.0,
            "draft_fired_frac": round(
                sum(1 for p in per if p["drafted"] > 0) / len(per), 3),
            "mean_out_len": round(n_out / len(per), 1),
            **{
                f"tok_per_step:{fam}": round(
                    sum(p["out_len"] for p in per if p["family"] == fam) /
                    max(1, sum(p["steps"] for p in per if p["family"] == fam)), 3)
                for fam in sorted({p["family"] for p in per})
            },
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8,16")
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]

    traces = [json.loads(line) for line in (HERE / "traces.jsonl").read_text().splitlines()]
    refs = json.loads((HERE / "reference_outputs.json").read_text())
    n_with = sum(1 for t in traces if t["trace_id"] in refs)
    print(f"references cover {n_with}/{len(traces)} traces", flush=True)

    builders: dict[str, Any] = {}
    # ngram-simple grid (only size_m >= size_n can ever draft)
    for n in (1, 2, 3, 4, 6, 8):
        for m in (m for m in depths if m >= n):
            builders[f"simple(n={n},m={m})"] = (lambda n=n, m=m: make_simple(n, m))
    # ngram-mod grid: n_match in {8,16,24}, n_min small, n_max = depth
    for nm in (8, 16, 24):
        for nmin in (1, 2):
            for depth in depths:
                builders[f"mod(match={nm},min={nmin},max={depth})"] = (
                    lambda nm=nm, nmin=nmin, d=depth: make_mod_full(nm, nmin, d))
    # region-copy oracle
    for depth in depths:
        builders[f"region-copy(d={depth})"] = (
            lambda region, d=depth: make_region(region, d))

    rows = evaluate(traces, refs, builders)
    rows.sort(key=lambda r: -r["tok_per_step"])
    print(json.dumps(rows, indent=1))
    (HERE / "sim_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
