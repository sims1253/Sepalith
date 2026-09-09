#!/usr/bin/env python3
"""V1d (eval-strategy v2 §3): blind pairwise preference judging on persisted
per-point proposals — "which completion would you rather have at this cursor".

Inputs are EXISTING artifacts, never new serving: the V1a episode-metrics
calibration runs persisted, per trajectory point, the exact prompt the
completion engine saw and the raw proposal each arm produced, at IDENTICAL
cursor states (paired by traj key + variant + t_ms + ctx window):

  pair "v8_2_vs_base"   /mnt/h/sepalith/runs/episode_judged_{base,sft_v8_2}.jsonl
  pair "v7_vs_rl_v2c"   /mnt/h/sepalith/datasets/sim_trajectories_v1/
                        judged_{v7,rl_v2c}.jsonl

Candidates are the EXTENSION-FAITHFUL ghost text: parse_prediction() applied
to the persisted raw proposal (eval_noop_fp.py — stop at >>>>>>>, marker-line
filter, triple-line collapse). A point is usable only if BOTH arms parse to
non-empty text; abstention quality is V1a/noopFP's axis, not this one.

BLINDNESS (hard requirement): the judge prompt carries ONLY the session goal,
the code context at the cursor, and the two candidates under generic slot
labels A/B. Model identities, file origins, accept/reject decisions and eval
outcomes never enter it — run_pair REFUSES to spend a call whose render trips
the sentinel. Per-call rows persist the identity mapping for analysis only,
after the fact.

POSITION DEBIASING: every point is judged TWICE, once in each order. Which
model is shown as slot A in the first call is a seed-locked coin flip
(Random(seed), pid-sorted draw order); the second call is the guaranteed
swap. Debias rule (chosen convention; the design doc names both-orders /
randomized assignment but no tie rule): a model WINS a point only if the
judge picks it in BOTH orders; anything else (tie in either order, or the
verdict flipping with order) is a tie. Secondary metric credits ties at 1/2.

Calibration (design doc §3 V1d):
  - anchor  GT-vs-corrupted (deterministic arg/token swap on the point's gt)
            must win ~>=90% consistently (doc says ~100%; n=50);
  - v8_2-vs-base must beat base DECISIVELY — the V1a anchor (8 accepts vs 0,
    intent-suite norm 0.117 -> 0.685; the doc's pass-band shaping reference
    is the audit's 0.170 -> 0.596 dropout-vs-v8 intent pair). Pass rule:
    Wilson 95% CI lower bound on the debiased v8_2 win-rate > 0.5.
  - v7-vs-rl_v2c is the second calibration pair; expected direction follows
    the intent suite (v7 0.809 vs rl_v2c 0.511 norm) — reported, no kill
    consequence (the noopFP/FP axis cuts the other way; see results doc).

Position-bias telemetry: verdict flip rate across orders + first-slot pick
rate over all calls (a judge that just picks "A" is broken; flag outside
0.35-0.65).

Scale: n=150 points/pair (stratified typing/noop, proportional, seed-locked)
x 2 orders + 50 anchor points x 2 orders ~ 680 calls (mission default; the
design doc gives no n guidance).

Judge backend: cases/backends.py contract — default "agy"
(gemini-3.7-flash-low), one of the design doc's panel_judge backends
(gemini/muse/ox); gemini passed glm-5.3's own three-gate judge calibration
120/120 (docs/research/judge-calibration-gemini-opus.md) and keeps clear of
the zai/muse quotas other agents are consuming. agy is a CLI (no usage
field) -> token figures are chars/4 estimates; call counts are exact.

Resumable wave-file pattern (TU1 precedent): append-only jsonl + .done.jsonl
sidecar of completed (pair, pid, order) keys; re-runs skip done work.

Usage:
  python pairwise_pref.py judge --pair v8_2_vs_base [--n 150] [--seed 3407]
      [--backend agy] [--limit 3]      # --limit = smoke
  python pairwise_pref.py judge --pair anchor_gt_corrupt [--n 50]
  python pairwise_pref.py analyze [--pairs v8_2_vs_base,v7_vs_rl_v2c,anchor_gt_corrupt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # eval_noop_fp
sys.path.insert(0, str(HERE.parent / "synthetic-data"))    # cases.backends
from eval_noop_fp import parse_prediction          # noqa: E402
from cases.backends import extract_json_object, make_backend  # noqa: E402

SEED = 3407
N_PER_PAIR = 150
N_ANCHOR = 50
MAX_CAND_CHARS = 1500        # proposals are persisted at <=600 anyway
MAX_CTX_CHARS = 9000         # median 4-5k, max observed 8k

# -- pair registry -----------------------------------------------------------

RUNS = Path("/mnt/h/sepalith/runs")
SIMT = Path("/mnt/h/sepalith/datasets/sim_trajectories_v1")

PAIR_SPECS = {
    "v8_2_vs_base": {
        "a": ("base", RUNS / "episode_judged_base.jsonl"),
        "b": ("sft_v8_2", RUNS / "episode_judged_sft_v8_2.jsonl"),
        "expected_winner": "sft_v8_2",
    },
    "v7_vs_rl_v2c": {
        "a": ("v7", SIMT / "judged_v7.jsonl"),
        "b": ("rl_v2c", SIMT / "judged_rl_v2c.jsonl"),
        "expected_winner": "v7",
    },
    "b4_vs_v7": {
        "a": ("v7", RUNS / "episode_judged_v7_traj2.jsonl"),
        "b": ("b4", RUNS / "episode_judged_b4_traj2.jsonl"),
        "expected_winner": "b4",
    },
}

ANCHOR_SPEC = {
    "source": RUNS / "episode_judged_sft_v8_2.jsonl",
    "models": ("gt", "corrupted"),   # candidate a = gt, candidate b = corrupted gt
    "expected_winner": "gt",
}

# -- the blind prompt --------------------------------------------------------

PREF_PROMPT = """You are a developer using an editor with inline AI ghost-text completion. \
The editor proposes code at your cursor while you type. Below are TWO candidate completions, \
A and B, produced for the SAME cursor position in the same editing session. \
Which one would you rather have appear as ghost text at this cursor?

Judge it the way the developer would: prefer the completion that is more useful at this exact \
position, correct for the evident intent, needs the least editing or trimming, and does not run \
past what belongs here. Length alone is not a virtue; a wrong or intrusive completion is worse \
than a modest useful one.

THE SESSION GOAL (what the developer is doing):
{goal}

CODE CONTEXT AT THE CURSOR (the completion engine's view: suffix part first, then the file \
prefix; the cursor sits at the seam between them):
<context>
{ctx}
</context>

CANDIDATE A:
<cand_a>
{cand_a}
</cand_a>

CANDIDATE B:
<cand_b>
{cand_b}
</cand_b>

Reply with ONLY a JSON object of the form:
{{"pref": "A"|"B"|"tie", "reason": "<one short sentence>"}}
Use "tie" only when the two are equally (un)desirable at this cursor."""


def render_prompt(goal: str, ctx: str, cand_a: str, cand_b: str) -> str:
    """Blind render: generic slot labels only. Pure, order-deterministic."""
    return PREF_PROMPT.format(
        goal=(goal or "").strip()[:500],
        ctx=(ctx or "").strip()[:MAX_CTX_CHARS],
        cand_a=(cand_a or "").strip()[:MAX_CAND_CHARS] or "(empty)",
        cand_b=(cand_b or "").strip()[:MAX_CAND_CHARS] or "(empty)",
    )


# Distinctive identity/origin/outcome strings that must NEVER appear in a
# rendered prompt (blindness sentinel). Natural code words ("exact", "score",
# "accepted") are deliberately NOT listed — they occur in R source and would
# false-positive the gate.
FORBIDDEN_SUBSTRINGS = (
    "sft_v8_2", "sft_v8", "sft_v7", "sft_v6", "sft_v3", "sft_v2",
    "minicpm5", "minicpm", "rl_grpo", "rl_v2c", "rl_v2d", "rl_v2b",
    "episode_judged", "judged_", "sim_trajectories", "/mnt/h", ".jsonl",
    ".gguf", "false_sug", "correct_stop", "false_suggestion", "base_model",
    "lora", "adapter",
)


def check_blindness(prompt: str) -> list[str]:
    """Forbidden identity substrings present in the prompt (empty = blind)."""
    low = prompt.lower()
    return [f for f in FORBIDDEN_SUBSTRINGS if f in low]


# -- loading + pairing -------------------------------------------------------

def pid_of(key, variant, t_ms, ctx) -> str:
    return hashlib.sha1(
        f"{key}\x00{variant}\x00{t_ms}\x00{ctx}".encode("utf-8", "replace")
    ).hexdigest()[:12]


def load_points(path) -> dict:
    """(key, variant, t_ms, ctx) -> {goal, prompt, gt, label, proposal}."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            for p in t["points"]:
                out[(t["key"], t["variant"], p["t_ms"], p["ctx"])] = dict(
                    goal=t.get("goal", ""), prompt=p.get("prompt", ""),
                    gt=p.get("gt", ""), label=p.get("label", "?"),
                    proposal=p.get("proposal", ""))
    return out


def ghost_text(raw_proposal: str) -> str:
    """Extension-faithful shown text for a persisted raw proposal."""
    return "\n".join(parse_prediction(raw_proposal or ""))


def pair_pool(pts_a: dict, pts_b: dict) -> list[dict]:
    """Common points where BOTH arms parse to non-empty ghost text."""
    pool = []
    keys = sorted(set(pts_a) & set(pts_b),
                  key=lambda x: (str(x[0]), x[1], x[2], x[3] or ""))
    for k in keys:
        a, b = pts_a[k], pts_b[k]
        ca, cb = ghost_text(a["proposal"]), ghost_text(b["proposal"])
        if not ca.strip() or not cb.strip():
            continue
        pool.append(dict(pid=pid_of(*k), traj_key=k[0], variant=k[1], t_ms=k[2],
                         ctx=k[3], goal=a["goal"], prompt=a["prompt"], gt=a["gt"],
                         label=a["label"], cand={"a": ca, "b": cb}))
    return pool


def stratified_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional label stratification, seed-locked, largest-remainder
    allocation (remainder to the last label in sorted order); per-stratum
    draw order = seeded shuffle of the pid-ordered rows."""
    strata: dict[str, list[dict]] = {}
    for row in sorted(pool, key=lambda r: r["pid"]):
        strata.setdefault(row["label"], []).append(row)
    if not pool or n <= 0:
        return []
    labels = sorted(strata)
    take: dict[str, int] = {}
    assigned = 0
    for lab in labels[:-1]:
        take[lab] = math.floor(n * len(strata[lab]) / len(pool))
        assigned += take[lab]
    take[labels[-1]] = n - assigned
    out = []
    for lab in labels:
        rows = strata[lab]
        rng = random.Random(f"{seed}:{lab}")
        order = list(range(len(rows)))
        rng.shuffle(order)
        out.extend(rows[i] for i in order[:take[lab]])
    out.sort(key=lambda r: r["pid"])
    return out


def first_slot_assignments(points: list[dict], seed: int) -> dict[str, bool]:
    """pid -> True iff candidate 'a' is shown as slot A in the FIRST call.

    Seed-locked: one Random(seed) stream over pid-sorted points; the second
    call is always the swap.
    """
    rng = random.Random(seed)
    return {r["pid"]: rng.random() < 0.5
            for r in sorted(points, key=lambda r: r["pid"])}


# -- anchor corruption (GT-vs-corrupted) -------------------------------------

_ARG = r"[A-Za-z._$][\w.$]*|\"[^\"]*\"|'[^']*'"
_ARG_SWAP_RE = re.compile(rf"\(\s*({_ARG})\s*,\s*({_ARG})")


_WORD_TOK = r"[A-Za-z][A-Za-z0-9.]*"        # underscore NOT continued:
# snake_case identifiers split into their word parts so single-identifier
# gt lines (skip_if_not_geweke) still corrupt.


def _swap_words(s: str, i: int, j: int) -> str:
    parts = re.findall(rf"{_WORD_TOK}|[^A-Za-z]+", s)
    words = [k for k, p in enumerate(parts) if re.fullmatch(_WORD_TOK, p)]
    ii, jj = words[i], words[j]
    parts[ii], parts[jj] = parts[jj], parts[ii]
    return "".join(parts)


def corrupt_gt(gt: str) -> str | None:
    """Deterministic corruption of a gt continuation (the V1b corrupted-twin
    rule, adapted): swap the first two comma-separated arguments of the first
    call; else swap the first two distinct word tokens (snake_case splits).
    None when unchanged."""
    m = _ARG_SWAP_RE.search(gt)
    if m and m.group(1) != m.group(2):
        return gt[:m.start()] + f"({m.group(2)}, {m.group(1)}" + gt[m.end():]
    toks = re.findall(_WORD_TOK, gt)
    for i in range(len(toks) - 1):
        for j in range(i + 1, len(toks)):
            if toks[i].lower() != toks[j].lower():
                return _swap_words(gt, i, j)
    return None


# -- judging -----------------------------------------------------------------

def parse_pref(text: str | None) -> str | None:
    """Slot verdict 'a' | 'b' | 'tie' (order mapping happens per-row)."""
    obj = extract_json_object(text)
    if isinstance(obj, dict):
        v = str(obj.get("pref", "")).strip().lower()
        if v in ("a", "b"):
            return v
        if "tie" in v:
            return "tie"
    if isinstance(text, str):                    # lenient fallbacks
        t = text.strip().lower()
        m = re.search(r'"pref"\s*:\s*"(a|b|tie)"', t)
        if m:
            return m.group(1)
        if re.search(r"\btie\b", t):
            return "tie"
        if not re.search(r"\ba\b.*\bb\b.*\ba\b", t):   # "A ... B ... A" = prose
            m = re.search(r"pref(?:erence)?(?:d)?\s*[:=]?\s*\**\[?([ab])\b", t)
            if m:
                return m.group(1)
            m = re.search(r"candidate\s+([ab])\b", t)
            if m:
                return m.group(1)
    return None


def load_done(path: Path) -> set:
    done = set()
    side = done_path(path)
    if side.exists():
        with open(side) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(tuple(json.loads(line)))
    return done


def done_path(path: Path) -> Path:
    return path.with_name(path.stem + ".done.jsonl")


def append_jsonl(path: Path, rows) -> None:
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_sample(pair_name: str, n: int, seed: int) -> tuple[list[dict], tuple]:
    """Return (sampled points with cand dict, model names (a-slot, b-slot))."""
    if pair_name == "anchor_gt_corrupt":
        src = load_points(ANCHOR_SPEC["source"])
        pool = []
        for k, v in src.items():
            cg = corrupt_gt(v["gt"] or "")
            if v["label"] != "typing" or not v.get("gt", "").strip() or not cg:
                continue
            pool.append(dict(pid=pid_of(*k), traj_key=k[0], variant=k[1],
                             t_ms=k[2], ctx=k[3], goal=v["goal"],
                             prompt=v["prompt"], gt=v["gt"], label="typing",
                             cand={"a": v["gt"], "b": cg}))
        return stratified_sample(pool, n, seed), ANCHOR_SPEC["models"]
    spec = PAIR_SPECS[pair_name]
    pool = pair_pool(load_points(spec["a"][1]), load_points(spec["b"][1]))
    return stratified_sample(pool, n, seed), (spec["a"][0], spec["b"][0])


def run_pair(pair_name: str, n: int, seed: int, backend_name: str,
             out_path: Path, limit: int | None = None,
             expected_winner: str | None = None) -> dict:
    """Judge one pair (or the anchor) — both orders per point, resumable.

    `expected_winner` overrides the registry lookup (tests use synthetic
    pairs); production callers leave it None.
    """
    sample, models = build_sample(pair_name, n, seed)
    if limit is not None:
        sample = sample[:limit]
    if expected_winner is None:
        expected_winner = ("gt" if pair_name == "anchor_gt_corrupt"
                           else PAIR_SPECS.get(pair_name,
                                               {}).get("expected_winner")
                           or models[0])
    assign = first_slot_assignments(sample, seed)
    backend = make_backend(backend_name)
    done = load_done(out_path)
    t0 = time.time()
    judged = skipped = errors = 0
    est_in_chars = est_out_chars = 0
    for i, row in enumerate(sample):
        for order in (0, 1):
            if (pair_name, row["pid"], order) in done:
                continue
            a_first = assign[row["pid"]] if order == 0 else not assign[row["pid"]]
            cand_a = row["cand"]["a"] if a_first else row["cand"]["b"]
            cand_b = row["cand"]["b"] if a_first else row["cand"]["a"]
            first_is = models[0] if a_first else models[1]
            second_is = models[1] if a_first else models[0]
            prompt = render_prompt(row["goal"], row["prompt"], cand_a, cand_b)
            rec = dict(pair=pair_name, pid=row["pid"], order=order,
                       label=row["label"], first_is=first_is, second_is=second_is,
                       pref=None, reason=None, error=None, latency_s=0.0)
            leaks = check_blindness(prompt)
            if leaks:                            # refuse to spend a leaking call
                rec["error"] = f"blindness:{leaks}"
                append_jsonl(out_path, [rec])
                continue
            verdict, err, latency = None, None, 0.0
            for _attempt in range(2):            # one same-content retry
                tc = time.time()
                try:
                    text = backend.complete(prompt)
                except Exception as e:
                    err = f"backend:{type(e).__name__}:{str(e)[:120]}"
                    break
                latency += time.time() - tc
                verdict = parse_pref(text)
                if verdict is not None:
                    est_in_chars += len(prompt)
                    est_out_chars += len(text or "")
                    obj = extract_json_object(text)
                    if isinstance(obj, dict) and obj.get("reason"):
                        rec["reason"] = str(obj["reason"])[:400]
                    break
                err = "unparsed"
            rec.update(pref=verdict, error=err, latency_s=round(latency, 2))
            append_jsonl(out_path, [rec])
            append_jsonl(done_path(out_path), [[pair_name, row["pid"], order]])
            judged += 1
            errors += verdict is None
        if (i + 1) % 10 == 0:
            print(f"[{pair_name}] {i + 1}/{len(sample)} points, "
                  f"{judged} calls, {errors} unparsed ({time.time() - t0:.0f}s)",
                  flush=True)
    return dict(pair=pair_name, models=list(models),
                expected_winner=expected_winner,
                sampled=len(sample), new_calls=judged, unparsed=errors,
                out=str(out_path), backend=backend_name,
                stats=backend.stats_summary(), est_tokens_in=est_in_chars // 4,
                est_tokens_out=est_out_chars // 4,
                elapsed_s=round(time.time() - t0, 1))


# -- analysis ----------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test on wins vs losses (ties dropped)."""
    m = wins + losses
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(0, min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** m)


def debias_by_pid(rows: list[dict]) -> dict:
    """pid -> {label, pick per order (model name or None for tie), winner,
    cls} — a winner exists only when BOTH orders agree on the same model."""
    by: dict[str, dict] = {}
    for r in rows:
        if r.get("pref") not in ("a", "b", "tie"):
            continue                              # unparsed/failed calls drop
        pid = r["pid"]
        d = by.setdefault(pid, dict(label=r["label"], pick={}))
        if r["pref"] == "tie":
            d["pick"][r["order"]] = None
        else:
            d["pick"][r["order"]] = (r["first_is"] if r["pref"] == "a"
                                     else r["second_is"])
    out = {}
    for pid, d in by.items():
        if set(d["pick"]) != {0, 1}:
            continue                              # need both orders judged
        v0, v1 = d["pick"][0], d["pick"][1]
        if v0 is None or v1 is None:
            winner, cls = None, "tie"
        elif v0 == v1:
            winner, cls = v0, "consistent"
        else:
            winner, cls = None, "flip"
        out[pid] = dict(label=d["label"], picks=d["pick"], winner=winner, cls=cls)
    return out


def analyze_pair(rows: list[dict], models: tuple[str, str],
                 expected_winner: str) -> dict:
    verd = debias_by_pid(rows)
    n = len(verd)
    other = models[1] if expected_winner == models[0] else models[0]
    wins = {m: 0 for m in models}
    flips = ties = 0
    per_label: dict[str, dict] = {}
    first_slot_picks = total_calls = 0
    for r in rows:                                # first-slot bias over raw calls
        if r.get("pref") in ("a", "b"):
            total_calls += 1
            first_slot_picks += r["pref"] == "a"
    for d in verd.values():
        L = per_label.setdefault(
            d["label"], dict(n=0, **{m: 0 for m in models}, ties=0, flips=0))
        L["n"] += 1
        if d["cls"] == "consistent":
            wins[d["winner"]] += 1
            L[d["winner"]] += 1
        elif d["cls"] == "flip":
            flips += 1
            L["flips"] += 1
        else:
            ties += 1
            L["ties"] += 1
    res = dict(models=list(models), expected_winner=expected_winner,
               n_points=n, wins=wins, ties=ties, flips=flips,
               win_rate=round(wins[expected_winner] / n, 4) if n else None,
               # Order flips are ties for scoring; retain separate counts
               # so callers can still inspect position sensitivity.
               ties_half_rate=round((wins[expected_winner] + 0.5 * (ties + flips)) / n, 4)
               if n else None,
               win_rate_wilson95=None, sign_p=None,
               first_slot_pick_rate=round(first_slot_picks / total_calls, 4)
               if total_calls else None,
               first_slot_calls=total_calls,
               per_label=per_label)
    if n:
        # Pointwise statistics; repeated points within trajectories are
        # not independent, and these estimates do not adjust for clustering.
        lo, hi = wilson_ci(wins[expected_winner], n)
        res["win_rate_wilson95"] = [round(lo, 4), round(hi, 4)]
        res["sign_p"] = sign_test_p(wins[expected_winner], wins[other])
    return res


def analyze(pairs: list[str], outdir: Path) -> dict:
    report = {}
    for pair_name in pairs:
        out_path = outdir / f"results_pairwise_pref_{pair_name}.jsonl"
        rows = [json.loads(l) for l in open(out_path) if l.strip()]
        if pair_name == "anchor_gt_corrupt":
            models = ANCHOR_SPEC["models"]
            expected = ANCHOR_SPEC["expected_winner"]
        else:
            spec = PAIR_SPECS[pair_name]
            models = (spec["a"][0], spec["b"][0])
            expected = spec["expected_winner"]
        cov = dict(n_rows=len(rows),
                   unparsed=sum(1 for r in rows
                                if r.get("pref") not in ("a", "b", "tie")))
        report[pair_name] = dict(coverage=cov, **analyze_pair(rows, models, expected))
    return report


# -- cli ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--pair", required=True,
                   choices=list(PAIR_SPECS) + ["anchor_gt_corrupt"])
    j.add_argument("--n", type=int, default=None)
    j.add_argument("--seed", type=int, default=SEED)
    j.add_argument("--backend", default="agy")
    j.add_argument("--limit", type=int, default=None,
                   help="judge only the first K points (smoke)")
    j.add_argument("--out", default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--pairs", default=",".join(list(PAIR_SPECS)
                                               + ["anchor_gt_corrupt"]))
    a.add_argument("--out", default=str(HERE / "results_pairwise_pref_analysis.json"))
    args = ap.parse_args(argv)

    if args.cmd == "judge":
        n = args.n or (N_ANCHOR if args.pair == "anchor_gt_corrupt" else N_PER_PAIR)
        out = Path(args.out or HERE / f"results_pairwise_pref_{args.pair}.jsonl")
        summary = run_pair(args.pair, n, args.seed, args.backend, out, args.limit)
        print(json.dumps(summary, indent=2))
        return 0

    report = analyze([p.strip() for p in args.pairs.split(",") if p.strip()], HERE)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
