#!/usr/bin/env python3
"""RL pipeline trial: GRPO with validator rewards on zeta2 edit scenarios.

First real RL run on our own model (night 2026-08-20). Thesis under test:
environments-as-data — single-model GRPO where the reward is the SAME
exact/validator score the eval harness uses, on TRAIN-split rows only.

Method (kept deliberately simple; SAO lesson: single-rollout-style RL is
fine for agentic/edit tasks, GRPO with small groups is the same spirit):

  1. MERGE the v6 SFT LoRA to a plain base (/tmp) via
     save_pretrained_merged — RL trains a FRESH LoRA on top, never the
     original adapter in place (artifacts stay separable).
  2. GRPO (trl GRPOTrainer, unsloth-patched) on TRAIN-split rows from
     /mnt/h/sepalith/datasets/sft_v6/train.jsonl for the chosen scenario
     families. Prompts/targets are the EXACT rows the assembler rendered
     with edit_row()/render_zeta2() — zero re-rendering, zero skew. Any
     row whose prompt appears in the materialized holdout split
     (sft_v3/eval.jsonl, the authoritative 3% package holdout that
     eval_scenarios.py scores against) is EXCLUDED (verified: 5 rows).
  3. Reward per completion = exact + 0.2 * line_f1 where
       exact    = parsed prediction == normalized target lines
                  (same norm as eval_scenarios' exact: rstrip per line,
                  trailing blanks popped — run_eval.parse_pred + norm),
       line_f1  = scenarios.exact_reward (verbatim copy below; the
                  scenarios module cannot load in .venv-sft because
                  tree_sitter_r is not installed there — the function is
                  pure difflib and identical to the eval path).
     Families whose target is an UNCHANGED region (no_op) reward
     emit-nothing correctness through the same code path: their target
     minus the UPDATED marker is empty, so the reward is 1.2 iff the
     model emits just the marker.
  4. BOS parity: SFT tokenized text with add_special_tokens=True (BOS id
     0 prepended); trl's GRPO tokenizes prompts with
     add_special_tokens=False, so every prompt string here starts with
     the tokenizer's literal BOS token — assert-checked at startup.

Usage (.venv-sft, GPU free per day-queue):
  python rl_smoke.py --merge                 # step 1 only
  python rl_smoke.py --smoke                 # 10 steps, 32 prompts, /tmp out
  python rl_smoke.py --steps 300             # full run
  python rl_smoke.py --schedule ordered ...  # E1 EL-scheduler arm
  python rl_smoke.py --dry-run 20            # E1 plan mode (CPU, no model)
Output: --out (default /mnt/h/sepalith/runs/rl_grpo_v1): checkpoints,
final_lora/, rl_metrics.jsonl (one line per optimizer step).

E1 PRE-REGISTRATION (EL-scheduler retrofit, queue row E1, 2026-09-04; paper
arXiv:2609.04128 "6/8 over 8 rollouts" tier admission, mapped to our GRPO
geometry — num_generations=4, one optimizer step == one generation batch ==
8 unique prompts x 4 completions, reward fn called once per step):

  Treatment (--schedule ordered; default --schedule random is BYTE-IDENTICAL
  to the banked runs: same build_dataset, same stock GRPOTrainer + trl
  RepeatSampler, same RNG streams — the ordered machinery is never
  constructed when the flag is off). Family order (tiers, easiest first):
  pipe_rewrite -> rename_propagation -> format_propagation, plus a final
  compound tier when any compound_* family is present in the pools. Tiers
  CUMULATE once admitted; families outside the tier list (no_op,
  finish_block, refine_*) are background: always drawable at their quota
  share, so the ordered-vs-random comparison holds the quota table fixed
  and varies only the ordering/gating of the tiered families.

  Adapted admission rule (paper: a tier is mastered when a task passes >=
  6 of 8 rollouts, i.e. an observed 75% pass rate; ours): the next tier is
  admitted when, pooled over the last K=5 optimizer steps, >= 75% of the
  FRONTIER tier's GRPO groups are FULLY solved (4/4 completions exact) and
  at least 8 frontier groups were observed in the window (min-count guard;
  below it the gate defers, never fails). Justification, in three parts:
  (1) unit-of-measure: our "8 rollouts of one task" is the GRPO group of
      num_generations=4 completions of one prompt; the ratio-preserving
      per-group analog of 6/8 is 3/4, but one 4-rollout group is far too
      noisy a gate (one binomial draw), so the rate is pooled over the
      window instead — with >=8 pooled groups the full-solve MLE's SE at
      the threshold is ~0.13, comparable to the paper's single-task
      8-rollout gate (SE ~0.15 at p=0.75).
  (2) why FULL groups (4/4) rather than >=3/4 groups as the pass unit: a
      fully-solved group is exactly a zero-advantage group in GRPO — it
      carries no further learning signal. The gate therefore fires when
      >= 75% of the tier's groups have stopped teaching, which is the EL
      principle restated in GRPO currency.
  (3) strictness: 0.75-of-fully-solved implies a per-completion exact rate
      of ~0.93 (0.93^4 ~ 0.75) vs the paper's 0.75. Deliberate: admitting
      the next tier early re-dilutes group variance — the exact pathology
      E1 exists to test. K=5 steps (~40 groups window-wide, ~12 frontier
      groups at the starting quota mix) keeps admission possible inside a
      50-step readout window. Known interplay: if the frontier tier's
      quota share is so small that 5 steps pool < 8 groups (e.g. run2
      quotas), admission defers indefinitely — visible as
      el_tiers_admitted flatlining in rl_metrics.jsonl.

  Pre-registered readout (paper Fig 5; shared field contract with O2/O3 so
  their telemetry reads the same names): every step line of
  rl_metrics.jsonl gains psg_rate / full_group_rate / n_groups (partial-
  solved = 0 < solved < 4 of 4 exact in a group); the step-50 line (or a
  final event line if the run is shorter) gains first50_psg_rate +
  first50_reward + first50_psg_<family> pooled over steps 1-50. Kill rule
  per queue row E1: no partial-solved-rate gain vs random at matched
  rollout budget -> KILL before any evolver build.

  CPU sanity (no GPU): --dry-run N prints the tier plan + first N
  generation-batch draws without loading a model; unit tests in
  test_rl_smoke_el.py cover tier admission, ordering, quota interplay and
  the sampler contract with a mocked reward stream.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))           # run_eval (light deps)
from run_eval import norm, parse_pred                    # noqa: E402  exact eval path

SFT_LORA = Path("/mnt/h/sepalith/runs/sft_v6_minicpm5/final_lora")
MERGED_BASE = Path("/tmp/merged_rl_v6_base")            # RL base (merged v6)
TRAIN_JSONL = Path("/mnt/h/sepalith/datasets/sft_v6/train.jsonl")
HOLDOUT_REF = Path("/mnt/h/sepalith/datasets/sft_v3/eval.jsonl")
UPDATED_MARK = ">>>>>>> UPDATED"
STOP = ">>>>>>> UPDATED"

# verbatim copy of scenarios.exact_reward (experiments/synthetic-data/
# scenarios.py) — scenarios.py imports tree_sitter_r at module load, which
# is absent in .venv-sft; the function is pure difflib and byte-identical.
import difflib                                           # noqa: E402


def exact_reward(pred_lines, region_new_lines) -> float:
    """1.0 on exact match after rstrip-normalisation, else line-F1 (difflib)."""
    p = [l.rstrip() for l in (pred_lines or [])]
    g = [l.rstrip() for l in (region_new_lines or [])]
    while p and p[-1] == "":
        p.pop()
    while g and g[-1] == "":
        g.pop()
    if p == g:
        return 1.0
    if not p or not g:
        return 0.0
    sm = difflib.SequenceMatcher(a=p, b=g, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    prec = matched / len(p)
    rec = matched / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


FAMILY_QUOTA = {          # train-split draws; held-out packages excluded
    "rename_propagation": 1400,   # v6 exact 0.820 — headroom, not cold
    "format_propagation": 1400,   # v6 exact 0.522 — most headroom
    "no_op": 350,                 # emit-nothing discipline (intent guard)
    "pipe_rewrite": 150,          # tiny anchor for the near-ceiling 0.944
}
# RL-run-2 no-op arm (docs/research/v8-noop-random-cursor.md layer 3): the
# no_op share rises to ~30% of prompts and — on v8 data — covers all SIX
# stop geometries (the 4 untrained ones included). The reward path is
# unchanged (empty-target rows already give all-or-nothing: 1.2 iff the
# completion is just the UPDATED marker); the run metric to watch is the
# per-family no_op exact in rl_metrics.jsonl — that IS 1 - FP-rate on the
# trained geometries. The untrained-geometry gate (<20% FP, from 100%) is
# eval-side: eval_noop_fp.py, never this trainer.
FAMILY_QUOTA_RUN2 = {
    "rename_propagation": 1400,
    "format_propagation": 1400,
    "no_op": 1600,                # ~30% of the prompt draw
    "pipe_rewrite": 150,
    "finish_block": 800,          # present in sft_v8 renders; skipped if absent
}
SHAPING = 0.2             # reward = exact + SHAPING * line_f1 (max 1.2)

# ---------------------------------------------------------------------------
# E1: ordered-difficulty (EL) scheduling — pure-python, torch-free so the
# unit tests run CPU-only. The torch/trl wiring lives in main() behind
# --schedule ordered; the default path never touches any of this.
# ---------------------------------------------------------------------------

EL_TIER_ORDER = ("pipe_rewrite", "rename_propagation", "format_propagation")
EL_THRESHOLD = 0.75       # adapted "6/8 over 8 rollouts" — see header
EL_WINDOW_K = 5           # optimizer steps in the admission window
EL_MIN_GROUPS = 8         # pooled frontier groups before a gate is valid


class OrderedScheduler:
    """Tier-gated prompt-draw scheduler (E1). Tiers cumulate; the next tier
    is admitted only when the frontier tier's fully-solved-group rate over
    the last K steps clears the threshold (see header pre-registration).

    pools: {family: [dataset_index, ...]} — the quota-satisfied draw pools
    (same pools the random baseline consumes; ordering is the only
    treatment). Families not in any tier are background: always drawable.
    """

    def __init__(self, pools, tier_order=EL_TIER_ORDER,
                 threshold=EL_THRESHOLD, window_k=EL_WINDOW_K,
                 min_groups=EL_MIN_GROUPS, seed=3407):
        self.pools = pools
        self.threshold = threshold
        self.window_k = window_k
        self.min_groups = min_groups
        compound = sorted(f for f in pools if f.startswith("compound"))
        self.tiers = [[f] for f in tier_order if f in pools]
        if compound:
            self.tiers.append(compound)      # final tier, admitted as one
        tiered = {f for t in self.tiers for f in t}
        self.background = sorted(f for f in pools if f not in tiered)
        self.admitted = 1 if self.tiers else 0
        self.window = deque()                # per-step {fam: [n_groups, n_full]}
        self.rng = random.Random(seed)
        self.decks = {f: deque() for f in pools}
        self.draws = 0                       # unique prompts drawn (batches)

    # -- draw side ----------------------------------------------------------
    def active_families(self):
        act = [f for t in self.tiers[: self.admitted] for f in t]
        return act + self.background

    def _pop(self, fam, in_batch):
        """Next index from fam's deck (reshuffle on exhaust); skip dupes
        that would repeat a prompt already drawn into this batch."""
        for _ in range(len(self.pools[fam]) + 1):
            if not self.decks[fam]:
                deck = list(self.pools[fam])
                self.rng.shuffle(deck)
                self.decks[fam] = deque(deck)
            idx = self.decks[fam].popleft()
            if idx not in in_batch:
                return idx
        return None                           # pool smaller than batch slot

    def next_batch(self, batch_prompts):
        """One generation batch: `batch_prompts` unique indices, families
        drawn quota-proportionally over the active set."""
        fams = self.active_families()
        weights = [len(self.pools[f]) for f in fams]
        out = []
        for _ in range(batch_prompts):
            fam = self.rng.choices(fams, weights=weights, k=1)[0]
            idx = self._pop(fam, out)
            if idx is not None:
                out.append(idx)
        self.draws += len(out)
        return out

    # -- observe side (called once per optimizer step by MetricsCb) --------
    def observe(self, per_family):
        """per_family: {family: [n_groups, n_full]} for THIS step's groups."""
        self.window.append({f: list(v) for f, v in per_family.items()
                            if f not in self.background})
        while len(self.window) > self.window_k:
            self.window.popleft()

    def frontier(self):
        if not self.tiers or self.admitted >= len(self.tiers):
            return []
        return list(self.tiers[self.admitted - 1])

    def maybe_admit(self):
        """Admit the next tier if the frontier clears the adapted gate.
        Returns the newly admitted tier's families, or None."""
        if not self.tiers or self.admitted >= len(self.tiers):
            return None
        frontier = set(self.frontier())
        n_groups = n_full = 0
        for step_counts in self.window:
            for fam, (g, full) in step_counts.items():
                if fam in frontier:
                    n_groups += g
                    n_full += full
        if n_groups < self.min_groups:
            return None
        if n_full / n_groups >= self.threshold:
            self.admitted += 1
            self.window.clear()              # fresh window for new frontier
            return list(self.tiers[self.admitted - 1])
        return None

    def state(self):
        """Compact per-step telemetry for rl_metrics.jsonl."""
        frontier = self.frontier()
        n_groups = n_full = 0
        for step_counts in self.window:
            for fam, (g, full) in step_counts.items():
                if fam in frontier:
                    n_groups += g
                    n_full += full
        return {"el_tiers_admitted": self.admitted,
                "el_frontier": "+".join(frontier) if frontier else "done",
                "el_window_groups": n_groups,
                "el_window_full_rate": round(n_full / n_groups, 4)
                if n_groups else None}


class ELSamplerStream:
    """Drop-in for trl's RepeatSampler index stream (duck-typed — no torch
    subclass needed): identical chunk/mini-repeat/repeat geometry, but the
    unique-prompt chunks come from the OrderedScheduler instead of a single
    randperm. One chunk == one generation batch == one optimizer step."""

    def __init__(self, scheduler, num_samples, num_generations,
                 unique_per_batch, repeat_count):
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.num_generations = num_generations
        self.unique_per_batch = unique_per_batch
        self.repeat_count = repeat_count
        self.num_chunks = num_samples // unique_per_batch

    def __len__(self):
        return (self.num_chunks * self.unique_per_batch
                * self.num_generations * self.repeat_count)

    def __iter__(self):
        for _ in range(self.num_chunks):
            chunk = self.scheduler.next_batch(self.unique_per_batch)
            for _ in range(self.repeat_count):
                for idx in chunk:
                    for _ in range(self.num_generations):
                        yield idx


def group_stats(recs):
    """Group-level stats from one step's STASH flush. recs: (family, exact,
    reward[, prompt]) tuples — prompt groups the completions into GRPO
    groups (4 per prompt at num_generations=4); legacy 3-tuples fall back
    to one group per completion. Returns None if recs is empty."""
    if not recs:
        return None
    groups = {}
    for r in recs:
        fam, ex = r[0], r[1]
        key = r[3] if len(r) > 3 and r[3] is not None else id(r)
        g = groups.setdefault(key, [fam, 0, 0])
        g[1] += 1
        g[2] += int(ex)
    n_groups = len(groups)
    full = partial = 0
    per_family = {}
    for fam, size, solved in groups.values():
        pf = per_family.setdefault(fam, [0, 0, 0])  # n, full, partial
        pf[0] += 1
        if solved == size:
            full += 1
            pf[1] += 1
        elif solved > 0:
            partial += 1
            pf[2] += 1
    return {"n_groups": n_groups,
            "full_group_rate": full / n_groups,
            "psg_rate": partial / n_groups,      # partial-solved-group rate
            "per_family": per_family}


class First50Readout:
    """Pools the pre-registered E1/O2 readout over steps 1-50: overall +
    per-family partial-solved-group rate, and mean reward."""

    def __init__(self, max_step=50):
        self.max_step = max_step
        self.n = 0
        self.reward_sum = 0.0
        self.g = 0
        self.partial = 0
        self.per_family = {}

    def add(self, step, recs, stats):
        if step > self.max_step or not stats:
            return
        self.n += len(recs)
        self.reward_sum += sum(r[2] for r in recs)
        self.g += stats["n_groups"]
        self.partial += round(stats["psg_rate"] * stats["n_groups"])
        for fam, (gn, _f, p) in stats["per_family"].items():
            pf = self.per_family.setdefault(fam, [0, 0])
            pf[0] += gn
            pf[1] += p

    def emit(self):
        if self.g == 0:
            return {}
        out = {"first50_psg_rate": round(self.partial / self.g, 4),
               "first50_reward": round(self.reward_sum / self.n, 4),
               "first50_n_groups": self.g}
        for fam in sorted(self.per_family):
            gn, p = self.per_family[fam]
            out[f"first50_psg_{fam}"] = round(p / gn, 4) if gn else None
        return out

# ---------------------------------------------------------------------------
# dataset: train.jsonl -> {prompt (BOS-prefixed), target, family}
# ---------------------------------------------------------------------------


def gt_lines(target: str):
    """Normalized target region lines: strip the UPDATED marker, eval norm."""
    body = target
    for suf in (f"\n{UPDATED_MARK}", UPDATED_MARK):
        if body.endswith(suf):
            body = body[: -len(suf)]
            break
    return norm(body.splitlines())


def build_dataset(tok, quotas=FAMILY_QUOTA, seed=3407,
                  data_path=TRAIN_JSONL, refine_path=None):
    import random
    holdout = set()
    for line in open(HOLDOUT_REF):
        r = json.loads(line)
        if r.get("family") in ("rename_propagation", "pipe_rewrite",
                               "format_propagation"):
            holdout.add(r["prompt"])
    pools, excluded = {f: [] for f in quotas}, {"holdout": 0, "dupe": 0}
    seen = set()
    for line in open(data_path):
        r = json.loads(line)
        fam = r.get("family")
        if fam not in quotas:
            continue
        if r["prompt"] in holdout:
            excluded["holdout"] += 1
            continue
        if r["prompt"] in seen:
            excluded["dupe"] += 1
            continue
        seen.add(r["prompt"])
        pools[fam].append(r)
    rng = random.Random(seed)
    rows = []
    for fam, pool in pools.items():
        rng.shuffle(pool)                       # random draw, not file order
        rows.extend(pool[: quotas[fam]])
    # refinement arm (recursive validator feedback, user idea 2026-08-23):
    # prompts embed the model's own failed attempt + a leak-free diagnostic
    # (#! validator: ... / #! feedback: dismissed); target/reward unchanged.
    # 1x — ~9% of the prompt draw: an arm-within-the-run, not a quota
    # family (the smoke showed occasional nonzero rewards zero-shot, so
    # GRPO has group variance to learn from without a format warm-start).
    if refine_path:
        n_ref = 0
        for line in open(refine_path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("family", "").startswith("refine_"):
                rows.append(r)
                n_ref += 1
        print(f"refinement arm: {n_ref} prompt slots", flush=True)
    rng.shuffle(rows)
    # BOS parity + length guard: prompts must fit max_prompt_length tokens
    # (trl truncates from the RIGHT, which would cut the <[fim-middle]> cue),
    # and targets must fit the completion cap (completions stop at the cap —
    # the SFT model emits no EOS — so rows with longer targets would be
    # un-winnable and are dropped, count logged). Prompt cap 480: HF generate
    # prefill computes full-vocab logits at EVERY prompt position of the
    # padded generation batch (32 x L x 99.6k x 2B); with L~800 that spiked
    # 6.25GB and OOM'd against the twin-coexistence fraction cap (step 41
    # of the first full attempt); 480 keeps the spike ~3GB.
    bos = tok.bos_token or ""   # Qwen3.5 tokenizer has no BOS (bos_token_id null): identity prefix
    max_prompt_tok, max_target_tok = 0, 0
    dropped_len = 0
    out = []
    for r in rows:
        p = bos + r["prompt"]
        pt = len(tok(p, add_special_tokens=False)["input_ids"])
        tt = len(tok(r["target"], add_special_tokens=False)["input_ids"])
        if pt > 480 or tt > 170:
            dropped_len += 1
            continue
        max_prompt_tok, max_target_tok = max(max_prompt_tok, pt), max(max_target_tok, tt)
        out.append({"prompt": p, "target": r["target"], "family": r["family"]})
    stats = {f: sum(1 for x in out if x["family"] == f) for f in quotas}
    excluded["len"] = dropped_len
    return out, dict(excluded=excluded, counts=stats,
                     max_prompt_tok=max_prompt_tok, max_target_tok=max_target_tok)


# reward-fn metric stash: (family, exact, reward[, prompt]) per completion,
# flushed per optimizer step by the callback below (on-policy: one reward
# call per generation round == one optimizer step). The optional prompt
# element (E1/O2 group readout) lets the flush regroup completions into
# GRPO groups; legacy 3-tuples are tolerated by group_stats().
STASH = deque()


def scenario_reward(prompts, completions, completion_ids=None, target=None,
                    family=None, trainer_state=None, **kw):
    out = []
    for fam, comp, tgt, p in zip(family, completions, target, prompts):
        pred = parse_pred("zeta2", comp)      # exact eval parsing path
        gt = gt_lines(tgt)
        ex = int(pred == gt)
        rew = ex + SHAPING * exact_reward(pred, gt)
        STASH.append((fam, ex, rew, p))
        out.append(rew)
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def pick_quotas(families_csv, run2=False, no_op_n=None, smoke=False):
    """Quota table after CLI overrides (pure; shared by train + dry-run)."""
    fams = [x.strip() for x in families_csv.split(",")]
    quotas = {f: n for f, n in
              (FAMILY_QUOTA_RUN2 if run2 else FAMILY_QUOTA).items() if f in fams}
    if no_op_n is not None and "no_op" in quotas:
        quotas["no_op"] = no_op_n
    if smoke:
        quotas = {f: min(n, 8) for f, n in quotas.items()}
    return quotas


def _dry_run_tokenizer(model_dir):
    """Tokenizer for --dry-run only: the real one when the merged base
    exists, else a whitespace-counting stand-in (approximate length
    filter — printed caveat; good enough to show tier/batch structure)."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir), True
    except Exception as e:                    # missing dir / no fast tokenizer
        print(f"dry-run: real tokenizer unavailable ({e}); "
              "using whitespace approximation for the length filter",
              flush=True)

        class _FakeTok:
            bos_token = "<bos>"
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": text.split()}
        return _FakeTok(), False


def do_dry_run(args, n_batches=20):
    """E1 plan mode: tier plan + first N generation-batch draws, no model,
    no GPU context, no CUDA. Admission shown at its INITIAL state (tier 1
    only) — live admission depends on observed pass rates (unit tests in
    test_rl_smoke_el.py cover the dynamics with a mocked reward stream)."""
    tok, real_tok = _dry_run_tokenizer(args.model)
    quotas = pick_quotas(args.families, args.run2, args.no_op_n, args.smoke)
    rows, dstat = build_dataset(tok, quotas, data_path=Path(args.data),
                                refine_path=args.refine_data)
    pools = {}
    for i, r in enumerate(rows):
        pools.setdefault(r["family"], []).append(i)
    sched = OrderedScheduler(
        pools, threshold=args.el_threshold, window_k=args.el_window,
        seed=3407)
    per_batch = max(1, args.bs * args.ga // args.num_generations)
    print(json.dumps(dict(
        dry_run=True, real_tokenizer=real_tok, n_rows=len(rows),
        dataset=dstat, tier_order=[t for t in sched.tiers],
        background=sched.background,
        el=dict(threshold=args.el_threshold, window_k=args.el_window,
                min_groups=EL_MIN_GROUPS, num_generations=args.num_generations,
                unique_prompts_per_batch=per_batch))), flush=True)
    for b in range(n_batches):
        batch = sched.next_batch(per_batch)
        comp = {}
        for idx in batch:
            fam = rows[idx]["family"]
            comp[fam] = comp.get(fam, 0) + 1
        print(f"ELDRAW batch {b + 1}: " + json.dumps(
            dict(families=comp, tiers_admitted=sched.admitted,
                 frontier="+".join(sched.frontier()) or "done"),
            sort_keys=True), flush=True)
    print("DRYRUN DONE (initial-admission draws only; live gating is "
          "reward-driven)", flush=True)


def gpu_guard(limit_mib=20000):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                          "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    util, mem = [x.strip().split()[0] for x in gpu.split(",")]
    print(f"GPU check: util={util}% mem={mem}MiB", flush=True)
    # shared-machine policy is abort >8GB; tonight's 03:40 user override lets
    # the junior twin POC (~13.7GB cap) share the 5090 with RL (senior), so
    # the launch gate is 20GB instead — RL needs ~15GB and 15+13.7 < 32.
    if float(mem) > limit_mib:
        raise SystemExit(f"GPU busy (>{limit_mib}MiB used) — aborting")


def do_merge():
    from unsloth import FastLanguageModel
    if (MERGED_BASE / "config.json").exists():
        print(f"merge: {MERGED_BASE} already exists, skipping")
        return
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(SFT_LORA), max_seq_length=2048, dtype=None,
        load_in_4bit=False)
    model.save_pretrained_merged(str(MERGED_BASE), tokenizer,
                                 save_method="merged_16bit")
    print(f"merged -> {MERGED_BASE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="only merge v6 LoRA")
    ap.add_argument("--smoke", action="store_true", help="10 steps, 32 prompts")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8, help="per-device batch (completions)")
    ap.add_argument("--ga", type=int, default=4, help="grad accumulation")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--memfrac", type=float, default=0.62,
                    help="cuda memory fraction cap (twin POC co-existence)")
    ap.add_argument("--families", default=",".join(FAMILY_QUOTA))
    ap.add_argument("--data", default=str(TRAIN_JSONL),
                    help="train split jsonl (family-keyed zeta2 renders)")
    ap.add_argument("--model", default=str(MERGED_BASE),
                    help="merged GRPO base dir")
    ap.add_argument("--run2", action="store_true",
                    help="RL-run-2 no-op arm profile (see FAMILY_QUOTA_RUN2)")
    ap.add_argument("--refine-data", default=None,
                    help="refinement prompt jsonl (recursive validator "
                         "feedback arm, build_refinement_set.py output)")
    ap.add_argument("--no-op-n", type=int, default=None,
                    help="override the no_op prompt quota (the FP-intent "
                         "knee test: 15%% share ≈ 800)")
    ap.add_argument("--schedule", default="random",
                    choices=["random", "ordered"],
                    help="random (default) = banked fixed-quota draw, "
                         "byte-identical; ordered = E1 EL-scheduler "
                         "(pipe -> rename -> format tiers, pass-rate "
                         "admission — see module header)")
    ap.add_argument("--el-window", type=int, default=EL_WINDOW_K,
                    help="EL admission window K in optimizer steps")
    ap.add_argument("--el-threshold", type=float, default=EL_THRESHOLD,
                    help="EL admission threshold on frontier fully-solved-"
                         "group rate (adapted 6/8; see header)")
    ap.add_argument("--dry-run", type=int, default=None, metavar="N",
                    help="E1 plan mode: print the tier plan + first N "
                         "generation-batch draws and exit (no model load, "
                         "no GPU)")
    ap.add_argument("--out", default="/mnt/h/sepalith/runs/rl_grpo_v1")
    ap.add_argument("--prescreen", default=None,
                    help="O2 admission ledger (prescreen_v1.jsonl): keep only "
                         "admitted prompts; fail-closed on provenance mismatch")
    args = ap.parse_args()

    if args.merge:
        do_merge()
        return

    if args.dry_run is not None:
        do_dry_run(args, n_batches=args.dry_run)
        return

    gpu_guard()
    from unsloth import FastLanguageModel          # patch BEFORE trl import
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    quotas = pick_quotas(args.families, args.run2, args.no_op_n, args.smoke)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, dtype=None,
        load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
    model.config.use_cache = True      # KV cache for GRPO's generate calls
    if args.memfrac:
        torch.cuda.set_per_process_memory_fraction(args.memfrac)
        print(f"cuda memory fraction capped at {args.memfrac}", flush=True)

    # BOS parity self-check: our prompt strings must tokenize to BOS-first
    # (BOS-less tokenizers like Qwen3.5 have no BOS to assert — skip)
    if tokenizer.bos_token is None:
        print("BOS parity check skipped: tokenizer has no BOS (Qwen3.5)", flush=True)
    else:
        ids = tokenizer(tokenizer.bos_token + "x", add_special_tokens=False)["input_ids"]
        assert ids[0] == tokenizer.bos_token_id, \
            f"BOS parity broken: {ids[:3]} vs bos_id={tokenizer.bos_token_id}"

    rows, dstat = build_dataset(tokenizer, quotas,
                                data_path=Path(args.data),
                                refine_path=args.refine_data)
    if args.prescreen:
        import hashlib as _h
        ledger_path = Path(args.prescreen)
        prov_path = ledger_path.parent / "provenance.json"
        if not ledger_path.is_file() or not prov_path.is_file():
            raise SystemExit("O2 fail-closed: prescreen ledger or provenance missing")
        prov = json.loads(prov_path.read_text())
        if prov.get("seed") != 3407 or prov.get("k") != 16 or                 tuple(prov.get("band", ())) != (3, 13):
            raise SystemExit("O2 fail-closed: provenance seed/k/band mismatch")
        if prov.get("data") != str(Path(args.data).resolve()):
            raise SystemExit("O2 fail-closed: provenance data path mismatch: %s vs %s"
                             % (prov.get("data"), Path(args.data).resolve()))
        admitted = set()
        for line in ledger_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("admitted"):
                admitted.add(rec["prompt_hash"])
        kept, kept_f = [], {}
        for r in rows:
            ph = _h.sha1(r["prompt"].encode()).hexdigest()
            if ph in admitted:
                kept.append(r)
                kept_f[r["family"]] = kept_f.get(r["family"], 0) + 1
        print(json.dumps(dict(prescreen_filter=dict(input=len(rows), kept=len(kept),
                                                    per_family=kept_f))), flush=True)
        if not kept:
            raise SystemExit("O2 fail-closed: admission filter kept zero prompts")
        rows = kept
    print(json.dumps(dict(dataset=dstat, n_rows=len(rows),
                          bos=tokenizer.bos_token)), flush=True)
    ds = Dataset.from_list(rows)
    # E1 ordered-difficulty scheduling: build the per-family index pools
    # from the SAME quota-satisfied rows the random baseline consumes, then
    # hand prompt-draw ordering to the scheduler (ordering-only treatment).
    el_sched = None
    if args.schedule == "ordered":
        pools = {}
        for i, r in enumerate(rows):
            pools.setdefault(r["family"], []).append(i)
        el_sched = OrderedScheduler(
            pools, threshold=args.el_threshold, window_k=args.el_window,
            seed=3407)
        print(f"EL schedule: tiers={el_sched.tiers} "
              f"background={el_sched.background} "
              f"threshold={args.el_threshold} K={args.el_window} "
              f"min_groups={EL_MIN_GROUPS}", flush=True)
    max_prompt_len = 512
    assert dstat["max_prompt_tok"] + 1 <= max_prompt_len, \
        f"prompt overflow: {dstat['max_prompt_tok']} > {max_prompt_len - 1}"
    # 192-token cap: the SFT model emits no EOS, so every completion runs to
    # the cap; on WSL2 each decode step is launch-overhead-bound (~0.12s
    # regardless of batch), so cap length directly sets step time (~23s/round)
    max_completion_len = 192
    assert dstat["max_target_tok"] + 8 <= max_completion_len, \
        f"target overflow: {dstat['max_target_tok']}"

    steps = 10 if args.smoke else args.steps
    out = Path("/tmp/rl_smoke_out" if args.smoke else args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "rl_metrics.jsonl"

    class MetricsCb(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.last_flush = 0
            self.first50 = First50Readout()

        def on_step_end(self, a, state, control, **kw):
            recs = list(STASH)
            STASH.clear()
            if not recs:
                return
            n = len(recs)
            line = dict(step=state.global_step, n=n,
                        reward=round(sum(r[2] for r in recs) / n, 4),
                        exact=round(sum(e[1] for e in recs) / n, 4))
            for fam in sorted(set(r[0] for r in recs)):
                fr = [x for x in recs if x[0] == fam]
                line[f"exact_{fam}"] = round(
                    sum(x[1] for x in fr) / len(fr), 4)
                line[f"n_{fam}"] = len(fr)
            line["elapsed_s"] = round(time.time() - self.t0, 1)
            # E1/O2 group readout (paper Fig 5; field contract in header)
            gstats = group_stats(recs)
            if gstats:
                line["psg_rate"] = round(gstats["psg_rate"], 4)
                line["full_group_rate"] = round(gstats["full_group_rate"], 4)
                line["n_groups"] = gstats["n_groups"]
            if el_sched is not None and gstats:
                # feed the admission gate: {fam: [n_groups, n_full]}
                el_sched.observe({f: (v[0], v[1]) for f, v in
                                  gstats["per_family"].items()})
                admitted = el_sched.maybe_admit()
                line.update(el_sched.state())
                if admitted:
                    print(f"ELADMIT step={state.global_step} "
                          f"tier={'+'.join(admitted)} "
                          f"(frontier full-group rate cleared "
                          f"{args.el_threshold} over K={args.el_window})",
                          flush=True)
            self.first50.add(state.global_step, recs, gstats)
            if state.global_step == 50:
                line.update(self.first50.emit())
            with open(metrics_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print("RLMETRIC " + json.dumps(line), flush=True)

        def on_train_end(self, a, state, control, **kw):
            # short runs (< 50 steps): emit the pooled readout as a final
            # event line so the E1/O2 readout is always in rl_metrics.jsonl
            if 0 < state.global_step < 50 and self.first50.g:
                rec = dict(step=state.global_step, event="first50_short")
                rec.update(self.first50.emit())
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print("RLMETRIC " + json.dumps(rec), flush=True)
            # greedy probe: does the model still emit zeta2 completions?
            import torch as _t
            model = kw.get("model")
            was_training = model.training
            model.eval()
            probes = rows[:2]
            with _t.no_grad():
                for p in probes:
                    enc = tokenizer(p["prompt"], return_tensors="pt",
                                    add_special_tokens=False).to(model.device)
                    o = model.generate(**enc, max_new_tokens=120,
                                       do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
                    txt = tokenizer.decode(o[0][enc.input_ids.shape[1]:],
                                           skip_special_tokens=True)
                    print("PROBE[" + p["family"] + "] " + repr(txt[:220]), flush=True)
            if was_training:
                model.train()

    cfg = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        steps_per_generation=args.ga,      # gen batch == 1 optim step, on-policy
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=steps,
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10 if not args.smoke else 2,
        temperature=args.temp, beta=args.beta,
        loss_type="bnpo",                  # trl 0.24 default
        scale_rewards="group",
        bf16=True, seed=3407, report_to="none",
        logging_steps=10, log_completions=True, num_completions_to_print=2,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=50, save_total_limit=2,
    )
    if args.schedule == "ordered":
        class ELGRPOTrainer(GRPOTrainer):
            """GRPOTrainer whose prompt-draw order the EL scheduler owns.
            Overrides ONLY _get_train_sampler: identical chunk/repeat
            geometry to trl's RepeatSampler (same __len__ semantics), with
            scheduler-driven unique-prompt chunks. num_workers=0 in our
            config, so each step's chunk is drawn after the previous
            step's metrics callback ran -> admission is on-policy."""
            def _get_train_sampler(self, dataset=None):
                if dataset is None:
                    dataset = self.train_dataset
                return ELSamplerStream(
                    el_sched, num_samples=len(dataset),
                    num_generations=self.num_generations,
                    unique_per_batch=(self.args.generation_batch_size
                                      // self.num_generations),
                    repeat_count=(self.num_iterations
                                  * self.args.steps_per_generation))
        trainer_cls = ELGRPOTrainer
    else:
        trainer_cls = GRPOTrainer          # banked default path, untouched
    trainer = trainer_cls(
        model=model, processing_class=tokenizer, reward_funcs=scenario_reward,
        args=cfg, train_dataset=ds,
        callbacks=[MetricsCb()])
    trainer.train()

    final = out / "final_lora"
    model.save_pretrained(str(final))
    print(f"SAVED_LORA {final}")
    if not args.smoke:
        merged = Path("/tmp/merged_rl_grpo_v1")
        model.save_pretrained_merged(str(merged), tokenizer,
                                     save_method="merged_16bit")
        print(f"SAVED_MERGED {merged}")
    print("DONE")


if __name__ == "__main__":
    main()
