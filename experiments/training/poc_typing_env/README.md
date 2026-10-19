# poc_typing_env — typing-RL environment with a mixed judge cascade

PoC for the user idea (2026-10-16): an RL environment where the agent
**types code** (chunk by chunk, in generation order) and the reward comes
from a **cascade** — cheap open decision models for dense process signals,
an LLM judge (glm-5.3, the project's validated judge) for rubric signals,
and **execution anchors** (tests + benchmarks) that keep the whole thing
honest.

Status: working PoC. Results in [RESULTS.md](RESULTS.md) +
[dashboard.html](dashboard.html). Tests:
`python3 -m unittest test_typing_env -v` (stdlib-only; R integration
auto-skips if Rscript is absent).

## The reward cascade

| Tier | Cost/latency | Signal | When |
|---|---|---|---|
| Deterministic | ~0 | bracket balance, repeated-line detection | every typed line |
| Decision model (`vllm-sr/d3-edge`, 0.6B) | ~0.1-1 s CPU | `on_track/off_track/degenerate`, syntax plausible, risky side effects | every 3 typed lines (cap 12/episode) |
| Execution (Rscript) | ~1-4 s | test pass fraction; benchmark vs naive reference (same process, adaptive timer, median of 3) | terminal |
| Decision model (terminal) | ~0.3-1 s CPU | quality score (unusable→elegant), looks_correct, **hacky** (reward-hacking detector) | terminal |
| glm-5.3-flash rubric | ~2-5 s | same JSON rubric as the big judge, thinking off — the cheap LLM tier + ensemble second opinion | terminal, every rollout |
| glm-5.3 judge | ~5-15 s | strict-JSON rubric: elegance/readability/r_idiom/reward_hacking/confidence + notes | terminal, **routed by the tiers below it** |

Routing (`cascade_decision`): `dead` (exec failed AND d3 says unusable/poor
→ no LLM judge), `clean` (exec passed AND d3 confident → flash rubric only),
`run` (the ambiguous band → both judges; their agreement is recorded).

The user-linked collections, mapped to slots in this design:

- **vllm-sr Decision 3.0** (`d3-edge`…`d3` 26B): the decision tier used
  here. One `system_one` call = several choice/yes-no/score questions,
  each answered with a probability distribution, no text generation.
- **Cloudflare CLEF** (27B / flash 9B): decision models — same slot, needs
  GPU-class serving; not run in this PoC (CPU-only box).
- **convaiinnovations/laya** (MoE): generative cheap-judge slot (a judge
  that can argue, cheaper than glm-5.3).
- **LiquidAI liquid-nanos**: mixed family — includes the `d1-omni-600M`
  decision model (appears on the same decision board as d3) as an
  alternate decision backend; the rest are generative.

All of them sit behind one interface (`d3_client.BaseJudge.ask(state,
questions) -> {q: {label, probs}}`), so swapping backends is a `--d3`
flag away, and every emitted row records which backend produced it.

## Why this composition (repo lessons applied)

The coding-simulator work (`../coding_simulator/design.md`) measured that
graded judge scores wreck inter-judge agreement and invite plausible-drift
reward hacking. This PoC therefore:

1. **Anchors on execution**: correctness (0.55) + performance (0.20)
   dominate the demo scalar; judge+decision tiers are 0.25 combined.
2. **Logs every component separately** — the scalar is one demo
   aggregation, the component rows are the actual contract.
3. **Detects hacking at both judge tiers**: `hacky` (d3) and
   `reward_hacking` (glm rubric) feed the `guard` component; a gated
   broken-code multiplier (×0.3 below 0.5 correctness) means no style
   reward for broken code.
4. **Judges never gate correctness** — tests do. The cascade only decides
   whether the *expensive* judge runs at all.

Known PoC caveat, logged not hidden: glm-5.3 is BOTH the typing policy and
the expensive judge (user directive — one model to run). The judge rubric
asks for belief-of-correctness but that dimension is reported, not
rewarded; production should use the panel pattern (`../coding_simulator/
panel_judge.py`) or at least a different judge than the policy.

## The RL loop demonstrated

No GPU trainer on this box, so `run_poc.py` closes the loop the way GRPO
would, but with the reward acting through the prompt:

```
round 0: 3 rollouts (temp 0.2)  ──score──▶  best by scalar_total
                                              │ feedback = best code +
                                              ▼ component breakdown +
round 1: 3 rollouts (temp 0.8)  ◀── mechanically derived directives
round 2: 3 rollouts (temp 0.8)  ...
```

That exercises the full reward machinery (typing process rewards, exec
anchors, decision tier, cascade gating, per-component logging) end to end
and produces a visible reward trajectory per task. The gym-shaped env API
(`reset/step/finalize`, stdlib-only by repo convention) is what a real
trainer plugs into; `sepalith.training.rl` (GRPO/GSPO driver) and the
Prime Intellect `verifiers` env/rubric pattern are the intended homes —
see `../rl/README.md`.

## Tasks (`tasks.py`)

Three base-R tasks chosen so components separate (validated against
hand-written good/naive/broken references before any model call):

| task | correctness traps | perf signal |
|---|---|---|
| `top_k_words` | tie-break by first appearance, empty input, punctuation | weak (same algorithm class) |
| `rolling_mean` | **NA-window poison trap** (naive cumsum fails!), w>len, w=1 | ~150× (vectorized vs loop) |
| `grouped_total` | NA/empty-group drops, first-appearance order | ~8× (rowsum vs boolean rescans) |

The rolling_mean NA trap is the demo's showpiece: the obvious fast
solution (cumsum) passes every naive test and still fails `na_window` —
exactly the class of bug that execution anchors catch and LLM judges
wave through. (It caught two bugs in this PoC's own reference
implementations during validation.)

## Wiki guidance applied (~/nas/wiki, 2026-10-16)

The wiki was consulted in two rounds: a glm-5.3-flash lookup pass (run in
the session, not as repo tooling), then OpenCode session subagents doing a
thorougher two-lens sweep (verification/reward design; decision-model
economics). Load-bearing claims were checked against the source pages.

**Implemented (round 1):**

| Wiki page(s) | Directive | Implementation |
|---|---|---|
| `reward-hacking-equilibrium` | scalar = K=1, maximize information loss; keep signals multi-dimensional; audit implicit weights | every component logged separately; scalar labeled demo aggregation; LOC audit in RESULTS.md |
| `model-routing` | cheap tier triggers the expensive one only on ambiguity; route at boundaries | `cascade_decision`: dead/clean/run routing of the glm-5.3 rubric |
| `multi-model-review` | 2+ models = the reliable signal | glm-5.3-flash second opinion on every rollout; `judge_agreement` recorded |
| `environment-free-verification`, `uninfluenceable-verification` | passing tests beat any judgment; harness outside agent control; real-scale data | exec anchors dominate weights; harness protocol agent can't touch; 200k-1M element benchmarks |
| `validation-agent` | validate the verifier (oracle passes, no-op fails) | tasks validated against good/naive/broken references before model calls |
| `task-specific-edge-models`, `jev-typesafe` | calibrated probs ≠ accuracy — validate against execution | d3 quality gates the cascade; probs + expectation kept alongside labels |

**Implemented (round 2 — session-subagent findings):**

| Wiki page(s) | Directive | Implementation |
|---|---|---|
| `reward-hacking` §SpecBench | "test composition is a reward-hacking detector — hacks don't compose" | held-out composition suites per task (never agent-visible); `rh_gap` = visible − heldout pass rate logged per episode. Validated: a visible-test hardcoder scores 1.00 visible / 0.00 heldout |
| `aide2-rsi` | public/private score split — survival decided on a score the agent never sees | refinement-loop round winners selected on held-out correctness first, then scalar; feedback contains only visible-test info |
| `entities/seta`, `partial-credit-agent-evals` | fraction passed + bonus for full pass keeps reward variance near the frontier | `exec_term = frac + 0.2·(full visible pass)` (max 1.2) |
| `reward-hacking` §Prime Agent | penalize proxy-only gains in refinement loops | RESULTS.md flags rounds whose scalar rose while execution didn't |
| `advantage-collapse`, `agent-specific-difficulty` | admit tasks to RL only in the ~[0.25, 0.75] observed pass band; retire as they saturate | RESULTS.md difficulty-admission table computed from the run |

**Queued (from both rounds, not yet built):** logprob-distribution scoring instead of sampled verdicts (`llm-as-judge`); partitioned rubric groups + explicit conjunctions, never holistic scores (`reward-hacking`, `code-slop-metrics`); geometric-median aggregation with ≥3 judges and 2-of-N escalation (`entities/ropoll`); judgment/decision separation formalized — decision models veto/escalate but never override test outcomes (`trust-layer-agent-eval`); threshold-veto semantics pinned to raw probabilities of a hash-pinned build (`calibrated-relevance-filtering`, `entities/laya`); causal-evidence judging — feed the judge execution deltas (`heavy-duty-agent-verification`); CRAN-trained human-vs-model discriminator as the elegance tier (`rlvr-adversarial-discriminator`); brown-field minimal-diff task family (`minimal-editing`, `code-mergeability`); offline/batch loop with infrequent syncs — semi-online DPO on cascade-scored pairs as a GRPO alternative (`oapl`, `distributional-post-training`); reward-channel corruption audits before blaming the policy (`entities/kat-coder-v2.5`); periodic hackability census with parameterized fixtures (`reward-hacking`, `entities/seta`).

Not yet applied from round 1: causal-evidence selection beyond pass/fail, parameterized anti-hardcoding cases, Goodhart→Campbell monitoring.

## Running it

```bash
# ZAI_API_KEY must be in env (interactive shell has it); Rscript needed.
cd experiments/training/poc_typing_env
python3 -m unittest test_typing_env -v          # offline, stdlib-only

# decision-model runtime (once; .venv/ is gitignored):
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python "transformers==5.17.0" accelerate safetensors pillow
.venv/bin/python d3_client.py                    # downloads d3-edge, smoke check

# the PoC loop (d3 loads lazily from .venv when available):
.venv/bin/python run_poc.py --tasks all --rounds 3 --rollouts 3
#   --d3 off       decision tier disabled
#   --d3 stub      deterministic stub (offline demo, marked in every row)
#   --glm-off      no network at all (mock typer)
#   --quick        1 task, 1 round, 1 rollout — end-to-end smoke
```

Wiring the decision tier into a run without the venv: `run_poc.py` falls
back to `judge=None` (process/terminal decision components disabled, rows
say `backend: none`) — execution anchors and the glm judge still work.

## Pilot observations (v1 run, interrupted)

The first full grid (pre flash-tier, pre routing) crashed 17/27 rollouts in
on a z.ai mid-stream stall — which motivated two runner properties now in
place: per-rollout flush + resume, and honest finalization of partial
prefixes when a stream dies mid-typing (no silent regeneration over text
the env already scored). Before the crash it already showed:

- **The feedback loop moves the reward**: `rolling_mean` round means
  +0.51 → +1.08 (r0 had a 0.4-correctness rollout falling into the NA
  trap; by r1 every rollout was 1.0-correct and 100×+ faster than
  baseline).
- **A real failure mode the reward catches**: `top_k_words` solutions were
  correct but consistently *slower than the naive reference* (perf −1.0
  = ≥16× slower on several rollouts) — the reward said so every round;
  the feedback directive ("vectorize harder") did NOT fix it. Perf under
  adversarial temptation is exactly what a scalar-free component log is
  for.

## Sandbox & measurement caveats

- Candidate code runs as `nice -n 10 timeout … Rscript --vanilla`. That is
  PoC-grade isolation: fine for own-machine demo runs with self-authored
  code, NOT for untrusted rollouts at scale — production wants
  nsjail/firejail/docker with cgroup caps. The d3 `risky` question at
  least flags `system()`/eval/network attempts during typing.
- Benchmarks run on a shared machine (repo resource policy): naive and
  candidate are benched in the SAME R process on the SAME data with an
  adaptive timer (median of 3 samples), so the ratio is meaningful even
  under contention; absolute seconds are pessimistic.
- The decision tier runs capped at 8 torch threads (resource policy).

## Cost anatomy (from the recorded run)

See RESULTS.md `glm usage` and the per-row `timings`: typing generation
dominates wall time; the decision tier costs ~1 s CPU per check pack;
the cascade skips the glm judge for clearly-dead rollouts. That's the
economic argument for the cascade shape: the cheap tier both shapes
mid-typing behavior AND amortizes the expensive judge.
