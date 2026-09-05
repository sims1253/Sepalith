# TU2 — re-solve-vs-imitate SFT A/B (queue §3)

## TU2_PREP — teacher solve pass + arm dataset assembly (2026-09-05)

NO training fired (base pending B13's verdict + the card is queued; the arms
below are ready for the queue manager to arm). Arm (d) judge-gated is
CANCELLED per TU1's DEAD verdict. This section covers the solve pass and the
assembled a/b/c datasets; the training-arm section will be appended when the
arms run.

### Setup

- Teacher: glm-5.3 via `cases/backends.py` zai contract (thinking enabled,
  reasoning_effort low), ONE attempt per row. Prompt = EXACTLY the eval
  harness prompt: the deterministic zeta2 render from
  `assemble_sft_v2.edit_row` (byte-identical; render-parity sentinel in
  `test_teacher_resolve.py`), sole user-message content, no framing text.
  BOS discipline: dataset prompts ship BOS-free (SFT `text`-field
  convention; the tokenizer adds BOS at SFT tokenization — the RL-path
  explicit-BOS prepend is the RL harness's convention, not the SFT one).
- Solve settings = eval settings with one documented deviation:
  temperature 0 (verbatim), stop `>>>>>>> UPDATED` (verbatim), max_tokens
  1500 (DEVIATION: eval uses 640 for the local non-reasoning model; glm-5.3
  burns reasoning tokens from the same budget first — TU1's house-verified
  1500 keeps solves from being truncated by thinking, which would bias the
  gate).
- Scoring per row: `exact` (eval_scenarios convention: parse_pred
  normalization vs GT), `valid_pass` (the battery's validator,
  scenarios.validate_example rerun on the scenario row), `ast_equiv`
  (experiments/eval/ast_equiv.py, V1b N1-N7). **solved = exact OR ast_equiv**
  (queue-row TU2 (b) definition, pre-registered).
- Pacing 6s + minute-scale 429 patience (TU1-proven; 429s were rare: 6 on
  the final leg). Resumable append-only jsonl + done sidecar (3 legs: two
  harness-reaped background tasks at ~470 and ~871 rows — the final leg ran
  fully detached via setsid — plus one crash on a bare read-phase
  TimeoutError that escapes http.client un-wrapped by URLError; fixed by
  catching TimeoutError/OSError as net-class in SolveZaiBackend.complete
  and raising the per-request timeout to 300s; the in-flight row of the
  crashed leg was redone on resume).

### Sampling frame (documented; re-derivable, `teacher_resolve.build_frame`)

TU1's regeneration recipe, complement side: every row of the five family
files re-rendered through the exact `edit_row()`, held OUT of the
materialized `sft_v3/eval.jsonl` by PROMPT (the verbatim build_dataset
machinery: prompt-level exclusion + global dupe guard), seeded shuffle
(seed 3407) per family, capped. Split authority: all 307 five-family eval
prompts regenerate; 0 train-side overlap (asserted again at assembly).

| family | file rows | eval-side | dupes | train pool | capped to | in sft_v3/train.jsonl |
|---|---|---|---|---|---|---|
| rename_propagation | 5000 | 202 | 126 | 4672 | 550 | 550/550 |
| pipe_rewrite | 2592 | 18 | 0 | 2574 | 550 | 550/550 |
| format_propagation | 3924 | 67 | 0 | 3857 | 550 | 550/550 |
| doc_sync | 1087 | 15 | 3 | 1069 | 550 | 377/550 (173 post-split additions) |
| na_rm_propagation | 167 | 5 | 0 | 162 | 162 (all) | 162/162 |

Frame = 2,362 rows = 2,362 glm calls (+ ~25 retry calls; 830,051 tokens
total: 487,598 prompt + 342,453 completion; wall ~6.4h over 3 legs incl.
the crash/restarts). Package-field drift note: a handful of scenario rows
changed their `package` field after the v3 assembly (e.g. scen `devtools`
vs materialized `gh`) — prompt level is the pre-registered authority; the
sampled frame's (family,package) overlap with the eval side is 0 anyway
(logged per family in the manifest).

### Teacher solve pass (2,362/2,362 rows, 0 unattempted)

| family | n | exact | ast-only (+) | solved | solve rate | valid_pass | empty pred |
|---|---|---|---|---|---|---|---|
| rename_propagation | 550 | 162 | 13 | 175 | 31.8% | 177 | 26 |
| pipe_rewrite | 550 | 199 | 13 | 212 | 38.5% | 201 | 62 |
| format_propagation | 550 | 77 | 83 | 160 | 29.1% | 118 | 15 |
| doc_sync | 550 | 0 | 0 | 0 | **0.0%** | 8 | 4 |
| na_rm_propagation | 162 | 73 | 4 | 77 | 47.5% | 73 | 1 |
| pooled | 2362 | 511 (21.6%) | 113 (4.8%) | 624 | **26.4%** | 577 | 108 |

Readings:

- The teacher's one-attempt solve rate on the UNFRAMED zeta2 prompt is far
  below the SFT arms' eval exact rates (v8_2: rename 86.7%, pipe 88.9%).
  The SFT models were TRAINED on this format; glm-5.3 must infer the task
  from the diff + markers alone. The solve-gate therefore filters on
  task-inference ability + edit difficulty, not on target derivability.
- **doc_sync 0/550 solved** (0 exact, 0 ast, 8/550 validator-pass only).
  Verified not an instrument artifact: the teacher produces close
  paraphrases of the missing `@param` lines, but doc_sync targets are
  verbatim-pinned to the original author's wording (comment-text AST
  fallback scores paraphrases 0; the validator demands the exact line).
  Independently reproduces TU1's "0/4 exact on judge-sufficient rows" and
  gate B-β's construction verdict: the family is broken beyond context
  sufficiency — for teacher AND student.
- format is the only family where ast-only > exact-ish parity (83 ast-only
  vs 77 exact): formatting has genuine multi-rendering freedom, so arm (c)
  is format-dominated (83/113 rows).
- empty predictions (108 pooled, mostly pipe) = reasoning burned the
  token budget with no content emitted; recorded as failed solves
  (one-attempt semantics), not re-asked.

### Arm datasets (assembled, NOT trained)

**AMENDMENT 2026-09-05 (queue-mgr GO):** the PRIMARY arms are the 624-row
re-cut below (matched at |b'| = the full teacher-solve pool — caveat #1
option (ii)); the original 113-row cut remains on disk UNCHANGED as the
paired SECONDARY ((b)/(c) identical prompts). `--recut` emits the re-cut;
`--assemble` re-emits the 113 cut; both deterministic under seed 3407.

Primary re-cut (624 rows/arm; per-family sizes = the solved pool's;
doc_sync 0 naturally; a' independent seeded raw draw, b' = the full
solve pool; a' holds 412/624 teacher-unsolved rows; a'∩b' = 212 shared
solved rows):

| arm | rows | rename | pipe | format | na_rm | target source |
|---|---|---|---|---|---|---|
| (a') arm_a_raw_624 | 624 | 175 | 212 | 160 | 77 | ground truth (incl. unsolved) |
| (b') arm_b_solve_gated_624 | 624 | 175 | 212 | 160 | 77 | ground truth (solved rows only) |

Secondary cut (113 rows/arm; per-family min across arms a/b/c; doc_sync
min = 0 → dropped from ALL arms per the composition-matching rule). Seed
3407.

| arm | rows | rename | pipe | format | na_rm | target source |
|---|---|---|---|---|---|---|
| (a) arm_a_raw | 113 | 13 | 13 | 83 | 4 | ground truth |
| (b) arm_b_solve_gated | 113 | 13 | 13 | 83 | 4 | ground truth |
| (c) arm_c_teacher_target | 113 | 13 | 13 | 83 | 4 | TEACHER rendering |

- (a)/(a') = independent seeded draws from the RAW population at matched
  volume ((a) contains 82/113, (a') 412/624 rows the teacher did NOT solve
  — the filter contrast, paper-style matched volume).
- (b) and (c) hold IDENTICAL prompts (the smallest pool's seeded row set):
  they differ ONLY in target source, so (c)−(b) isolates the
  consistent-teacher mechanism with row identity controlled (verified:
  113/113 targets differ, all teacher renderings = pred + `\n>>>>>>> UPDATED`).
- (b') is EXACTLY the full 624-row teacher-solve pool (verified
  set-equality); (a')/(b') targets verified == ground truth on every row.
- Files: `results/tu2_arms/{arm_a_raw_624, arm_b_solve_gated_624,
  arm_a_raw, arm_b_solve_gated, arm_c_teacher_target}/{train,eval}.jsonl` —
  schema `{text, prompt, target, family, package_or_repo, has_types,
  tu2{...}}` (sft_v3 row schema + tu2 metadata; train_sft.py DATA-dir
  compatible, text=prompt+target verified). Every arm's eval.jsonl = the
  shared 307-row five-family sft_v3 eval split, VERBATIM (train-time eval
  loss only; the verdict eval stays eval_scenarios.py).
- Manifest: `results/tu2_arms_manifest.json` (counts, seed, composition,
  frame stats, solve rates, tokens, contamination block; `recut_624` block
  added by the amendment).

### Contamination (pre-registered protocol)

- Clean-split enforcement: prompt-level exclusion vs `sft_v3/eval.jsonl`
  (verbatim build_dataset machinery), re-asserted at assembly.
- Verbatim canary clean-vs-seen: the 307 five-family eval prompts are the
  canaries; every arm train file contains **0 of them** (asserted, logged
  per arm in the manifest — 0/113 per secondary arm, 0/624 per re-cut arm).
- Bonus checks logged: id = sha1(prompt)[:12] verified on every arm row
  (deterministic regen parity); eval.jsonl byte-identical across arms.

### Caveats / what should reshape the training-arm plan

1. **Matched volume is small (113 rows/arm)** — arm (c)'s natural size
   (AST-equiv-different solves are scarce: rename 13, pipe 13, na_rm 4).
   RESOLVED by the 2026-09-05 amendment: the primary (a')/(b') arms are the
   624-row re-cut (option (ii)); the 113-row sets stay as the paired
   secondary. Any further re-cut remains exact from
   `results/tu2_teacher_solve.jsonl` + `build_frame` (deterministic).
2. doc_sync contributes zero rows to every arm (0 solves). Any doc_sync
   rescue lives in TU3's construction lane, not TU2.
3. The solve-gate selects on the teacher's task-inference + difficulty
   (26.4% pooled), not on student-relevant derivability; arm (b)'s "pure
   filter effect" is a filter through THAT teacher's ability profile.
4. 47 rows are ast_equiv-solved but validator-failed (e.g. multi-line
   predictions on single-line families) and 8 doc_sync rows
   validator-pass without being solved — the pre-registered gate (exact OR
   ast_equiv) is what the arms use; per-row valid_pass is persisted for
   re-cuts.

### Artifacts

- `experiments/synthetic-data/teacher_resolve.py` (+ `test_teacher_resolve.py`,
  15 pure-fn tests: render-parity sentinel, payload byte-identity, scorer,
  teacher-target shape, arm membership, nested matched-subsample
  determinism, contamination canary).
- `results/tu2_teacher_solve.jsonl` (2,362 rows: raw, pred, gt, scores,
  usage, finish_reason, ts) + `.done.jsonl` sidecar + `.run.log`.
- `results/tu2_arms/` (three arm DATA dirs) + `results/tu2_arms_manifest.json`.

## TU2 training arms — TO BE APPENDED when the arms run

(one base: gate-B-β production rec b4-config; fixed recipe + seed; verdict
per queue row: WINNER-RESOLVE iff (b) or (c) beats (a) on eval_scenarios
exact (McNemar + per-family) with noopFP not worse; (c)−(b) isolates the
consistent-teacher mechanism; nothing beats (a) → raw-diff route stands.)

## TU2 VERDICT (2026-09-05, zcode-tu2-eval) — **raw-route-stands** (nothing beats (a'))

Cloud arms trained 2026-09-05 08:0x (Anyscale, ~$1.1, adapters
`scholzmx/sepalith-lora/{tu2_a_raw_624,tu2_b_solve_gated_624,tu2_c_teacher_target,tu2_b_solve_gated_113}/final_lora`;
the supplemental b@113 landed 08:44). Export + battery + readout legs below.

### Export + battery rig (CPU-only by construction)

- Merge/export: `export_gguf.py` PEFT re-merge (`MERGE_VIA_PEFT=1`,
  `CUDA_VISIBLE_DEVICES=""` — no CUDA context; the card stayed with
  zcode-o1-run's GRPO chain per gpu.md) → f16 → Q8_0 (`--no-nextn`
  qwen3_5 flow, b10453 converter/quantizer). ~1-2 min/arm.
  GGUFs: `experiments/models/tu2_{a624,b624,c113,b113}-Q8_0.gguf`.
- Battery: CPU llama-server (b10453, `-t 8 -ngl 0`) under
  `flock /tmp/b_battery.lock`; `eval_scenarios.py` on the shared 307-row
  five-family slice at the banked 150/family cap → **255 scored rows/arm,
  identical row ids across arms** (sha1(prompt)[:12]; join 255/255 on every
  pairwise comparison, **0 transport-error rows anywhere**).
- noopFP: 258 cases (204 scored no-op) per arm. b624's leg collided with a
  foreign eval server on the default port 18095 (10:41; o1-run's
  rl_o1_diverse16 server, board note 10:52) and was rerun 13:53-14:12 on
  18096 under the same lock — no rows affected, full file present.
- t/s: `llama-bench -t 8 -p 512 -n 128`, CPU, run UNDER LOAD from 2-3
  foreign CPU eval servers (18310/18311 persistent + 18095 transient;
  B13-documented convention: same rig across arms = internally comparable).

### Exact rates (255 paired rows; eval_scenarios, temp 0)

| arm | pooled exact | valid | rename | pipe | format | doc_sync | na_rm |
|---|---|---|---|---|---|---|---|
| (a') tu2_a624 raw 624 | **74.12** | 83.92 | 86.0 | 100.0 | 55.2 | 0.0 | 100.0 |
| (b') tu2_b624 solve-gated 624 | 67.06 | 78.04 | 73.3 | 88.9 | 59.7 | 0.0 | 100.0 |
| (c) tu2_c113 teacher-target 113 | 36.47 | 44.31 | 48.0 | 16.7 | 20.9 | 0.0 | 80.0 |
| (b) tu2_b113 solve-gated 113 | 46.27 | 59.61 | 42.0 | 88.9 | 50.7 | 0.0 | 100.0 |

### Pre-registered McNemar comparisons (paired by row id; b = ctl-fail&arm-win, c = ctl-win&arm-fail)

| comparison | delta pooled | b/c | exact p | per-family deltas (pp) |
|---|---|---|---|---|
| **(b') vs (a')** primary filter effect | **−7.06** | 10/28 | **0.0051** | rename −12.7 (p=6.6e-5) / pipe −11.1 (p=0.50) / **format +4.5** (p=0.58) / doc_sync 0=0 / na_rm 0 |
| (c) vs (a') | −37.65 | 3/99 | 7.0e-26 | rename −38.0 / pipe −83.3 / format −34.3 / na_rm −20 (n=5) |
| (b) vs (a') | −27.84 | 11/82 | 1.4e-14 | rename −44.0 / pipe −11.1 / format −4.5 / na_rm 0 |
| **(c) vs (b@113)** consistent-teacher mechanism, IDENTICAL prompts+steps | **−9.80** | 41/66 | **0.0199** | rename +6.0 (ns) / **pipe −72.2 (p=2.4e-4)** / **format −29.9 (p=3.3e-4)** / na_rm −20 (n=5) |

### noopFP guardrail (204 no-op cases; pre-registered: not worse than (a')'s)

| arm | FPR | vs (a') 95.59 | guardrail |
|---|---|---|---|
| (a') a624 | 95.59 | — | baseline |
| (b') b624 | 93.14 | −2.45pp | PASS (slightly better) |
| (c) c113 | 98.53 | **+2.94pp (6 cases)** | **FAIL** |
| (b) b113 | 96.57 | +0.98pp (2 cases) | marginal (≤2 cases = 0.49pp/case noise floor; flagged, not asserted) |

Absolute context: every TU2 arm is propose-always class (93-99%) vs the
production field 58.8-59.8 — structural for 300-step/624-row single-mixture
short adaptations (no restraint families in the mix); the pre-registered
guardrail is the relative one above, and it is what the verdict uses.

### t/s (llama-bench CPU t8, under load; same rig across arms)

pp512: a624 49.54 / b624 46.02 / c113 48.63 / b113 52.11 (±6% band — same
arch/quant, decode cost structurally identical). tg128: 7.84 / **1.49*** /
7.75 / 7.10 — *b624's tg128 is a load artifact (a transient third foreign
server was serving during that bench window; pp512 unaffected); per-row
scenario latencies were likewise load-noisy (mean 8.0-13.8 s/arm) and are
recorded per row in the persisted files but carry no verdict weight.

### Verdict (pre-registered rule)

**Neither (b') nor (c) beats (a') → raw-route-stands; teacher-in-the-loop
data closed on this rig.**

- (b') is significantly WORSE than (a') at matched 624-row volume: −7.06pp,
  p=0.0051. The paper's +15.4 imitate→re-solve does NOT transfer.
- (c) is catastrophically worse (−37.65pp, p=7.0e-26) AND fails the noopFP
  guardrail (+2.94pp).
- **Mechanism read ((c)−(b@113), identical prompts and steps): the
  consistent-teacher target effect is NEGATIVE** (−9.80pp, p=0.0199),
  concentrated exactly where the teacher had multi-rendering freedom: pipe
  −72.2pp, format −29.9pp (rename +6.0 ns). Training on the teacher's
  AST-equivalent-but-different renderings did not buy consistency — it
  bought noise relative to the verbatim-pinned GT targets.
- Volume/epoch-regime decomposition: b@113 (GT targets, same 113-row pool,
  same 300 steps ≈ 43 epochs) is itself −27.84pp vs (a') — MOST of the
  113-arm collapse is the small-data/high-epoch regime, with the
  teacher-target effect an additional −9.8pp on top.
- Pre-registered headroom-family expectation only half-lands: for the
  primary pair the single positive movement is format +4.5pp (ns) — right
  family, wrong magnitude — while rename (−12.7pp, p=6.6e-5) dominates the
  pooled result. Gains did NOT concentrate in headroom families strongly
  enough to offset losses elsewhere.
- **Honest note (pre-registered caveat, now empirically priced): the
  solve-gate filters through glm-5.3's 26.4% one-attempt task-inference
  profile.** The 412 teacher-unsolved rows (a') carries — rename/pipe-heavy
  — are rows the TEACHER could not infer the task for but the STUDENT
  learned from anyway; removing them removed real supervision. The gate is
  a teacher-ability filter, not a derivability filter.
- Degeneracy flag: (c)'s smooth memorization curve (train 1.293→0.0245,
  eval_loss 2.42 at ~43 epochs, no mid-run ckpt) makes the 113-regime arms
  hard to interpret at step 300 alone. **Optional follow-up recommended,
  NOT run: a save_steps-override ~100-step re-run pair (c100/b100)** to
  test whether an earlier checkpoint recovers the 113-regime — low
  priority given (b') is negative even at the clean 624-volume.

### Artifacts

- `experiments/synthetic-data/tu2_readout.py` (pure-stdlib readout; banked
  `mcnemar_exact` convention) → `results/tu2_verdict.json` +
  `results/tu2_paired_scenarios.jsonl` (255 paired rows × 4 arms, exact +
  valid + pred per arm per row, eval-v2 re-scorable).
- Full raw battery outputs: `experiments/eval/results_scenarios_tu2_*.jsonl`
  and `experiments/eval/results_noop_fp_tu2_*.jsonl` (4 arms each, incl.
  raw completions).
- Mirror: `/mnt/h/sepalith/runs/tu2_{a624,b624,c113,b113}/` (final_lora
  adapter + Q8_0 GGUF + battery/bench/export logs + results files);
  chain script `scripts/run_tu2_verdict.sh`, chain log
  `/mnt/h/sepalith/runs/tu2_verdict_chain.log` (08:32-13:51 + b624 noop
  rerun 13:53-14:12).
