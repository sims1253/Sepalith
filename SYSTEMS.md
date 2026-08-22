# SYSTEMS.md — Sepalith's pipelines and how to run them elsewhere

This is the portability doc: every system in this repo, the environment
contract, directory conventions, and the repeat-elsewhere recipe. The
detailed history/rationale lives in private session logs (gitignored);
this file is the public-facing operating manual.

## Environment contract

All secrets come from the environment (never the repo). Sourcing
convention used throughout:

    eval "$(grep -E '^export (ZAI|OPENCODE|OPENROUTER|NOUS_PORTAL|NINE_ROUTER)_API_KEY=|^export (HF_TOKEN|POSTPLAN_API_KEY|NINE_ROUTER_BASE_URL)=' ~/.zshrc)"

| Variable | Used by |
|---|---|
| `ZAI_API_KEY` | glm-5.3 author/judge (z.ai) |
| `OPENCODE_API_KEY` | muse-spark + ox-alpha on opencode zen (GO + free tiers) |
| `OPENROUTER_API_KEY` | openrouter models (dots-3, gemma, ox stealth, ...) |
| `NOUS_PORTAL_API_KEY` | ox-alpha on Nous Research |
| `NINE_ROUTER_API_KEY` + `NINE_ROUTER_BASE_URL` | gpt-5.6-sol via 9router (local proxy) |
| `HF_TOKEN` | dataset push (push_cases.py / push_hf.py) |
| `POSTPLAN_API_KEY` | dashboard upload |

## Directory conventions (CHANGE THESE when porting)

The codebase uses absolute paths (a deliberate ops rule — CWD drift is a
logged scar class). The roots:

- `NAS=/mnt/h/sepalith/datasets` — all data stores:
  `cases_v1/` (wave families), `scenarios_v1/`, `synthetic_analyst_v1/`,
  `sft_v1..vN/` (assembled mixtures), `packages/` (CRAN shards),
  `normalized/` (CRAN corpus, `<pkg>/<ver>/<pkg>/...`),
  `normalized_bioc/` + `bioc_staging/` (Bioconductor),
  `stack_staging/`, `pwc_staging/` (acquisitions).
- `/tmp` — caches (token blocks, corpus copies, merged models, llama.cpp
  builds). Volatile by design; everything rebuilds.
- Run dirs: `/mnt/h/sepalith/runs/<model>_<dataset>/` (checkpoints +
  `final_lora`).

## The systems

### 1. Synthetic-data waves (author-LLM generation)
- Driver: `experiments/synthetic-data/rewrite_author_spark.py`
  (mine → author → judge subcommands; spec pools under
  `results/rewrite_author_spark/spec_pool*.jsonl`).
- Backends: `experiments/synthetic-data/cases/backends.py` (zai, opencode
  spark/GO+free via the Responses API, openrouter+model-suffix, agy CLI)
  plus driver-level backends (GO-ox, Nous-ox, 9router sol, xpreview).
  Each provider contract is documented in its class docstring — read
  them before adding a provider; several endpoints have non-obvious
  requirements (reasoning-token budgets, /v1 mounts, User-Agent headers).
- Everything runs DETACHED (`setsid nohup`) with supervisors: the
  harness reaps long-lived tracked tasks (~1h), and providers flap.
  Rows append to `<family>_<source>.jsonl` + `.done.jsonl`/`.stats.json`
  sidecars (resume keys). Every row: parent-link (`base_sample_id`
  content hash + rule@version) + model tag (purge-safe by design).
- The spec-supply is scan-bound, not quota-bound: `mine` with a NEW seed
  when pools drain (zero quota, CPU only).

### 2. Rule registry + cases library
- `experiments/synthetic-data/cases/` — declarative families
  (`specs/*.json`), corpus mining (`corpus.py`), validators
  (`validators.py`), tests (`test_cases.py`, 37 green).
- Rule registry: `cases/rules/` — detector/rewrite/verify rules with
  SELFTEST snippets; `uv run python -m cases.rules.run_rules --selftest`.
- Compounding: `cases/compound.py` (one base sample → many cases).

### 3. Assembler (mixtures)
- `experiments/post-processing/assemble_sft_v5.py --out <dir>` —
  family registration (`SCENARIO_FILES`) + caps (`FAMILY_CAPS`),
  3%-per-family package holdout, suffix-convention rendering, whole-
  mixture dedup, torn-line tolerant (wave files are appended live).
  ~15 min from cold NAS read.

### 4. Training
- SFT: `experiments/training/train_sft.py MODEL STEPS DATA [OUT]
  [RESUME]` in `.venv-sft` (unsloth LoRA; RESUME="auto" globs newest
  checkpoint). bs4×ga4, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
  True`. 6k steps ≈ 3h on a 5090 at 1B.
- RL: `experiments/training/rl_smoke.py` (trl GRPO + validator rewards;
  the proven trial config + the WSL2 lessons are in its docstring).
- From-scratch POC trainer: `experiments/training/poc_twin/` (Muon/
  Aurora/QK-Clip verified; the seed of the real pretrainer).
- Export: `experiments/training/export_gguf.py` (arch-gated --no-nextn;
  the converter must match the model family — see its history).

### 5. Eval battery
- `experiments/eval/`: `eval_scenarios.py` (validators, spawns its own
  server), `eval_ablation.py` (whole-region; `--resume` skips scored
  rows), `judge_drafting.py` (glm judge), `run_intent_suite.py`,
  `run_eval.py` (midtyping; has the zeta1/zeta2/zeta2_1 renders),
  `cache_bench*.py`, `keystroke_sim.py`.
- **SERVING**: use a CUDA-enabled build (`cmake -DGGML_CUDA=ON`) — the
  stock binaries are CPU-only and `-ngl` fails SILENTLY (74x difference,
  verified the hard way). Readiness = POST /v1/completions 200, never
  /health. Kill by tracked PID/port (`fuser -k PORT/tcp`) — never
  `pkill -f` (three self-match incidents on record).
- GPU when free, CPU when a trainer owns the card; never both.

### 6. Ideation tournament (recurring)
- `experiments/synthetic-data/ideation_tournament.py` + detached loop
  (`results/ideation_tournament/run_loop.sh`; event-paced 45-min gap,
  6 rounds/24h, `STOP_TOURNAMENT`). Bands: BUILD ≥4.4 / BANK 3.5-4.4 /
  RECYCLE (feeds next round's brief). Ox-powered triage stage per round.
  Ambient domain seeding: 26-domain rotation in every propose brief.

### 7. Acquisitions (license-gated corpus growth)
- Template: `experiments/data-mining/ingest_bioc.py` (mirror → normalize
  (air+jarl) → license ledger → staging tree). Also `ingest_cran.py`.
- LICENSING RULE (standing): permissive only (MIT/Apache/BSD/GPL-family/
  CC-BY/CC0/CC-BY-SA-marked); NC/ND/unspecified excluded AND LEDGERED.
  GPL text contains "noncommercial" — never naive-scan for NC strings.
- HF dataset sync: `experiments/post-processing/push_cases.py`
  (unified projection: corpus/ families/ mixtures/; batched directory
  commits — per-file uploads exhaust HF's 128-commits/hour).

### 8. Dashboard
- `experiments/dashboard/build_dashboard.py` → index.html (two tabs) →
  `npx postplan upload`. State in `dashboard_state.json`; live-computed
  synthetic-data landscape. Never a bare `<` in state strings; no
  inline event handlers (postplan blocks both).

## The repeat-elsewhere recipe

1. Clone the repo; create `.venv` (py3.10+) and `.venv-sft`; `pip
   install -e` equivalents per requirements (torch, unsloth, trl,
   transformers, tree-sitter, tree-sitter-r).
2. Create the NAS-equivalent data roots; populate `normalized/` from
   CRAN/Bioconductor via the ingest scripts (license ledger on).
3. Put the API keys in the environment (names in the table above).
4. Build llama.cpp WITH CUDA.
5. Smoke: `cases.test_cases` → `cases.rules.run_rules --selftest` →
   mine a small spec pool → one mock-backend wave → assemble a small
   mixture → 200-step train → export → eval_scenarios on the GGUF.
6. Then scale each system independently — they are deliberately
   loosely-coupled (files + conventions, no orchestration framework).

## Ops rules (every line is a scar)

- Absolute paths everywhere; kill by PID/port, never pkill-by-name;
  detached (setsid) supervisors for anything long-lived; sequential GPU
  (one trainer at a time; evals yield); verify trainer liveness by
  telemetry/checkpoint mtimes, never stdout tails; `timeout` wrappers
  can silently fail — verify completion yourself.
