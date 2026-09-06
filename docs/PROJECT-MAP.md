# Sepalith project map

Updated 2026-09-06. Sepalith develops local R edit suggestions through product integration and continuing model/data research. This map describes the delivered code boundary. It does not designate every numbered experiment script as a supported production pipeline.

## Start here

| Task | Entry point | Environment | Inputs → outputs | Check or evidence |
|---|---|---|---|---|
| Work on shared contracts or local dispatch | [`packages/sepalith`](../packages/sepalith/README.md) | Python 3.10+, standard library; separate optional package install | Versioned records/recipes → prompt snapshots, manifests, durable run attempts | Core unittest command below; [runner guide](EXPERIMENT-RUNNER.md), [prompt contract](PROMPT-CONTRACT.md) |
| Develop the VS Code/Positron integration | [`extensions/vscode-sepalith`](../extensions/vscode-sepalith/README.md) | Node/npm and editor tooling; native llama-server only for integration | Document/cursor, runtime settings, compatible GGUF → inline edit suggestions | Compile and context fixtures; actual editor acceptance is a separate check |
| Run or interpret research | [`experiments`](../experiments/README.md), [queue](EXPERIMENT-QUEUE.md) | Per-workflow research environment and external data/model stores | Frozen recipe/data/model identities → checkpoints, predictions, result summaries | Family-specific RESULTS and manifests; [research audit](RESEARCH-AUDIT.md) limits broader claims |
| Develop learned workspace memory | [latent-memory specification](LATENT-WORKSPACE-MEMORY.md) | Proposed encoder/decoder training and runner integration, not installed by core | Parent-workspace evidence → proposed trained slots and adapted decoder | Executable module-memory gate and CPU contracts in `experiments/latent_memory`; pretrained learnability run awaits a released resource window |

This cleanup delivers the core package, core CI/checks, audit and prompt/context documentation, and the V1d accounting correction. The reviewed product batch also includes managed runtimes, packaging and Zed launcher tooling. These have offline fixture checks; native-platform tests and public release remain separate.

## Shared code: independent and lightweight

The Python distribution is named `sepalith-core`, located at `packages/sepalith`; imports use `sepalith`. The root `pyproject.toml` is a separate research environment with `package = false` and heavy dependencies. Product and core development should not require syncing that training environment.

| Module | Public responsibility | Explicit limit |
|---|---|---|
| [`protocol.py`](../packages/sepalith/src/sepalith/protocol.py) | Versioned `EditContext`/`EvidenceRecord`, snapshots, baseline `zeta2-v1` rendering and tokenizer-backed size rejection | Does not collect workspace evidence, select context, truncate automatically, parse output or apply edits |
| [`evaluation.py`](../packages/sepalith/src/sepalith/evaluation.py) | Paired binary outcomes, exact McNemar and deterministic bootstrap intervals | Caller establishes row alignment; row-level intervals do not account for trajectory clustering |
| [`memory.py`](../packages/sepalith/src/sepalith/memory.py) | Latent payload identity, consumer compatibility and source-freshness validation | Does not train a compressor, load tensors, inject embeddings or restore hybrid model state |
| [`runner.py`](../packages/sepalith/src/sepalith/runner.py) | Snapshot source, enqueue immutable recipes, serial dispatch, drain, explicit retry, recover attempts and hash outputs | Linux/WSL2, its own state directory only; no cloud scheduler or automatic training-checkpoint resume |

From the repository root:

```sh
python3 scripts/check_core.py
```

No NAS, weights, credentials, GPU or model server is required. The wrapper resolves paths from its own location, so invoking it by absolute path also works outside the repository. From another directory, use an absolute `PYTHONPATH` or install `packages/sepalith` in a separate development environment for direct module access. The installed CLIs are `sepalith-run` and `sepalith-context`; without installation, use the corresponding modules with that `PYTHONPATH`.

`sepalith-context normalize --legacy example.json` validates and preserves a legacy record as versioned JSON. `sepalith-context render context.json` writes the baseline UTF-8 prompt without an added trailing newline. These operations use `context_cli.py` and do not load a model. The core suite covers other-working-directory CLI behavior and prompt-byte parity. The suite includes compatibility tests for the v5 assembler and historical paired audit. Separate V1d and product checks have their own commands.

The runner guide specifies a local state directory outside the checkout, recipes and a tiny receipt example. Queue inspection uses `--state /absolute/state/path plan`. New queues begin paused. `resume` permits dispatch; it does not launch a process. `run-next`/`run` do launch declared jobs and must wait for operational cutover. Do not start them alongside existing dispatchers.

## Product and serving boundary

The VS Code source separates context building, runtime provisioning, owned-process
termination and editor interaction. It retains its TypeScript renderer. The shared
Python renderer is used by the v5 assembler, with literal and legacy parity tests;
these implementations are not assumed interchangeable for all context choices.

```sh
npm --prefix extensions/vscode-sepalith ci
python3 scripts/check_product.py
```

Once npm dependencies are installed, the check uses fake assets and tiny child
processes. It compiles TypeScript, checks context handling, offline provisioning,
manifest compatibility, shutdown, packaging and fake quantization. It needs no
NAS, training environment, credentials, model server or GPU.

[Serving/packaging](SERVING-PACKAGING.md) documents pinned runtime preparation,
manifest generation, quant export and their real-model gates. The editor caches
validated manifests for offline restart; refresh is explicit. Python and
TypeScript consume the same manifest fixtures. Zed has [launcher/settings
guidance](../extensions/zed-sepalith/README.md); its actual editing behavior still
needs editor validation. Product CI runs offline checks, while the runtime-bundle
workflow requires manual dispatch and prepares artifacts without publishing them.

## Existing research workflows

These workflow families retain their existing tracked entry points and environments; local-only additions are not promised by this map. They are not migrated runner recipes. Versioned datasets, outputs and model paths must be selected from the relevant recipe/result record, not guessed from the largest version suffix.

| Workflow | Existing entry points | Environment and input | Output and verification |
|---|---|---|---|
| Corpus acquisition | `experiments/data-mining/ingest_cran.py`, `ingest_bioc.py`, family-specific miners | Root research environment; mirrors/source metadata, storage and network as required | Normalized corpus/staging plus license/provenance records; inspect the acquisition ledger and named recipe |
| Synthetic cases and authors | `experiments/synthetic-data/cases/`, `rewrite_author_spark.py`, other named author drivers | Root environment; corpus/spec pools; provider credentials only for author/judge calls | JSONL with provenance and resume sidecars; case contracts, validators, `cases/test_cases.py` and rule self-tests |
| Dataset assembly | `experiments/post-processing/assemble_sft_v5.py` and preserved earlier assemblers | Root environment; selected corpus/family files and split policy | Versioned train/eval mixtures; verify counts, IDs, source overlap, rendering and holdout policy before training |
| SFT | `experiments/training/train_sft.py` | Prepared `.venv-sft` or explicitly named recipe environment; model, assembled data, parameters | Checkpoints/final adapter and trainer state; preparation checks, finite-loss telemetry and held-out battery |
| RL | `experiments/training/rl_smoke.py`, `experiments/training/rl/` | Named training environment; selected SFT checkpoint, prompts and reward definitions | Checkpoints, rollouts/rewards and paired readouts; offline reward scores do not substitute for online-policy evaluation |
| From-scratch and architecture research | `experiments/training/poc_twin/`, `poc_diff/`, `poc_ddot/`, `poc_cma/`, `poc_stab/` | Family-specific config, tokenizer, objective, data and resource claim | Named checkpoints/logs and RESULTS; compare compatible objectives/budgets, retain negative results and caveats |
| Quality evaluation | `eval_scenarios.py`, `eval_noop_fp.py`, `eval_ablation.py`, intent/pairwise/episode tools under `experiments/eval/` | Evaluation environment, exact model/server/render settings and held-out rows; some judges need API credentials | Per-row predictions/scores and summaries; aligned paired comparisons and explicit denominator/failure accounting |
| Latency/speculation | `experiments/eval/spec_bench.py`, cache/keystroke/load probes | Pinned runtime/model, frozen traces and quiet resource window | Raw request/trace timings plus hardware/cache/contending-work record; a shared-machine number is not a quiet-window result |
| Export | `experiments/training/export_gguf.py` | SFT environment for model merge and compatible converter/quantizer | GGUF and the selected recipe's existing records; model/runtime compatibility and full quality battery before promotion |
| Dataset publication | `experiments/post-processing/push_cases.py`, related named push scripts | Explicit publication authorization, selected validated dataset, HF credentials | Published revision and transfer records; not an automatic consequence of local assembly |
| Dashboard | `experiments/dashboard/` current builder/loop and state files | Existing operational environment; queue/result sources, upload credentials when publishing | Generated views and logs; dashboard status is a view, not scientific verdict authority |

`SYSTEMS.md` records workstation environment variables, storage roots and legacy operating practices. It is operational context, not a dependency requirement for core/product checks. Many scripts still resolve `/mnt/h/sepalith` datasets/runs and workstation paths. NAS data, model weights, caches and large prediction streams are not recreated by cloning the repository.

## Current operations versus historical evidence

- [EXPERIMENT-QUEUE.md](EXPERIMENT-QUEUE.md) contains pending/current research status. `comms/gpu.md` records resource claims; `comms/board.md` records decisions and handoffs. Read their latest entries before resource use. This map deliberately carries no live PID or ETA.
- `scripts/run_*` includes old and current chain wrappers. A filename is not evidence that a script is safe to dispatch now. Inspect automatic follow-on commands and current ownership before migration.
- Family `RESULTS.md` files and associated receipts are verdict evidence. Old plans, incomplete tables and queue summaries may lag them. [RESEARCH-AUDIT.md](RESEARCH-AUDIT.md) records comparison and inference problems that matter to future work.
- Some audit sources were inspected in local ignored research notes or external run stores. They are not all distributed in a clone. Treat absent source artifacts as a reproducibility limitation; the tracked audit does not imply that every cited historical file is published.
- Named assembler versions, tokenizers and split policies can define a historical scientific comparison. Keep their identities even when extracting shared implementation.
- New runner success means commands and declared artifact checks succeeded. It is not a scientific adoption verdict. The new runner has not imported the Markdown queue or taken over existing supervisors.

## Remaining work

[Cleanup delivery and migration](CLEANUP-PLAN.md) separates four next steps: operational cutover, migration of recurring callers, latent-memory experiments, and product release validation/publication. None is silently completed by the existence of a package or specification. The current usable boundary is shared tests/contracts, a documented local runner for explicit recipes, existing research entry points, and reviewed editor/runtime packaging tools. Release validation and other concurrent research work remain separate.

[Reports and raw evidence](ARTIFACTS.md) explains which records belong in Git
and how to retrieve verified raw artifacts from the private archive.
