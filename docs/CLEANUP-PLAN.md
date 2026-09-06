# Cleanup delivery and remaining migration

Updated 2026-09-06. The user authorized a repository rebuild supporting both research and product development. The delivered boundary is an independent shared package, documented local execution and prompt/memory contracts, core CI/checks, the V1d accounting correction, and an evidence-backed research backlog. This is no longer an initial proposal. It is also not a claim that every historical pipeline has been migrated or that a public product release exists.

The [project map](PROJECT-MAP.md) identifies current entry points, environments, inputs, outputs and checks. Detailed contracts live in [runner operations](EXPERIMENT-RUNNER.md), [prompt contracts](PROMPT-CONTRACT.md).

## Delivered scope

| Area | Implemented or recorded | Boundary |
|---|---|---|
| Shared Python package | `packages/sepalith`: standard-library runner, protocol, memory and context CLI with focused tests; `scripts/check_core.py` is the lightweight check entry point | Independent of the root training environment; existing experiment callers remain at their old paths |
| Local runner | Captured source, immutable recipes, serial dispatch, durable attempts, dependencies, drain, explicit retry, process reconciliation and hashed outputs | Linux/WSL2; one state directory coordinates its own jobs only; no live Markdown queue import, cloud dispatch or automatic checkpoint resume |
| Prompt contract | Versioned edit/evidence records, serialization, `zeta2-v1` baseline rendering and tokenizer-supplied budget checks | Evidence records do not silently enter the prompt; context selection, output parsing and editor application remain separate |
| Latent-memory contract | Payload identity, consumer compatibility and source-freshness validation | No compressor training, tensor execution, decoder injection or hybrid-cache integration |
| Product baseline | Previously tracked VS Code extension remains in place | Concurrent runtime, packaging and Zed work is excluded from this cleanup delivery |
| Scientific audit | [RESEARCH-AUDIT.md](RESEARCH-AUDIT.md) records findings, evidence limits and recovery priorities; V1d tie accounting and ML1 control specification were corrected | Correcting a conclusion is not rerunning an experiment or changing its authorization |
| Context research | [PROMPT-CONTEXT-RESEARCH.md](PROMPT-CONTEXT-RESEARCH.md) and [LATENT-WORKSPACE-MEMORY.md](LATENT-WORKSPACE-MEMORY.md) specify the user-selected learned latent-memory direction | Architecture, controls and falsifiers are proposals; no trained workspace compressor is delivered |

Shared code must not import experiment scripts. Product setup must not require NAS access, author/judge credentials or the training stack. Research may depend on shared contracts while keeping its own environments and named recipes. A candidate model becomes a product model only through a promotion record that identifies model/tokenizer, renderer, serving configuration, evaluation evidence and release decision.

## Actual layout

```text
packages/sepalith/       independent Python distribution: sepalith-core
  src/sepalith/          runner.py, protocol.py, memory.py, context_cli.py
  tests/                standard-library contract and runner tests
  examples/             tiny receipt writer for runner validation
experiments/            existing data, evaluation, training and research workflows
extensions/             previously tracked VS Code extension
scripts/                existing experiment chains and machine operations
docs/                   current contracts, operations, audit and research specifications
comms/                  live coordination and append-only history
.github/workflows/      core.yml lightweight validation
```

The root `pyproject.toml` remains the research dependency environment with `package = false`. It was not replaced by the independent `packages/sepalith/pyproject.toml`. There is no delivered `src/sepalith/` at the repository root and no bulk move into a new `pipelines/` tree. Retaining old paths protects existing imports, frozen recipes and active jobs while callers migrate deliberately.

## Operational migration still pending

The user authorized finishing the current experiment, cutting over, and resuming eligible pending experiments through the new system. That intent does not mean cutover has happened. The separate queue manager owns current execution. Consult `comms/gpu.md`, the latest board entries and [EXPERIMENT-QUEUE.md](EXPERIMENT-QUEUE.md) before any operational action; dated process IDs from the initial assessment are not current liveness evidence.

1. Inventory every active dispatcher and its automatic follow-on commands. Finish the current experiment through required evaluation, verdict recording and artifact mirroring; training completion alone may not be the drain boundary.
2. Arrange and verify the old dispatch drain with its owner. Protect quiet benchmarks and their full resource window; CPU affinity alone does not establish isolation.
3. Convert one eligible next experiment into foreground runner steps. Remove detached inner supervisors and absolute references to the mutable development checkout. Declare source includes, explicit interpreter, immutable external inputs and expected outputs.
4. Capture the reviewed working bytes, including needed uncommitted source. Inspect the new queue plan and run the receipt/failure-recovery checks in an appropriate window. A worktree from HEAD alone does not capture active uncommitted changes.
5. Activate exactly one dispatcher, run that experiment, verify its receipts, then migrate the remaining eligible queue. Keep parked decisions parked and scientific verdicts separate from execution status.
6. Preserve the old queue snapshot for rollback. Reconcile new attempts before restoring old dispatch; never run both systems concurrently.

The runner cannot see unmanaged jobs or jobs in another state directory. Its unknown-launch crash window requires operator investigation; an operator-resolution command is not implemented. External datasets, archives and backups remain operational responsibilities. See the runner guide for these limits before cutover.

## Research work still pending

The user selected learned vectors or decoder state, consumed by a specially trained edit model, as the primary workspace-context research direction. The proposed first architecture encodes modules independently, projects slots into the decoder embedding space, and optionally caches the decoder's complete hybrid prefix state. Text retrieval and summaries are controls. No claim of superiority is established by this choice.

Before training: freeze parent-workspace splits, candidate selection, encoder/decoder identities, LoRA target list, positional layout, source/payload manifests and paired budgets. Then follow the staged contract, learnability, edit-quality and economics gates in the latent-memory specification. Generic retrieval vectors or shape-compatible cross-model KV are not substitutes for trained memory.

Other recovery work follows the scientific audit: use existing artifacts first, distinguish recipe failures from mechanism failures, and repair comparison specifications before spending compute. Historical negative results and their implementations remain useful evidence. Do not delete a family merely because its adoption gate failed.

## Concurrent product work: outside this delivery

Other contributors are developing managed runtime support, packaging scripts, quant-export additions and Zed integration in the shared local checkout. Their untracked files and related product edits are not included in this cleanup push. In particular, `runtime.ts`, `scripts/packaging/`, the Zed directory, serving/quant-packaging documents and `runtime-bundles.yml` are not delivered interfaces. This cleanup retains the previously tracked VS Code baseline and does not claim a runtime extraction, packaged distribution or product release.

Those changes need their own review, native-platform and editor validation, model promotion and publication steps. The shared core package does not depend on them. Keep their local presence separate from what a fresh clone of this delivery contains.

## Further extractions

Migrate recurring callers one behavior at a time: prompt rendering, row contracts/split policies, aligned scoring, then common serving lifecycle and root resolution. Preserve literal prompt/target fixtures, IDs, split membership and denominator rules. Different historical 2%/3% holdout policies or PSM/Zeta formats must retain names and versions rather than being silently unified.

Centralize path configuration while retaining resolved absolute execution paths. Keep incompatible training stacks separate. Consolidate queue/dashboard views only after authoritative run records exist; the old Markdown queue and board remain operational sources until migration. Archive code only after checking imports, launch recipes, external runbooks and active use.

## Validation boundary

The core check command is `python3 scripts/check_core.py`; tracked product-baseline checks are listed in [PROJECT-MAP.md](PROJECT-MAP.md). Core tests use fake jobs and small local fixtures, including other-directory CLI and prompt-byte parity checks; they do not certify training, real-model resume, quiet-window isolation or a live-queue cutover. Product compilation and context fixtures do not replace real editor or native-platform checks. Some historical audit evidence lives in ignored notes or external run stores and is not distributed with a clone. No training, model evaluation, process control or benchmark was run for this documentation update.

Completion of this cleanup delivery means the maintained boundary is navigable, independently checkable and honest about its limits. Operational cutover, caller migration, research experiments and public release remain explicit next work, not hidden prerequisites for understanding the delivered code.
