# C17. Qualify C19 checkpoint 100 for release

Started by the user on 2026-11-21. Work on `plan/C17-release-prep`.
The candidate is
`/mnt/e/sepalith/campaign-20260915/checkpoints/C19-grpo-exec-pilot-1/full/checkpoint-100`.
[C19's verdict](../c19/RESULTS.md) adopted it using the pre-registered exec
hold-out: all-pass 0.392 to 0.399, difference +0.0076, CI95 [0.0034, 0.0122].
That hold-out is frozen. C17 neither reads it nor runs another selection on it.
The campaign's sealed final set is a separate release gate.

Use [RULES.md](../RULES.md), [CONTEXT.md](../CONTEXT.md), and the
[operating rules](../README.md). This card authorizes preparation and CPU
checks. Publishing requires a separate user go-ahead. Sealed evaluation requires
an explicit user freeze of weights and harness. Never infer either permission
from the tier decision.

## 1. Bind the chosen quant tiers

Quantization is complete. Reuse the C09 DEV250 versus bf16 gate from
`/mnt/e/sepalith/campaign-20260915/quant-work/c19-100/gate-receipt.json`.
Its SHA256 is `b864c3a754d2537e604c74b832a9cda0da0feeb06bacd9f0cfce32c4999ae972`.
The runtime was CUDA llama.cpp commit `3cb7ffb1a`, cap 1024.
Do not rerun quant selection or replace the gate's failed verdicts.

The user's decision is in the same directory, `release-tiers-decision.json`,
dated 2026-11-21. It supersedes the generic Q8 plus smallest-tier proposal.

| Release role | Gate label | File | DEV250 |
| --- | --- | --- | --- |
| Default | Q6_K | `c19-100-imat-Q6_K.gguf` | Gate pass; score250 111 vs 112; exact -0.4 pp; nofp +0.4 pp |
| Small | Q4_K_M-mixB-e4 | `c19-100-mixb-e4-Q4_K_M.gguf` | User exception; score250 110 vs 112; exact -0.8 pp; nofp 10 to 15, +2.0 pp; KLD 0.023608 |

Pass when the decision's receipt hash matches, both files' full hashes match
the gate, Q6_K passes all existing checks, and mixB-e4 fails only
`nofp_rise_pp`, as the user accepted. Preserve that failure in release notes.
The precedent is C18b 300's `quant-work/c18b-300/release-tiers-decision.json`
and [C18b results](../c18b/RESULTS.md). Its exception does not authorize
additional C19 exceptions. [C17 evidence](../receipts/C17-quant-evidence.json)
records the decision and file identities without copying row-level results.

## 2. Establish serving parity

Compare against `sepalith.campaign_protocol`, renderer `zeta2-prm03-v1`, and
[the prompt contract](../../../PROMPT-CONTRACT.md). Use PR #89's extension
renderer, pinned at `c205e41f8f3f5ad860b62ea491315dcedb15b6c9`, through
`git show`. Do not merge or edit that branch for C17.

Run `scripts/packaging/release_parity.py` as described in the
[runbook](../c17/README.md). Pass the CPU stage only with zero byte differences
across all pinned golden contexts and C17's eight synthetic contexts, the
native request contract intact, and identical tokenizer metadata in both GGUFs.
Require BOS 0, EOS 1, native EOG 130073, vocabulary size 130560, no special-token
parsing, and exactly one manual BOS in the completion request.

The GGUF `tokenizer.chat_template` is a generic chat template, **not the editing
training template**. Applying it introduces role markers and an assistant
prefix. Record its hash and the negative comparison. Serve raw PRM03 token IDs
through `/completion`; a successful chat API response does not pass parity.

[C17's CPU receipt](../receipts/C17-serving-parity.json) checks 232 contexts
and both GGUFs. It does not establish runtime tokenizer or editor UI parity.
Before release, run PR #89's golden, completion, TRAIN-token and CPU end-to-end
checks against each shipped GGUF and the pinned release runtime. Require exact
prompt token IDs against the training tokenizer, native stop IDs 1 or 130073,
no truncation, and correct application of replacement, deletion and no-op cases.
Record commands, runtime/model hashes and editor versions. VS Code and Positron
share the renderer, but each needs an editor acceptance smoke.

Zed's integration on that branch still uses Zed's own `zeta2` renderer.
It has no demonstrated PRM03 parity. Block C19 release support in Zed until an
adapter produces identical prompts and passes the same token and edit checks.
Do not call the entire serving gate passed based on the CPU receipt.

## 3. Measure notebook latency

Use the AMD notebook, CPU and Vulkan, for each chosen tier. Do not contact the
notebook until the user provides access. Follow the runbook. Freeze
`c17/prompts.json` before measurement: eight original synthetic PRM03 contexts,
with 0, 4, 16, 32, 64, 128, 256 and 512 repeated prefix lines. No DEV, final, or
hold-out rows enter this workload. Record its SHA256.

Pin llama.cpp `3cb7ffb1a`, matching the quant gate.
Run one server at a time, context 8192, eight CPU threads, one slot, greedy
1024-token cap, seed 3407. Use `-ngl 0` for CPU and `-ngl 99` with a Vulkan-only
build for Vulkan. Confirm the backend and offload in the server log.
For each prompt, exclude one warm-up and measure five serial requests with
`cache_prompt=false`. Keep model load time outside request timing.

Report client time to first nonempty streamed content, server decode tokens/s,
and client wall time. Report nearest-rank p50 and p95 over all 40 requests per
configuration, plus every raw sample, prompt token count and stop type.
Require all four configurations to finish without errors or truncated prompts,
with finite positive timings and backend logs confirming the intended device.
This is a measurement gate; no performance target has been approved. Do not
invent a latency cutoff after seeing results. Present the four receipts to the
user for release acceptance before publishing.

## 4. Recover and freeze the sealed final protocol

The promised `docs/campaign/work/final-*` evaluator and admission guards were
not found on 2026-11-21 in this checkout, the main checkout
`/home/m0hawk/sandbox/Sepalith`, or the available worktrees' campaign directories.
The documented original planning worktree
`/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb` is absent.
`/mnt/e/sepalith/campaign-20260915/campaign-overlay/work` contains `lead`, with
no `final-*` package. These searches inspected code locations only.
Do not substitute the exec hold-out evaluator or open any sealed data to search.

This step is blocked until the original evaluator and guards are recovered.
Before any sealed run, add a reviewed addendum that cites their paths and code
hashes, names their admission commands and result schema, and copies their
existing pass criteria. Reuse the campaign evaluator, PRM03 render and greedy
cap 1024 with stop IDs 1 and 130073. If the recovered protocol disagrees, stop
for the user before execution. Do not design a replacement evaluator here.

Require a written user freeze binding the candidate weight/config/tokenizer
hashes, both GGUF hashes, evaluator and admission-guard hashes, harness commit,
runtime build, decoding settings, metrics and comparisons. Pass admission only
when the recovered guards validate that frozen manifest and their data/split
identities. The user must approve the addendum and freeze before the final set
opens once. Never print, copy or publish final rows or predictions.

After approval, run the frozen protocol once and retain its audit receipt and
aggregate metrics locally. Any guard failure, hash drift, incomplete result,
crash or missing criterion stops C17 for the user. No tuning, selection or
unapproved rerun follows final results. Qualification passes only when the
recovered protocol's pre-registered criteria pass and the user accepts the
latency report. Publishing remains a separate gate.

## Handoff

Update [status/C17.json](../status/C17.json) with each receipt and open gate.
CPU preparation needs no PC GPU job. A future sealed bf16 run, if required by
the recovered protocol, must be defined as a reviewed night-queue job after
its code is merged to main. The queue runs from
`/home/m0hawk/sandbox/Sepalith`; do not enqueue from this worktree.
Its exact command cannot be registered until the original evaluator is found.
Notebook Vulkan work runs on the notebook after access is granted.
