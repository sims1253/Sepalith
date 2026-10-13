The saved RUN-05 evidence supports the counts below. The independent review
passed 789 assertions, rebuilt six prompts with the pinned Python renderer,
compared all 30 native tokenizations with the pinned local HF tokenizer, and
decoded all 14 completed outputs. The 14 outputs contain 304 token IDs in
total, including terminal EOS IDs. Native output counts agree with the
returned token arrays.

| Profile | Natural events | Natural responses within 5 s | Input-budget rejections | Deadline timeouts | Repeat controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA 2048 | 5 | 0 | 6, including control | 0 | 1 rejected |
| CUDA 4096 | 5 | 5 | 0 | 0 | 1 returned, excluded |
| CUDA 8192 | 5 | 5 | 0 | 0 | 1 returned, excluded |
| Notebook Vulkan 4096 | 5 | 0 | 0 | 5 | 1 returned, excluded |
| Notebook Vulkan 8192 | 5 | 0 | 0 | 5 | 1 returned, excluded |

The CUDA 2048 client made six `/tokenize` calls and no `/completion` calls.
Each prompt plus the reserved 192 output tokens exceeded 2048. Its stored
`protocolRejected=6` includes `context_budget` exceptions. These are input
rejections; there are no model outputs to classify as malformed.

CUDA natural-event request times were 186.056–363.059 ms at 4096, with a
202.630 ms median, and 162.393–341.277 ms at 8192, with a 192.129 ms median.
These match the root's saved review. The actual prompt IDs, output IDs,
decoded text, operation, and recorded plan hash agree across both CUDA
profiles. The two notebook controls agree with that corresponding CUDA
output. Plan hashes were compared across traces; this review did not
independently reconstruct their JavaScript serialization.

The notebook's only returned response per profile is the unchanged-repeat
control: 23 output IDs, a parsed no-op, and request times of 2225.323 ms at
4096 and 2324.705 ms at 8192. Each control's native `prompt_n` is **1**.
The total prompt still contains 1967 tokens; almost all prompt tokens were
cached. Neither control is a successful natural transition or fresh-prompt
prefill measurement. All ten notebook natural requests timed out after
5000.015–5001.662 ms and have no client-visible completed output.

The client source distinguishes caller cancellation from deadline aborts.
`clientAbortRequested=false` means that no newer-transition caller signal
fired. The deadline handler still calls the internal `AbortController`.
These runs use serial mode, so they do not test overlapping editor events or
stale-response publication. The trace's generic
`server_cancellation_unverified` field alone proves no server stop.

The saved notebook server logs add stronger evidence: each of five natural
tasks has a cancel record followed by a release of the same task and slot.
Server cancel-to-release intervals are 725.273, 1926.797, 625.703, 1353.711,
and 807.010 ms at 4096; they are 742.105, 1978.092, 688.043, 1445.105, and
909.667 ms at 8192. Task IDs and line numbers are in `review.json`. This
proves eventual stopping in these saved runs. It does not prove immediate
stopping, a synchronized client-to-server cancellation delay, or editor
nonpublication. Queued requests can wait for the prior task's release.

`elapsedMs` comes from local `process.hrtime.bigint()` nanoseconds divided by
1,000,000. It spans the client operation, including tokenization, completion,
and protocol/application-plan handling. Notebook measurements include the
local client and SSH tunnel. They do not measure a notebook editor or the
first visible suggestion. Server `prompt_ms` and `predicted_ms` use
milliseconds; server task-log intervals use that server's relative clock.
These timing scopes must remain separate.

Saved `/props` identifies build `b10453-3cb7ffb1a`, one slot, the requested
2048/4096/8192 allocation, and the selected theta0 Q8 path. Every server log
reports 43/43 layers offloaded. CUDA logs/maps identify the CUDA backend and
RTX 5090, with recorded `GGML_CUDA_GRAPH_OPT=0`. Notebook logs identify
Vulkan/RADV RENOIR, while the device audit records `libggml-vulkan` and a
`/dev/dri/renderD128` descriptor. The pinned supervisor's recorded model
digest is `22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559`.
This is root-reported evidence; this reviewer did not read or hash model
bytes.

All recorded server/client exits are zero. CUDA terminal receipts record
that their owned wrappers and children were absent. Notebook receipts record
zero SSH/server/client exits. These are saved observations; live PID release
remains root-owned. The notebook source and launch records specify a 65 s
client timeout, 140 s server timeout, and 180 s outer timeout. CUDA launch
records specify a 95 s server timeout. The per-request deadline is 5000 ms.
The runs finished before their process deadlines, so the evidence does not
test every timeout escalation path.

Every profile used the same 1927–1975-token prompts and a 192-token output
cap. An 8192 allocation is not evidence of an 8192-token prompt. Five natural
events per profile do not support a stable p95, a general performance ranking,
or a quality estimate. Source hashes bind the enumerated sources and staged
supervisors; they do not form a complete transitive dependency closure.

Replay this read-only review from its directory:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B review.py
```

The script uses saved traces and exact tokenizer assets. It does not launch
a server, make a network request, open final data, or load model weights.
