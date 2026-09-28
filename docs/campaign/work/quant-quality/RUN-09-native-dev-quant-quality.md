# RUN-09 native DEV quantization quality preparation

**Status:** CPU-only source preparation complete; root launch and quality
measurement pending. This v2 packet does not select a quantization artifact,
start a server, load a model, read sealed final data, or change campaign
state. The v1 source and receipt are preserved beside this packet.

The client is
[`run09_native_dev_quality.py`](run09_native_dev_quality.py). It is a native
llama.cpp b10453 client and scorer for the exact 75-row DAT-07 DEV panel. The
file name `DAT-07-final-evaluator-cases.jsonl` is historical; the panel rows
are `split=dev` and the panel SHA is checked before any server request.

## Frozen identities

| Item | Path | Identity |
|---|---|---|
| DEV panel | `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl` | 75 rows, SHA256 `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21` |
| HF tokenizer | `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain` | revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, `tokenizer.json` SHA256 `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, `tokenizer_config.json` SHA256 `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b` |
| PRM-03 protocol | `EXEC/packages/sepalith/src/sepalith/campaign_protocol.py` | SHA256 `5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156` |
| Reviewed scorer | `EXEC/experiments/training/campaign_eval.py` | SHA256 `7064385d63e3900b87241da75525897acb6b55d9ce1a023689c9addfbaacfc9f` |
| Reviewed native client | `EXEC/extensions/vscode-sepalith/src/campaign_client.ts` | SHA256 `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333` |

The live loader uses the local frozen HF tokenizer only; no model weights are
opened by this client. It checks vocab size 130560, BOS 0, EOS/PAD 1, and EOS
text `</s>`, then encodes all rendered prompts with
`add_special_tokens=false, split_special_tokens=true`. Native `/tokenize`
must return exactly the same IDs without BOS.

The permitted static preparation passed all 75 rows. The ordered case-ID
digest is `6e0c4c5bd7de4f0d9aa6eab602796c13da110f56edab6b0dd8a9e04fb66ee6bb`;
the actual prompt-with-BOS band is 132–2619 tokens, so every row fits
4096 context with the full 512-token output budget.

## Native contract

Every row is rendered from `PromptContext` with `render_prompt(context)`;
operation, family and `region_new` remain out-of-band scoring labels. The
client sends:

```json
{"content":"<rendered PRM-03 prompt>","add_special":false,"parse_special":false,"with_pieces":false}
```

It prepends exactly integer BOS `0` to the native IDs and sends this request:

```json
{"prompt":[0,"<native prompt IDs>"],"n_predict":512,"temperature":0,"stream":true,"cache_prompt":false,"return_tokens":true}
```

The quality route uses SSE so it can retain TTFT and every returned ID. b10453
partial frames carry generated IDs, including EOS; the terminal stop frame is
required to carry an empty `tokens` array. A nonempty terminal frame is
recorded as possible duplication and invalidates the case. There is no text
stop list, so the protocol terminal is judged from returned IDs and the exact
PRM-03 parser. The case deadline is 120 seconds for combined tokenize and
completion, and the global deadline is at most 7200 seconds. This is not the
5-second runtime probe.

The server `/props` response must report exactly `n_ctx=4096`. If
`tokens_evaluated` is present, it must equal the full integer prompt length,
including manual BOS. The generated sequence must end in canonical EOS 1,
contain no native CONTROL IDs before EOS, and contain at most 512 IDs including
EOS. Native EOG 130073 is retained as a noncanonical failure.

## Quality output and denominators

The output is one JSON receipt at a fresh native-ext4 path. The filesystem
preflight uses `/usr/bin/findmnt -T <output-parent> -n -o FSTYPE`; this avoids
GNU `stat`'s `ext2/ext3` compatibility name for ext4. An initial `running`
receipt is fsync'ed before static checks; it is atomically rewritten after
static preflight, server preflight, and every case. A transport, SSE, token
identity, parser, EOS, text-parity or cap failure is retained immediately with
its error and does not enter successful quality counts. No case is retried. If
the global deadline expires, remaining IDs are listed as unattempted.

If the HTTP stream has started and then times out or fails at the transport
layer, the receipt preserves the text, token IDs, timing fields and parser
frame counters observed so far under `partial_stream`. Such a row has
`response_received=true` and `response_complete=false`, is counted under
`partial_responses` and `transport_failures`, and cannot contribute a quality
metric.

Each case records its ID, family, package, operation label out of band,
prompt/target hashes, full HF and native prompt IDs, request payloads, raw
server text, returned IDs, decoded body text, server timing fields, TTFT,
wall timing, EOS and inclusive-cap status, parser status, and quality flags.
The quality flags match the SFT evaluator's exact/protocol/no-op axes:

* `protocol_valid` requires the pinned `valid_generation_tokens` guard,
  native `stop_type=eos`, exact equality between wire text and HF decoding of
  the non-EOS body IDs, an exact parser-accepted response, and clean SSE
  framing. The only EOS normalization is excluding the terminal EOS ID from
  the HF body decode; wire text is never stripped or rewritten.
* `exact_edit` compares a valid parsed replacement (or delete) with the
  out-of-band `region_new` label.
* `strict_noop_correct` and `noop_false_positive` are counted only for the
  32 labeled no-op rows.
* `edit_exact` is counted only for the 43 labeled edit rows. `exact_edit`
  remains an all-row exact-region flag, so the receipt reports separate
  `edit_cases=43`, `strict_noop_cases=32`, `exact_region_rows`, and
  `edit_exact_rows` denominators.
* `cap_hit` records a length/limit stop or overflow; EOS at ID 512 remains
  within the inclusive cap policy.

Family counts and panel denominators are retained separately. A terminal
`complete` status means that all 75 rows were attempted and is independent of
`all_protocol_accepted`; protocol errors and cap failures remain visible in
their case records and denominators. Transport failures are counted separately
from mechanical quality failures. Native serving does not expose logits, so prompt/target NLL is explicitly recorded as
`unavailable_native_no_logits`; this packet makes no NLL equivalence claim.
Run each F16, Q8 or mixed-K server separately with a distinct
`--model-label` and root-supplied `--model-provenance`. The client never picks
an artifact or infers provenance from a filename.

## Root launch recipe

Root owns the server, model URL/path and provenance. After the root wrapper has
started the matched b10453 server on a loopback port with `-c 4096`, use a
fresh output path on native ext4:

```sh
EXEC=/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912
PLAN=/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb
PYTHON=/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python
PANEL=/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl
TOKENIZER=/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain
MODEL_LABEL=<F16-or-Q8-or-mixed-K>
MODEL_URL=<root-wrapper-model-url-or-local-provenance>
MODEL_PROVENANCE=<root-wrapper-manifest-or-quantization-provenance>
OUTPUT=<fresh-native-ext4-output.json>
/usr/bin/timeout --signal=TERM --kill-after=60s 7260s \
  env PYTHONPATH="$EXEC/packages/sepalith/src:$EXEC/experiments/training:$PLAN/docs/campaign/work/quant-quality" \
  "$PYTHON" "$PLAN/docs/campaign/work/quant-quality/run09_native_dev_quality.py" \
  --url http://127.0.0.1:<root-port> \
  --panel "$PANEL" --tokenizer-dir "$TOKENIZER" --execution-root "$EXEC" \
  --output "$OUTPUT" --model-label "$MODEL_LABEL" --model-url "$MODEL_URL" \
  --model-provenance "$MODEL_PROVENANCE" --case-deadline-seconds 120 \
  --deadline-seconds 7200 --reserve-seconds 60
```

The outer timeout is 60 seconds beyond the client global deadline for receipt
flush and server-side cleanup. Root may run this client locally through an SSH
loopback tunnel; the tokenizer and static sources are local to the client host,
while the server URL remains the loopback tunnel origin. Run one quantization
at a time and retain each output hash. A complete quality result requires
`evaluation_complete=true` and root review of `all_protocol_accepted`,
exact/no-op/family metrics, and any protocol or transport failures; the
script's terminal `complete` status alone does not select a release artifact.

## Offline verification and remaining gates

The focused CPU test command is:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -m unittest -v test_run09_native_dev_quality.py
```

It passed 15 tests covering exact payload flags, outer-wrapper provenance,
partial SSE EOS handling, malformed SSE, final-token duplication,
noncanonical EOG, cap-without-EOS, parser failure, typed timeout/no-retry,
real temporary-path ext4 preflight, negative/out-of-vocabulary IDs, whole-SSE
deadline cancellation, wire/HF text mismatch, tokenizer identity mismatch,
partial-stream evidence preservation, server context identity, and denominator
separation. `py_compile`, CLI help
and `git diff --check` also pass. No server, model, CUDA, SSH, cloud or
sealed-final access occurred during preparation.

Remaining live gates are root-owned: server/model provenance, model-to-server
identity, native tokenizer parity for all 75 rows, full response persistence,
and the F16/Q8/mixed-K quality comparison. The existing SFT NLL path is not
available through native llama.cpp and remains explicitly unresolved.
