# Runtime worker receipt

## RUN-01 — serving architecture and host placement

**STATUS:** partial

**OWNER:** worker-runtime

**STARTED:** unknown — assignment timestamp was not captured by the worker shell

**ENDED:** 2026-09-12T03:45:29+02:00

**DEPENDENCIES_CHECKED:** PRE-01 (docs/campaign/receipts/PRE-01-source-snapshot.json) and PRE-05 (docs/campaign/receipts/PRE-05-environment-lock.json). The requested briefs were run from the plan root:

~~~
python3 docs/campaign/campaign.py brief RUN-01
python3 docs/campaign/campaign.py brief PRE-08
python3 docs/campaign/work/runtime/fixture_check.py
python3 docs/campaign/work/runtime/gguf_tensor_compare.py
~~~

**ACTION:** Inspected the owner extension source and its Sidecar.start path. Pinned the local CPU fallback to llama.cpp b10453 and wrote an explicit host/owner profile. The full JSON receipt is [RUN-01-serving-architecture.json](../../receipts/RUN-01-serving-architecture.json).

**INTENDED PLACEMENT:**

- The VS Code extension host owns one child llama-server, the model file, and 127.0.0.1:18099. In a Remote-SSH workspace this is the remote workspace host; in a local workspace it is the local Linux host.
- The editor or notebook UI receives InlineCompletionItem results through the VS Code extension protocol. It does not own the inference port. No server-to-notebook tunnel is assumed because no measured need exists.
- The CPU fallback stays on the same extension host. Its owner is explicit: the extension child when managed, or the terminal/launcher owner when run manually.
- The CUDA training host and PID 1700431 remain lead-owned. This worker did not start a server or touch that process. CUDA serving waits for lead scheduling and cannot overlap training.
- An AMD notebook/Vulkan host is only a possible separate scout. It has no serving role until its identity is supplied and a host lease is recorded.

**OBSERVED PLACEMENT:** The current shell has no /home/m0hawk/.ssh/config or readable SSH alias; known_hosts entries are hashed. No VS Code/editor or extension-host process was discoverable, and no listener answered on 127.0.0.1:18099 or the checked 180xx ports. PID 1700431 is the active experiment/CUDA owner, not evidence of an editor session. The intended workspace placement is therefore recorded separately from observed editor placement. The authorized SSH alias/hostname and VS Code Running Extensions host identity remain missing.

**PINNED PROFILE:**

~~~
server: /home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server
server sha256: 123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804
server version: 0.1.0-dev (build 10453, commit 3cb7ffb1a), GNU 11.4.0, Linux x86_64
model: /home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf
model sha256: e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d
model bytes: 2012011904
host: 127.0.0.1
port: 18099
parallel: 1
threads: 8
context: 8192
temperature: 0
gpu layers: 0
alias: sepalith
~~~

The owner extension source is extensions/vscode-sepalith/src/extension.ts, SHA256 368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285. It spawns the child in array form without a shell:

~~~
-m /home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf
--alias sepalith --temp 0 --port 18099 --host 127.0.0.1
-c 8192 --parallel 1 -t 8 -ngl 0
~~~

Sidecar.start first checks /health, then requires a real one-token POST /v1/completions readiness response. A healthy answering port is marked external; the extension never spawns or kills that server. Teardown is tracked-child-only. This prevents two owners from contending for 18099.

The historical GPU quality archive used the same build/commit but a different runtime asset, SHA256 e42d5362c31f9149e36a94677e46c31b7b56ee0e4128d67e6a32383d4cc1c0ee, with -ngl 99 -lv 4 --seed 20260905 on port 18475. That is an archived identity, not the CPU fallback or a current launch request.

**FEASIBILITY NOTES:**

- ngram-simple is the lossless model-free baseline with near-1x prior. ngram-mod is the highest-value separate CPU scout; its proposed flags are --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64. It needs a fresh paired exact-output and latency gate.
- Released MiniCPM5-2B-DSpark has five draft layers, 323,776,001 parameters, seven draft tokens, target taps [1, 10, 20, 30, 39], and published acceptance 5.5174 at temperature zero. The released draft is paired to the released post-trained target; that result does not transfer to Midtrain/SFT/RL weights.
- A lighter/adapted draft is possible only after tokenizer/config/tap compatibility and a TRAIN-only target cache. The local b2_qwen35_08b-Q8_0.gguf is an 811,842,944-byte comparator, not a validated draft pair.
- The pinned Midtrain config and staged weights are ordinary LlamaForCausalLM metadata. The 381-tensor header has no mtp, nextn, dspark, draft, or speculator matches. This is metadata/header evidence, not tensor-proof native MTP evidence.

**ACCEPTANCE:** The child path, b10453 identity, one-slot localhost controls, ownership rules, and host responsibilities are reproducible. The remote/AMD target identity remains unresolved, so this receipt is partial until the authorized SSH alias/hostname and VS Code Running Extensions host identity are available.

**UNRESOLVED:** Remote host identity; no live completion smoke because the packet forbids server starts and the CUDA owner remains active.

**NEXT:** Lead supplies the target-host identity if needed, or performs one idle-window CPU completion smoke with the pinned profile. Lead retains promotion and CUDA scheduling.

## PRE-08 — b4 fallback and historical ledger

**STATUS:** verified

**OWNER:** worker-runtime

**STARTED:** unknown — assignment timestamp was not captured by the worker shell

**ENDED:** 2026-09-12T03:45:29+02:00

**DEPENDENCIES_CHECKED:** PRE-01 and PRE-03 (docs/campaign/receipts/PRE-03-storage-headroom.json). No model was regenerated or copied.

**ACTION:** Resolved the banked b4 artifact by matching the archived quality recipe, preserved the distinct b4 artifact, matched renderer/stop/parser source identities, retained executable fixture checks, and recorded rollback branches. The full JSON receipt is [PRE-08-fallback-ledger.json](../../receipts/PRE-08-fallback-ledger.json).

**ARTIFACT IDENTITY:**

The banked quality comparator is packaging_b4-Q8_0.gguf, not b4_qwen35_2b-Q8_0.gguf:

~~~
packaging_b4-Q8_0.gguf
  bytes: 2012011904
  sha256: e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d
  sidecar: /home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf.json

b4_qwen35_2b-Q8_0.gguf
  bytes: 2012011904
  sha256: 82012be8f9a5c776126919ea01cb5c12e7e1a5823cd17e061594a5291f26c8ea
  sidecar: absent
~~~

The archived b4-export-paired-quality-20260908 recipe lists the first hash as its Q8 input. The second hash is PRE-01 protected but does not appear in that quality recipe. Both files remain preserved.

The Sept 10 S2 Q8 file at /mnt/h/sepalith/runs/s2-quality-20260910/assets/Q8_0.gguf has the same full-file SHA256 as packaging_b4 and its evaluation records 242/307 exact, 269/307 valid, and 136/204 strict scored no-op false suggestions. The Sept 10 server log records the historical b10453 GPU runtime, -ngl 99, -lv 4, eight threads, one slot, context 8192, seed 20260905, and port 18478. The Sept 11 O2 receipt reports the same bank value (242/307), but does not independently hash or serve a bank file. Its cases.jsonl and selection.json are byte-identical to the Sept 10 files (SHA256 447543b6b074f38667c8fd36736c2c729faad1c1ff066129ff18fe19ae4edf78 and 419e3cd0d1d368c1153d580925c89ce1a9c74a5768cd36680bbbd18dc329871a), so the Sept 11 bank value is identity-consistent with the direct Sept 10 artifact without claiming a second bank hash. The Sept 11 raw proposal counts are O2 192/262 and bank 194/262 generated legacy points; 58 mixed mid-typing points are excluded from the strict no-op denominator, so the raw 194 count must not be treated as a 204-denominator rate. The Sept 10 strict bank value remains 136/204.

The retained [gguf_tensor_compare.py](gguf_tensor_compare.py) command streams each tensor through local gguf-py in 8 MiB chunks. It compared all 320 tensors and found 320 payload-hash matches, with no differences. Both files are 2,012,011,904 bytes and share qwen35/file_type 7 and data offset 10,961,024; their GGUF names differ (Pft1_B4_Merged versus Merged_B4_Qwen35_2B). The tensor payloads are therefore identical while full-file metadata differs. The packaging file remains the bank identity because its complete hash is the direct S2 bank hash.

**MATCHED CONTRACT:** run_eval.render_zeta2 (source SHA256 7fc6d4d796856ef3697365a462a8a1f5b0a9876d1f2e55cb88325a6ff7ef493d) is the renderer used by eval_scenarios (SHA256 da81802f553e91182ca7dbece30614ad5b12479f3f4e3f5a856a16dcd23fefc3) and the assemble_sft_v2 training-time path (SHA256 dd519c61f0684040251f5f886c6f6307a6d0a99329fc7ac5792974a23cd43f12). The zeta2 markers are:

~~~
<[fim-suffix]>
<[fim-prefix]><filename>edit_history
<filename>{path}
<<<<<<< CURRENT
<|user_cursor|>
=======
<[fim-middle]>
~~~

The extension parser is inline in owner extension.ts (SHA256 368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285). The banked no-op runner preserves the same algorithm in eval_noop_fp.py (SHA256 947d5dc6280bf85b831012116541f067f1104c46a682b3e3dc7ef722893f637d), and the extracted helper is experiments/eval/prediction_parser.py SHA256 421ba755f26912a0e931e2363d2264e4748208511816a1478c0241f7fb0e6467. The helper matched the TypeScript algorithm on marker/cursor, marker-echo/repetition, and blank-line fixtures.

The extension stop list is exactly:

~~~
[">>>>>>> UPDATED", "<<<<<<< CURRENT", "=======", "<[fim-middle]>", "<[fim-suffix]>", "<[fim-prefix]>", "<|outline|>"]
~~~

Scenario evaluation uses only >>>>>>> UPDATED, max 640 tokens, temperature 0, and stream false. The no-op evaluation uses the full extension stop list and max 320. Historical server settings are context 8192, eight threads, one slot, and loopback binding.

The retained [fixture_check.py](fixture_check.py) is the exact read-only command for the three parser fixtures and the zeta2 renderer fixture:

~~~
python3 docs/campaign/work/runtime/fixture_check.py
~~~

It checks both the active owner files and /mnt/e/sepalith/campaign-20260915/source-snapshots/20260912T005156Z/owner against the recorded hashes for extension.ts, eval_noop_fp.py, prediction_parser.py, run_eval.py, eval_scenarios.py, assemble_sft_v2.py, and scenarios.py. The observed result is PASS; no server was started and no CUDA module was imported.

**FRESH-START RECIPE:** Run only when 18099 is free. The following recipe was shell-parsed and hash/version-checked but deliberately not executed:

~~~bash
set -euo pipefail
BIN='/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server'
MODEL='/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf'
EXPECTED_BIN='123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804'
EXPECTED_MODEL='e343feacbdb262f515c11c1b6b69c93781b3803f26184d810c5c3ed25f32512d'
test -x "$BIN"
test -f "$MODEL"
test "$(sha256sum "$BIN" | cut -d ' ' -f1)" = "$EXPECTED_BIN"
test "$(stat -c %s "$MODEL")" = 2012011904
test "$(sha256sum "$MODEL" | cut -d ' ' -f1)" = "$EXPECTED_MODEL"
if (exec 9<>"/dev/tcp/127.0.0.1/18099") 2>/dev/null; then
  exec 9>&-
  echo '18099 has a TCP listener; classify its owner or unhealthy service before starting' >&2
  exit 75
fi
exec "$BIN" -m "$MODEL" --alias sepalith --temp 0 --port 18099 --host 127.0.0.1 -c 8192 --parallel 1 -t 8 -ngl 0
~~~

**EXISTING-SIDECAR RECIPE:** First make a loopback TCP connect to 127.0.0.1:18099 so an unrelated or unhealthy listener is caught before relying on /health. Then query /health and a one-token /v1/completions request to classify the answering port. If the extension owns the tracked child, invoke its Stop server command, wait for 18099 to stop answering, and use the fresh-start recipe. If the server is external, leave it running and ask its owner to stop it. The extension explicitly will not kill an external server. If that owner is unavailable, use the alternate-port shell in the JSON receipt; it repeats the binary/model hash checks and TCP probe on 18109 before the exec, then set these extension settings:

~~~json
{
  "sepalith.modelPath": "/home/m0hawk/Documents/Sepalith/experiments/models/packaging_b4-Q8_0.gguf",
  "sepalith.serverPath": "/home/m0hawk/Documents/Sepalith/experiments/bin/llama/llama-b10453/llama-server",
  "sepalith.backend": "cpu",
  "sepalith.gpuLayers": 0,
  "sepalith.port": 18109,
  "sepalith.threads": 8,
  "sepalith.contextSize": 8192,
  "sepalith.autoStart": false
}
~~~

Never use pkill llama-server, kill by port, or start a second owner on 18099. The TCP probe catches any listener, including an unhealthy or unrelated service. It is a preflight check with a race before exec; the server bind is the final ownership decision. The alternate-port shell command and all checks are in the JSON receipt.

**HISTORICAL DENOMINATORS:**

- The 2026-09-08 paired quality archive has 1,539 rows: 255 scenario rows and 258 no-op rows per arm across three arms. Q8 is 196/255 exact and 217/255 valid; its false-suggestion result is 120/204 scored no-op cases. Q4 regular is 193/255 exact and 216/255 valid; imatrix is 195/255 exact and 213/255 valid. The 54 mixed-expectation no-op rows are excluded from the 204-case restraint denominator. The receipt is a frozen export comparison with adoption not assessed.
- The old packaging smoke is 15 cases (three rows from each of five families): Q8 is 9/15 exact and 10/15 validator pass. Its raw files are absent at the documented paths, so it is historical smoke evidence.
- CPU timing is 120 rows: four arms (Q8, Q4 regular, Q4 imatrix, Q8 bookend), 10 traces x 3 repetitions each. Q8 cycle median is 13,521.289 ms; Q4 regular 10,362.639 ms (1.304811x); Q4 imatrix 10,442.432 ms (1.294841x); bookend 13,651.7575 ms (0.990443x). These repeated traces are not independent and do not promote a quant.
- Intent judging is 132 rows, 44 per arm: Q8 31/44 satisfied, Q4 regular 34/44, imatrix 34/44; 78 unique inputs and 54 cache hits. Judge uncertainty is unmeasured.
- V1d b4 versus v7 is 147 pairwise points sampled from 1,208 common V1a points (57 frozen trajectories, 32K cohort): b4 wins 32, v7 wins 35, ties 61, six unparsed, 300 calls, sign p 0.807. It is a felt-preference tie.
- O2f1800 is a separate historical comparator: 255/307 exact versus bank 242/307, 15 gains and 2 losses, paired p 0.00235; raw proposal counts are O2 192/262 versus bank 194/262 generated legacy points, with 58 mixed mid-typing points excluded from strict scoring. The Sept 10 strict bank no-op value is 136/204. Its preference column has 78 consistent pairs of 200, with 65 rate-invalid and 20 order-conflict pairs; b4 wins 46 versus O2 32 (p 0.1405), so adoption remained unissued.

These denominators are historical. They do not establish Tuesday MiniCPM target quality or promotion.

**ACCEPTANCE:** pass. The artifact/hash, launch contract, independent rollback recipes, and historical denominators are recorded. No rollback server was started under the packet constraints.

**UNRESOLVED:** No live serving smoke; historical evidence remains bounded by its archived source/runtime identities; AMD notebook/editor identity requires user-provided access information.

**NEXT:** Lead uses packaging_b4-Q8_0.gguf for the banked rollback, performs a live one-slot CPU completion smoke in an idle window, and reruns the Tuesday development-set quality gate before promotion. Concrete Opus5 question: on the pinned b10453 CPU binary and packaging_b4 model, which smallest matrix of threads (4/8/12), batch or ubatch, mmap/mlock, and flash-attention settings should be tested to reduce one-slot end-to-end cycle time while preserving the exact zeta2 renderer, parser, stop list, temperature 0, and one-listener ownership contract? Require a paired quality check against the 242/307 bank and stop if exact output or p95 worsens.

## Return protocol

**CHANGED_FILES:**

- docs/campaign/receipts/RUN-01-serving-architecture.json
- docs/campaign/receipts/PRE-08-fallback-ledger.json
- docs/campaign/work/runtime/worker.md
- docs/campaign/work/runtime/fixture_check.py
- docs/campaign/work/runtime/gguf_tensor_compare.py

**ARTIFACTS:** The two JSON receipts and the two executable read-only checks above; protected model files and archived quality evidence were read only.

**LEASE_RELEASED:** yes — CPU metadata and architecture inspection only; no server, CUDA, training, benchmark, cloud, or host-configuration lease was held.

## RUN-06 — speculative-paths and serving-contract audit

**STATUS:** partial. The JSON receipt is [RUN-06-speculative-paths-audit.json](../../receipts/RUN-06-speculative-paths-audit.json). RUN-01's target host/editor identity is still missing, and no converted Midtrain GGUF or released DSpark pair is present in the bounded local paths.

**PINNED SOURCE:** b10453 is git `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`; the executable is build 10453 with SHA256 `123dc314b4a796bd091419171f5190966074460fa5863263847eb6b90208f804`. `--help` proves `--spec-type ... ngram-mod ... draft-dspark`, `--spec-draft-n-max`, all three `--spec-ngram-mod-*` flags, `--parallel`, and `--cache-reuse`. The server help has no parse-special flag. `common/arg.cpp:3160-3194` scopes `--parse-special`/`--no-parse-special` to tokenize/imatrix examples; `/completion` hardcodes `add_special=true, parse_special=true` at `tools/server/server-context.cpp:4193-4199`.

**PATHWAY DECISIONS:** `ngram-simple` is the model-free lossless baseline; `ngram-mod` is the first scout with exact flags `--spec-type ngram-mod --spec-draft-n-max 64 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`. The pinned defaults are also visible in `common/common.h:351-355`; `common/speculative.cpp:1812-1978` uses the ngram `n_min/n_max` and a shared 4 MiB-entry state, while `:2276-2300` selects its output limit. The released DSpark facts remain five draft layers, 323,776,001 parameters, seven draft tokens, taps `[1,10,20,30,39]`, and published acceptance 5.5174 at temperature zero, but no local DSpark artifact or MiniCPM target-specific converter is available. Its published result is paired to the released final and cannot be transferred to Midtrain/SFT/RL. The pinned Midtrain header has 381 BF16 tensors and no MTP/NextN/DSpark/draft/speculator keys, so native MTP is not claimed. Existing MiniCPM1B 12/16/20-layer HF candidates share tokenizer.json byte hash `3e065a...fed81` but use hidden 1536 versus target hidden 2048; they are `draft-simple` hypotheses only. Existing B1 GGUF metadata reports pad 130559 versus target HF pad 1 and needs a preflight failure or explicit correction.

**PRM04 TOKEN CONTRACT:** POST `/tokenize` accepts `content`, `add_special`, `parse_special`, and `with_pieces` (`tools/server/server-context.cpp:4915-4952`; README `:618-632`). Use `parse_special:false` so literal `</s>` remains text; `parse_special:true` recognizes EOS ID 1. BOS ID 0 is conditional on the loaded GGUF `tokenizer.ggml.add_bos_token` metadata; the HF target config says `add_bos_token:false`, so export metadata must be checked. Assert exactly one leading 0 and no 1 for a fixture that contains no intentional EOS, or explicitly prepend integer 0 and record that policy. Preserve `/tokenize`'s integer IDs and send them as an all-number `/completion` prompt: `server-common.cpp:942-990` copies all-number arrays raw, and `tokenize_mixed:833-867` suppresses automatic BOS when the first mixed element is an integer. Native final response exposes `stop_type` (`none/eos/limit/word`) and `stopping_word` (`server-task.cpp:253-365`). Stop handling is literal full/partial string matching; `t_max_predict_ms` only gates after newline timing and `n_indent` is indentation logic. Line-aware extraction remains extension/parser work.

**CACHE REUSE:** Opus advice to try `--cache-reuse 256` first is a hypothesis. The server guard at `server-context.cpp:3133-3191` requires `llama_memory_can_shift` and prompt caching, and disables the request when unsupported. Plain KV returns true except STEP35 or `n_pos_per_embd()>1` (`llama-kv-cache.cpp:1171-1180`); hybrid Qwen3.5 delegates to attention memory (`llama-memory-hybrid*.cpp`), while recurrent memory alone returns true. Therefore b4 Qwen3.5 GDN/recurrent eligibility is unresolved until load-time logging reports the actual guard. Ordinary MiniCPM Llama likely passes the static plain-KV guard, but no target GGUF was loaded. If enabled, compare paired continuity and end-to-end time; then run the separate ngram-mod scout. Do not infer a second warm server or FIFO queue behavior.

**DEFERRED SMOKES:** After target conversion, use one slot, `-c 10240 -b 256 -ub 256 -t 2 -tb 2 -ngl 0`, temperature zero, unchanged renderer/parser/stop list. Ngram-mod uses 20 frozen 2K + 20 frozen 8K traces, one pass; require exact parity, at least 1.4x paired wall speed, bootstrap lower bound above 1.0 and no p95 regression. A future released DSpark smoke uses `-m <released-target> -md <released-dspark> --spec-type draft-dspark --spec-draft-n-max 7` only after help/artifact/tokenizer checks. Any adapted DSpark training must first build a target cache from TRAIN rows only, with exact taps and a maximum two-GPU-hour reassignment; the current target cache is empty. These are estimates and command shapes, not executed runs.

**CODE GAPS:** `experiments/eval/spec_bench.py` currently hardcodes Qwen b4/b2 and baseline/ngram-simple/draft-mtp/model-draft arms; it needs ngram-mod and admitted MiniCPM arms with artifact/tokenizer identity checks. No validated Midtrain HF-to-GGUF layer-drop/DSpark exporter exists. The PRM04 fixture runner should exercise `/tokenize` with `parse_special:false`, then integer `/completion`, and retain native stop fields; this worker did not run a server.

**RUN-06 CHANGED FILES:** `docs/campaign/receipts/RUN-06-speculative-paths-audit.json` and this worker receipt. No source, model, extension, host, CUDA, or campaign-control file was changed.

## PRM-05 — CPU history-provider preparation

**STATUS:** partial preparation; formal PRM-05 completion remains gated on PRM-04 and integration.

**OWNER:** worker-runtime

**OBSERVED:** 2026-09-12T04:55:37+02:00

**SOURCES:** PRM-03 schema `18b19dc54a2c6441efca259ecd1ac051f951d68ca0a9141bef9509590fdf7a6d`; PRM-04 explicit-token-mode receipt `989186747dc7093b7443162d402264d639caeacb86ceb94286cd9cbcaea71a9c`; execution HEAD `a79d6de38890355ec66d7227c974140160ab314e`; unchanged protected `campaign_protocol.ts` `34b9018a85ced948de070f808ce1b49e1a85646f6985f4993e29b3bf70323520`, `context_build.ts` `3bc148464dde243d6d4edca49ba2a3536de541f035afb1c5a68d756e36ef1047`, and `extension.ts` `368d6e502bb0f7c743ca5e38ec79c800c74b0776a14669bb1166985094217285`. The exact JSON receipt is [PRM-05-context-selector-preparation.json](../../receipts/PRM-05-context-selector-preparation.json).

**IMPLEMENTATION:** Added new execution files [history_provider.ts](/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/src/history_provider.ts:1) and [check-history.ts](/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith/scripts/check-history.ts:1). `HistoryProvider.seed` keeps one complete trusted pre-edit buffer per URI. `record` consumes a VS Code-shaped post-change event, recovers removed text from that buffer, replays simultaneous disjoint ranges using UTF-16 offsets, validates the result against `document.getText()`, then commits the next snapshot. Missing snapshots, stale versions, invalid/overlapping ranges, and post-event desynchronization produce no history and clear the affected URI where a prior snapshot could be unsafe. The wire event retains the pre-edit version, full-buffer SHA256, URI, exact old/new text, kind, workspace revision, and UTF-16 range. Batch entries are ordered by pre-edit range offset for reproducible sequence IDs.

The provider retains at most eight complete events and 4096 rendered diff characters, evicting oldest events as whole units and recording their IDs. `selectSourceRange` binds current unsaved text to URI/version/full-buffer SHA and classifies same-line ranges as `inline`; multi-line ranges require a separately guarded user-accepted WorkspaceEdit. CRLF/mixed buffers and trailing spaces remain in exact source evidence.

**FIXTURE EVIDENCE:** The executable fixture contains 51 assertions covering two disjoint simultaneous edits, sequential versions, emoji UTF-16 columns and surrogate-split rejection, CRLF and lone-CR/mixed-EOL policy, trailing spaces, typing/undo/redo, same-version empty save preservation, changed/lower-version rejection, defensive returned-event copies, stale/missing/desynchronized snapshots, per-document isolation, unsaved untitled buffers, exact source/hash/range identity, and oldest-first whole-event eviction. `node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types scripts/check-history.ts` passed (`PRM-05 history checks passed: 51`). Both targeted and full-tree invocations of the pinned TypeScript compiler passed. `npm run compile` cannot run in the execution worktree because its local `tsc` is absent; the equivalent pinned compiler invocation passed. No server, model, CUDA, cloud, benchmark, or host configuration was touched.

**EOS IDENTITY CHECK:** Vocab-only GGUF `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain-vocab-only.gguf` is 5,124,962 bytes, SHA256 `5452fb552a3970a7e8b15b74ccdeb6582201642c9885ddfe751c3f2dc69602b9`. Metadata has BOS 0, EOS 1, `add_bos_token=false`, `add_eos_token=false`, and no EOG key. Indexed tokens are `1='</s>'` and `130073='<|im_end|>'`, both GGUF token type 3 (`LLAMA_TOKEN_TYPE_CONTROL`). In pinned b10453 `llama-vocab.cpp` SHA `3fea10f4481b504d5ca894b32fc177bf2eb83ffdf3f38f3f9c9175f62f62cd4b`, lines 2815–2874 add both token texts to `special_eog_ids`, and lines 2888–2892 ensure configured EOS 1 is also EOG. `server-context.cpp` SHA `26f130b76c27be72e4674943754575cf5efa14b6a6325591be07df57f651e681`, lines 1885–1890, maps any `llama_vocab_is_eog` token to `stop_type=eos`. The HF generation config declares [1,130073]; native stopping includes both even though GGUF metadata stores scalar EOS 1. The live gate must classify a 130073 termination as noncanonical/early unless the campaign explicitly admits it. `parse_special=false` affects input parsing and does not remove this generation EOG behavior.

The HF source of the EOS array is `/mnt/e/sepalith/campaign-20260915/models/minicpm5-2b-midtrain/generation_config.json` (213 bytes, SHA256 `9ac4f32e5f32358697a9f438a3ea89ef80e6ba786c72c49e932f9f21c122fdb1`) and declares `eos_token_id=[1,130073]`, `bos_token_id=0`, `pad_token_id=1`. The scalar GGUF EOS metadata does not capture the full HF EOS list; native b10453 still recovers both EOG IDs from their token texts.

**REMAINING INTEGRATION:** PRM-04 live integer-prompt parity and terminal behavior; provider seeding before the first change per URI; `onDidChangeTextDocument` wiring; workspace-relative path/revision capture; stale checks and omission/refresh after desync; context-builder history consumption; and guarded same-line/multi-line acceptance. This preparation does not claim a live provider or any serving/performance result.

**CHANGED FILES:** execution `extensions/vscode-sepalith/src/history_provider.ts` and `extensions/vscode-sepalith/scripts/check-history.ts`; plan receipt `docs/campaign/receipts/PRM-05-context-selector-preparation.json`; this worker receipt. Lease released.

## PRM-05 — CPU integration and gated native serving path

**STATUS:** partial integration. The implementation and offline checks pass. Formal completion remains with lead review, context-dropout/data-consistency gates, and a live managed primary-profile provider smoke.

**OBSERVED:** 2026-09-12T05:28:40+02:00. The exact JSON receipt is [PRM-05-context-selector-integration.json](../../receipts/PRM-05-context-selector-integration.json), SHA256 `2e26c2b50e00ab3add3a50bf85d63ac529cb6e9281a3ccb7583ca574bcbf420e`.

**EVENT INTEGRATION:** `HistoryProvider` is initialized during activation, seeds all open R documents, seeds new R documents through `onDidOpenTextDocument`, clears closed URIs through `onDidCloseTextDocument`, and handles each change before post-accept swallowing, cancellation or debounce. Empty save events preserve history when the version and complete buffer are unchanged. A missing snapshot, stale version, invalid/overlapping replay, or desynchronization is followed by a complete-buffer reseed. The existing latest-request cancellation and debounce behavior remains in place after history capture.

**PRIMARY GATE:** `runtime.ts` now accepts a strict `zeta2-prm03-v1` profile in addition to the existing `zeta2-v1` profile. The primary profile pins PRM-03 schema/renderer, tokenizer revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, tokenizer JSON SHA `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, config SHA `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`, tokenization policy, BOS 0, canonical EOS 1, vocab size 130560, and native EOG IDs `[1,130073]`. `provision()` returns the already validated manifest; the extension selects primary only for a managed child whose model override or installed model passed the manifest byte/SHA check. Manual and external unidentified sidecars retain the legacy string `/v1/completions` path, so a b4 sidecar is never paired silently with the primary renderer.

**NATIVE CLIENT:** `campaign_client.ts` renders the frozen PRM-03 context, calls `/tokenize` with `add_special:false`, `parse_special:false`, and `with_pieces:false`, prepends exactly one integer BOS 0, then calls native `/completion` with integer `prompt`, `n_predict`, `temperature:0`, `stream:false`, `cache_prompt:false`, and `return_tokens:true`. It sends no substring stop list. The response must contain integer IDs in `[0,130559]`, exact-parser-valid content with one `>>>>>>> UPDATED` line, `stop_type=eos`, and canonical final token 1. Pinned MiniCPM CONTROL ranges `0..7`, `10..21`, and `130072..130559` are rejected before the terminal; final 130073 or any other CONTROL/EOG is `noncanonical_eog`. IDs 8 and 9 remain ordinary USER_DEFINED text according to the vocab audit. This makes the terminal-token evidence authoritative even though native `--special=false` can hide CONTROL text in `content`.

**CONTEXT/APPLICATION:** `context_select.ts` captures the current unsaved buffer, relative POSIX path, UTF-16 range, code-point/UTF-16 cursor columns, document version/full-buffer SHA, bounded prefix/suffix and defensive provider history. Existing scope outline and enclosing-function remainder are represented in PRM-03 fields. References, diagnostics and retrieval remain empty until real providers exist. `next_edit.ts` calls the frozen stale/hash guard immediately before planning; same-line output is automatic inline, while multiline plans are marked user-accepted WorkspaceEdit and are never returned as automatic inline text. Mixed/lone-CR buffers stay outside the automatic primary path. Legacy prompt rendering/parser/stop behavior remains unchanged when the profile gate is absent.

**EXECUTION SOURCE HASHES:**

- `extensions/vscode-sepalith/src/extension.ts` — `95a682d5d5d6b0ac781eaa6127e0d0fbf7f5d23867fdc5cf4ff4eb75420b88de`
- `extensions/vscode-sepalith/src/runtime.ts` — `7a3027e15466dff7efb14b9fa3d6c2e7e99627b1164257705d427e956da0e92f`
- `extensions/vscode-sepalith/src/history_provider.ts` — `b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251`
- `extensions/vscode-sepalith/src/context_select.ts` — `e521af74d9985ceb8e4e2da8f97b9778b6a73f19fa4b83dc4d61ee586b1ebf53`
- `extensions/vscode-sepalith/src/campaign_client.ts` — `19b5aea145e328b1c105f3bfd57a3fc4fbcf06f06f212defd09167446066c35c`
- `extensions/vscode-sepalith/src/next_edit.ts` — `3aa796a9498e30c9016b12fff62a89f19af06048e994b027f8a2e97134f79924`
- `extensions/vscode-sepalith/src/campaign_protocol.ts` (root-frozen shared guard) — `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824`
- `extensions/vscode-sepalith/scripts/check-prm05.ts` — `062d69c2e0bbdb20fcb9be178d057dcaea4c02de5488bb835ff1e3eb92248103`
- `extensions/vscode-sepalith/package.json` — `0ce45f80f596cbc798062e86d12f4f5d1f5d60338843e7d9baa66b42c679ee31`

**CHANGED FILES (INTEGRATION):** execution `extensions/vscode-sepalith/src/extension.ts`, `src/runtime.ts`, `src/context_select.ts`, `src/campaign_client.ts`, `src/next_edit.ts`, `src/history_provider.ts`, `scripts/check-prm05.ts`, and `package.json`; plan `docs/campaign/receipts/PRM-05-context-selector-integration.json` and this worker receipt. The frozen `campaign_protocol.ts` was read and consumed through its root-owned guards; it was not edited by this worker.

**VERIFICATION:** `check-prm05.ts` passed 24 assertions using mocked `/tokenize` and `/completion` transport, including exact no-special fields, integer BOS prompt, no stop key, `return_tokens`, exact terminal parsing, stale identity rejection, multiline WorkspaceEdit planning, noncanonical EOG, all CONTROL-before-terminal, out-of-range/noninteger IDs and strict manifest matching. The accepted history fixture passed 51 assertions. The pinned full TypeScript compiler passed with the canonical `@types` tree; the pinned esbuild bundle passed (`96.1kb`). Existing context checks passed 46, completion boundary checks passed, and all 58 shared runtime manifest fixtures plus download/provision checks passed. No server was started by this worker.

**LIMITATIONS:** No live VS Code host was available for event timing, InlineCompletionItem display, or WorkspaceEdit application command. No real primary manifest/model was provisioned here; lead owns the live CPU probe and promotion. The explicit apply-next-edit command is wired and identity-guarded but remains unexecuted live. The automatic selector intentionally emits same-line ranges only. `npm run compile` in the execution worktree remains unavailable because local `node_modules` is absent; the equivalent pinned compiler invocation passed.

**NEXT:** Lead reviews the source/receipt pair, then exercises one managed primary-profile child in an idle approved host window against the accepted PRM-04 live fixture. Keep external/manual sidecars on the legacy path until identity is proven.

## PRM-05 — v2 integration corrections

**STATUS:** partial integration; v2 supersedes the preceding integration receipt for the corrected source behavior. The exact receipt is [PRM-05-context-selector-integration-v2.json](../../receipts/PRM-05-context-selector-integration-v2.json), SHA256 `07163f740544af47512288cc0635b27440fb0ed03209d9ced978dbfdc4c523ab`.

**OBSERVED:** 2026-09-12T05:45:16+02:00. Execution HEAD remains `a79d6de38890355ec66d7227c974140160ab314e`. The frozen protocol source is root-owned and unchanged at SHA256 `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824`.

**CORRECTIONS:**

- `next_edit.ts` now routes from range geometry only. A same-line replacement range may carry multiline `insertText` in an automatic `InlineCompletionItem`; only a multiline replacement range requires the explicit user-accepted `WorkspaceEdit` path.
- `RequestGeneration` owns the active lease. It increments and aborts on a superseding request or explicit edit/close invalidation, and `finally` clears only its own `inFlight` promise. Primary and legacy responses check generation, non-aborted signal, VS Code `CancellationToken`, and URI/version/full-buffer SHA before response logging, display planning or cache writes. Request keys also include cursor position, so cursor moves in an unchanged buffer supersede an earlier request.
- `fetchSymbols` records the starting document version and full-buffer SHA, stores both in each cache entry, and discards an awaited language-server result if either changes. It never caches symbols tied to an earlier buffer.
- The primary manifest profile declares `contextSize` and `maxOutputTokens`; the native client rejects prompt BOS plus generation overflow and requests above the profile cap. The managed sidecar rejects a profile/configuration context mismatch. Final values remain pending lead profiling; fixtures use 8192 and 320.
- Native `/completion` now sends `cache_prompt: true`, retains integer prompt/`return_tokens` evidence, and rejects `truncated: true`, `stop_type=limit/length`, noncanonical EOG/control IDs and parser-invalid output. A primary manifest on an occupied loopback listener is refused before any legacy fallback. A TCP connect catches an unhealthy listener before spawn; the bind race remains explicit.
- Change events replay history synchronously before request invalidation, post-accept swallowing and debounce. Empty same-version saves preserve history; desync/stale/missing snapshots are reseeded from the complete current buffer. Close clears the URI and provider state.

**SOURCE HASHES:**

- `extensions/vscode-sepalith/src/extension.ts` — `d135eddeca3681c003053c2489efd9df2076f8d8988811f8cc6da4a35d455c88`
- `extensions/vscode-sepalith/src/runtime.ts` — `3a26fb1677eb97488fa03491ba795ec0cca264db24af8f5f7db983bf46e5bfbf`
- `extensions/vscode-sepalith/src/history_provider.ts` — `b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251`
- `extensions/vscode-sepalith/src/context_select.ts` — `e521af74d9985ceb8e4e2da8f97b9778b6a73f19fa4b83dc4d61ee586b1ebf53`
- `extensions/vscode-sepalith/src/campaign_client.ts` — `87a2cbebfda5ed9eee1ccb63accbf02562330156bb3a5881e9011a615f039bfd`
- `extensions/vscode-sepalith/src/next_edit.ts` — `c184e0d26499290ce3a8f339f2172343ab233b0799fac57a86476ff14e0e2214`
- `extensions/vscode-sepalith/src/campaign_protocol.ts` (root-owned frozen input) — `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824`
- `extensions/vscode-sepalith/scripts/check-prm05.ts` — `1f580496b463d0035b62f5562891b28c439550ab43f03f00f27152cafcab19f8`
- `extensions/vscode-sepalith/scripts/check-history.ts` — `8f410e8036002aaa4d5c4256ce7d6bfa7bb2d8c61a6e3c63e3def627c71b568e`
- `extensions/vscode-sepalith/package.json` — `0ce45f80f596cbc798062e86d12f4f5d1f5d60338843e7d9baa66b42c679ee31`

**VERIFICATION:** `check-prm05.ts` passed 32 assertions, including the corrected same-line multiline behavior, supersede/invalidate ownership, context/output cap rejection, truncation rejection and cache reuse. `check-history.ts` passed 51 assertions. The pinned TypeScript compiler passed the full extension tree, the esbuild bundle passed at 104.2kb, and the existing runtime manifest/download/provision checks passed (58 fixtures). `git diff --check` passed. The repository lint command was attempted but is blocked because the execution worktree has no `@oxlint/plugins` package for its JS plugin; no lint finding was asserted. No server, model load, CUDA, cloud, benchmark or host configuration was touched.

The 6000 rendered-source-character selector remains a documented candidate policy. Its source/hash is recorded in the v2 JSON, but Python/data-consistency and context-dropout review have not frozen it as an optimum. No live VS Code host was available for event timing, cancellation delivery, inline display, cursor moves or WorkspaceEdit application. No real primary manifest/model was provisioned by this worker; lead retains live serving and promotion.

**CHANGED_FILES:** execution `extensions/vscode-sepalith/src/extension.ts`, `src/runtime.ts`, `src/history_provider.ts`, `src/context_select.ts`, `src/campaign_client.ts`, `src/next_edit.ts`, `scripts/check-prm05.ts`, `package.json`; plan `docs/campaign/receipts/PRM-05-context-selector-integration-v2.json` and this worker receipt. Frozen `campaign_protocol.ts` was read only.

**ACCEPTANCE:** partial — corrected offline integration evidence passes; formal PRM-05 completion remains gated on lead review, final profile values, context-dropout/data consistency and a managed primary VS Code smoke.

**NEXT:** Lead reviews the v2 receipt/source hashes, then runs one matched managed primary host smoke plus the occupied-port refusal path before promotion.

## PRM-05 — v3 provider ordering and native accounting correction

**STATUS:** partial integration; v3 supersedes v2 for request ordering and native response accounting. Receipt: [PRM-05-context-selector-integration-v3.json](../../receipts/PRM-05-context-selector-integration-v3.json), SHA256 `4f5d7a5d8929150671b63a34b4b14a86c1a1289c19c2a8038f788b15ecd2ab28`.

**OBSERVED:** 2026-09-12T05:56:48+02:00. Source ownership remains the execution extension files and `check-prm05.ts`; frozen `campaign_protocol.ts` was not edited.

`RequestGeneration.begin()` now runs at `provideInlineCompletionItems` entry, before sidecar/document checks and every awaited scope/LSP operation. The lease is passed through `fetchSymbols`, scope validation, primary serving and legacy serving. A stale scope result cannot start a request or supersede a newer cursor invocation. The outer `finally` disposes the invocation's VS Code `CancellationToken` subscription and finishes only its own lease. Stored in-flight entries carry their lease; same-key reuse requires that lease to remain current and non-aborted, and a cancelled joiner resolves to an empty result without reviving stale work.

The mocked orchestration fixture resolves scope and transport stages out of order while ignoring abort. It proves the old request is discarded and the newest response remains publishable. Native completion validation now checks optional `tokens_evaluated` against the full integer prompt length including manual BOS 0, and rejects generated token evidence longer than `n_predict`.

**SOURCE HASHES:**

- `extensions/vscode-sepalith/src/extension.ts` — `f83fadef7d9c2e13912f012f6908e2dbda168a072166e77bbf73d4aae8c855ba`
- `extensions/vscode-sepalith/src/runtime.ts` — `3a26fb1677eb97488fa03491ba795ec0cca264db24af8f5f7db983bf46e5bfbf`
- `extensions/vscode-sepalith/src/history_provider.ts` — `b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251`
- `extensions/vscode-sepalith/src/context_select.ts` — `e521af74d9985ceb8e4e2da8f97b9778b6a73f19fa4b83dc4d61ee586b1ebf53`
- `extensions/vscode-sepalith/src/campaign_client.ts` — `0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333`
- `extensions/vscode-sepalith/src/next_edit.ts` — `c184e0d26499290ce3a8f339f2172343ab233b0799fac57a86476ff14e0e2214`
- `extensions/vscode-sepalith/src/campaign_protocol.ts` (root-owned frozen input) — `ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824`
- `extensions/vscode-sepalith/scripts/check-prm05.ts` — `0a476f17b5096b2f3b07f9645ea8e314d2a544b602c25211a6ec0e29f4127df6`
- `extensions/vscode-sepalith/scripts/check-history.ts` — `8f410e8036002aaa4d5c4256ce7d6bfa7bb2d8c61a6e3def627c71b568e`
- `extensions/vscode-sepalith/package.json` — `0ce45f80f596cbc798062e86d12f4f5d1f5d60338843e7d9baa66b42c679ee31`

**VERIFICATION:** `check-prm05.ts` passed 38 assertions. `check-history.ts` passed 51 assertions. The pinned TypeScript compiler passed the full extension tree; esbuild produced a 106.1kb bundle; `git diff --check` passed. The existing runtime manifest/download/provision fixtures passed. No server, model load, CUDA, cloud, benchmark or host configuration was touched.

Final profile context/output limits and the 6000-character source selector remain candidate/pending lead profiling and Python/data-consistency review. No live VS Code host was available for actual event timing, token cancellation delivery, cursor movement, InlineCompletionItem display or WorkspaceEdit acceptance. Lead retains live serving, host identity and promotion.

**CHANGED_FILES:** execution `extensions/vscode-sepalith/src/extension.ts`, `src/runtime.ts`, `src/history_provider.ts`, `src/context_select.ts`, `src/campaign_client.ts`, `src/next_edit.ts`, `scripts/check-prm05.ts`, `package.json`; plan `docs/campaign/receipts/PRM-05-context-selector-integration-v3.json` and this worker receipt. Frozen `campaign_protocol.ts` was read only.

**ACCEPTANCE:** partial — v3 ordering, cancellation and native accounting corrections pass offline; formal PRM-05 completion remains gated on lead review, final profile values, context-dropout/data consistency and a managed primary VS Code smoke.

**NEXT:** Lead reviews the v3 source/hash receipt, then exercises an out-of-order scope/transport cancellation case, occupied-port primary refusal and fresh matched child in an approved host window before promotion.

## PRM-05 — selection and dropout parity v2

**STATUS:** partial preparation. The freshness and fixed-dropout correction supersedes the first selection receipt. See [PRM-05-selection-parity-v2.json](../../receipts/PRM-05-selection-parity-v2.json).

**OBSERVED:** 2026-09-12T06:18:30+02:00. Execution HEAD is `a79d6de38890355ec66d7227c974140160ab314e`. `extension.ts`, `runtime.ts`, `campaign_protocol.ts` and `history_provider.ts` remained root-owned/read-only for this packet.

The new pure `campaign_selection` modules select source lines and evidence without reading target edits or creating history. `selectSourceWindow` counts complete logical lines in UTF-16 units plus LF separators, retains the exact current region and all lines in a valid enclosing pin, and alternates nearest prefix/suffix lines in the remaining budget. Required region/scope overflow is explicit and source omissions carry inclusive line spans. `context_select.ts` delegates its automatic source window to this policy while preserving the strict PRM-03 `PromptContext` shape; `selectPromptSource` exposes the provenance for the later real-adapter wiring.

Evidence content integrity is checked separately from freshness. Each candidate carries a source identity/version and content SHA; it is fresh only when all three match a current source snapshot. Therefore an old excerpt with a self-consistent old hash is still stale when its current source version differs. Fresh required records are retained and required stale records block support-verified dropout. Optional records remain included on overflow unless support metadata verifies the target remains supported, the fixed policy ID and seed are present, and required IDs/fingerprint are unchanged. Verified support applies a deterministic 20 percent dropout, records `dropped_by_policy`, then packs the remainder by the evidence budget.

The TypeScript and Python fixtures contain eight identical cases: unsaved Unicode/CRLF, oversized scope, balanced missing-provider source, required-only evidence, verified 20 percent dropout, unverified optional overflow, content-hash-stale required evidence and self-consistent-but-old source/version evidence. Compact UTF-8 fixture payloads are byte-identical at SHA256 `b4a1c4c5121c6a023103e28bc117b02eabeeb1e2ff36a422ac6ef97a951caac6`.

**SOURCE HASHES:** `campaign_selection.ts` `c5eb9ea9088a06d2a8849e83addf5a2266d40530cde1e32bd3a862716e0cc6cc`; `context_select.ts` `a72edaf733395d7b4839f2ca9ed0c2092e4051359279f6839c93e2d5d04b9d59`; `check-campaign-selection.ts` `a7ee9193df15b346154e9098a0671cc6401d0ba0b1fac1f2cd6ce0cd169a3bda`; Python module `5517beec87f13e5fb925b4d18c5f6c8c908fd7666011873ecedbdf4f7aa52598`; Python tests `03abc08400a199fe098defc57268296bb20e163a0df2607f1f8f1101250fe24d`.

**VERIFICATION:** `check-campaign-selection.ts` passed eight fixture rows; existing `check-prm05.ts` passed 38 assertions; Python selection tests passed 11 and selection plus protocol tests passed 30. No live VS Code/provider/sidecar/host action occurred. The 6000-unit source limit and 20 percent dropout are deterministic profiling policies, not accepted quality/latency optima. Scope outline accounting and real provider freshness/support metadata remain root integration gates.

**CHANGED_FILES:** execution `extensions/vscode-sepalith/src/context_select.ts`, `src/campaign_selection.ts`, `scripts/check-campaign-selection.ts`, `packages/sepalith/src/sepalith/campaign_selection.py`, `packages/sepalith/tests/test_campaign_selection.py`; plan `docs/campaign/receipts/PRM-05-selection-parity.json` and `PRM-05-selection-parity-v2.json`; this worker receipt.

**ACCEPTANCE:** offline parity and freshness/dropout corrections pass; formal PRM-05 completion remains gated on root provider integration, profile sizing and live host smoke.

**NEXT:** Root reruns the pinned full TypeScript compiler and provider regression, then supplies real current source snapshots/support metadata before considering the candidate limits for profiling.

## RL-01 — fixed-ID runner parameterization preparation

**STATUS:** CPU implementation complete; live training blocked. Receipt: [RL-01-runner-parameterization-preparation.json](../../receipts/RL-01-runner-parameterization-preparation.json), memo: [RL01-implementation-runner-parameterization.md](RL01-implementation-runner-parameterization.md).

**CHANGED_FILES:** execution `experiments/training/campaign_rl_data.py`, `experiments/training/campaign_rl_train.py`, `experiments/training/test_campaign_rl_train.py`; plan receipt and memo above. No shared source, data, registry, host, model or GPU state was edited.

**OBSERVED:** The runner requires admitted canonical PRM-03 rows, a hashed pre-edit context sidecar, and an ordered hashed selected-ID file. It consumes stored prompt IDs with manual BOS 0, native EOG `[1,130073]`, canonical terminal 1, exact parser reward `exact + 0.2*line-F1`, r16/a16 seven-module identity, and future merged-SFT manifest/weight hashes. TRL 0.24.0's conflicting generation/steps fields are resolved as a rows-sized generation/per-device batch with gradient accumulation 1 and TRL-derived `steps_per_generation=1`; default rows/update is 32 (G=4: 8 groups, G=2: 16), with a one-group memory arm available. Full checkpoint, control and development-evaluation callbacks are wired; the builder constructs but never trains.

**VERIFICATION:** system Python 22/22 tests passed with three expected TRL skips; pinned `.venv-sft` 22/22 passed with one skip because optional `mergekit` prevents importing the complete GRPOTrainer; actual pinned GRPOConfig and `trl.models` fixed-ID generation fixtures passed. `py_compile` passed and legacy `test_rl_smoke_el.py` passed 26/26. No model/CUDA/server/cloud/final-set action occurred. Exact source hashes and unresolved live transition/resume/parent gates are in the JSON receipt.

**ACCEPTANCE:** preparation complete; lead retains admitted-data, merged-SFT parent, post-Trainer tokenizer, Unsloth/PEFT `_flag_for_generation`, G/memory, real Trainer interruption/resume, launch and promotion gates.
