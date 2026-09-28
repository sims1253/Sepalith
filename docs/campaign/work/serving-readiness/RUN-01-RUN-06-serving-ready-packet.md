# RUN-01/RUN-06 serving-readiness packet

**Observed:** 2026-09-12, Europe/Berlin. This packet is preparation only. It
does not download or transfer a model, start a server, run a benchmark, read a
final set, alter the extension, or alter shared campaign state.

**Task IDs:** RUN-01 (runtime and host placement), RUN-06
(SPEC-READINESS-90M / speculative-path audit).

## Decision

The fixed CPU serving host is m0hawk@192.168.178.40 (m0pad). The live
read-only SSH check at 2026-09-12T13:05:14Z matched the earlier runtime-build
receipt: CachyOS Linux 7.2.3, AMD Ryzen 5 PRO 5650U, AVX2, 6 physical/12
logical CPUs, 14 GiB RAM (12 GiB available at intake), and 338 GiB free on
/home. The pinned CPU AVX2 llama-server and llama-cli are present on that host.
The bounded check found no llama-server, llama-cli, Node/code process, tmux
session, or listener on ports 18099, 18109, or 18401. This is a current
bounded observation, not a claim that all user jobs are absent.

Place the model, its hash manifest, the TRAIN-only fixture, the server binary,
and the benchmark harness on m0pad. If the editor is used, open the workspace
through Remote-SSH so the VS Code extension host is also m0pad. The extension's
127.0.0.1 is resolved by the extension host; a local editor cannot reach a
remote loopback server without a separately measured tunnel. No tunnel is
part of this recipe. The notebook/editor owns display and documents; the
remote extension host owns one sidecar and one port.

The manual CPU run owner is a named tmux session and tracked PID. The extension
owns a child only when it spawns that child. An answering external server is
never stopped by the extension. Keep extension port 18099 and speculative
benchmark port 18401 (then the next free 184xx port for a second arm) distinct.

## Pinned inputs and evidence classes

| Input | Evidence and hash |
| --- | --- |
| llama.cpp source | /home/m0hawk/Documents/Sepalith/experiments/bin/src/llamacpp-b10453, commit 3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70, tree f9a9f82f92eb23b6dbc05494e542ddb1f907a0c4; tracked archive SHA-256 f2c0528a047e084862dd9c34103c7cdb2de394526dbd0ae62929d85123cf6f34 |
| remote server | /home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-server, 16,000 bytes, SHA-256 e68d96b6dbc7f4ef3bed329f4f7cf146283cb10f443747e7fa5208078f2d69f6, version 0.1.0-dev (build 10453, commit 3cb7ffb1a) |
| remote CLI | /home/m0hawk/.local/share/sepalith-campaign-20260915/build-b10453-avx2/bin/llama-cli, 15,992 bytes, SHA-256 2546ca42ca7d31b8e9ea1ad9f43cc6e4542778e2db4458c527e385869d3af323, same version |
| remote build | Release/Ninja, AVX/AVX2/BMI2/OpenMP, native tuning and all GPU backends off, server and CLI built with two workers; UI assets absent. Historical build receipt: RUN-01-runtime-preparation.json. |
| target | Merged SFT theta0 GGUF is not available to this packet. PRE-05 records Midtrain revision 8dc5f6055b90fe4b9422340810b270b9569f37f3, LlamaForCausalLM, 42 layers, hidden 2048, vocab 130560, and 381 header tensors with no mtp, nextn, dspark, draft, or speculator keys. This is header/config evidence, not loaded tensor proof. |
| renderer | Frozen zeta2-prm03-v1, tokenizer revision 8dc5f6055b90fe4b9422340810b270b9569f37f3, tokenizer JSON SHA-256 3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81, tokenizer config SHA-256 e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b, manual BOS 0, canonical EOS/PAD 1, native EOG [1,130073]. Frozen contract: PRM-08 / PRM-09. |
| extension source | EXEC /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith; worktree HEAD 40c35a0baed4e48149254179d7dbb26fea4b348d. src/extension.ts SHA-256 86367f6615e02e59bed425dea8b33f0c210aaf1c0be6a9c6d91e840111bf5cd9; src/runtime.ts 3a26fb1677eb97488fa03491ba795ec0cca264db24af8f5f7db983bf46e5bfbf; src/process_lifecycle.ts a621d5c4a0e227b5e5a2cd12191142b6d1c4a779e1329893389efa5c35068e27. |

Historical/local facts above are retained with their source receipts. The
live SSH facts are limited to host identity, capacity, binary presence/hash/
version, bounded process names, and selected listeners. No model path or model
hash was invented from the missing target.

## Exact launch and placement recipe for the lead

Run this only after the merged theta0 GGUF and a TRAIN-only fixture have been
admitted with hashes. Substitute angle-bracket paths; do not use a filename as
an identity. The following is a recipe, not an execution record.

On m0pad, make one immutable run directory and record inputs before binding:

~~~sh
ssh -o BatchMode=yes m0hawk@192.168.178.40
set -eu
BASE=/home/m0hawk/.local/share/sepalith-campaign-20260915
RUN_ID=run-01-theta0-cpu-$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$BASE/serving/$RUN_ID
BIN=$BASE/build-b10453-avx2/bin/llama-server
MODEL=<admitted-theta0-gguf-on-m0pad>
TRAIN_FIXTURE=<admitted-train-only-fixture-on-m0pad>
mkdir -p "$RUN_DIR"
sha256sum "$BIN" "$MODEL" "$TRAIN_FIXTURE" > "$RUN_DIR/inputs.sha256"
stat -c '%n %s %a %y' "$BIN" "$MODEL" "$TRAIN_FIXTURE" > "$RUN_DIR/inputs.stat"
~~~

For an extension-host smoke, use the exact child argument order in
EXEC/.../src/extension.ts:375-387. CONTEXT_PROFILE and MAX_OUTPUT_PROFILE must
come from the reviewed theta0 manifest/profile. PRM-09's RL mechanism profile is
context 2240 with response cap 192, while the S1 speculative rig requires
context 10240. These are different profiles and must not be silently mixed.

~~~sh
PORT=18099
CONTEXT_PROFILE=<reviewed-extension-context>
THREADS=8
LOG=$RUN_DIR/extension-sidecar.log
SESSION=sepalith-$RUN_ID
tmux new-session -d -s "$SESSION" \
  "exec \"$BIN\" -m \"$MODEL\" --alias sepalith --temp 0 --port $PORT --host 127.0.0.1 -c $CONTEXT_PROFILE --parallel 1 -t $THREADS -ngl 0 >\"$LOG\" 2>&1"
PID=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}')
printf '%s\n' "$PID" > "$RUN_DIR/server.pid"
ps -o pid=,ppid=,stat=,comm= -p "$PID" > "$RUN_DIR/server.pid.observed"
~~~

Readiness is a real completion probe, not /health alone. Poll
POST /v1/completions with {"prompt":"x","max_tokens":1,"temperature":0,
"stop":null,"stream":false} until HTTP 200 within the source's 180-second
budget. The native PRM-03 path calls /tokenize with add_special:false and
parse_special:false, prepends exactly integer BOS 0, then calls /completion
with the all-integer prompt, reviewed n_predict cap, temperature:0,
stream:false, cache_prompt:true, and return_tokens:true. NativeCampaignClient
rejects truncation, a noncanonical terminal, an early native control token, or
a parser-invalid response.

For a CPU speculative screen, stop the extension child first and use port
18401. The first arm is a baseline; the candidate is one mode only. Both use
the pinned b10453 binary, -t 2 -tb 2 -b 256 -ub 256 -np 1 -c 10240 -ngl 0,
temperature zero, and the fixed stop marker. The ngram-mod candidate adds the
source-confirmed flags below:

~~~sh
# baseline, on m0pad; bind 18401
$BIN -m "$MODEL" --host 127.0.0.1 --port 18401 \
  -t 2 -tb 2 -b 256 -ub 256 -np 1 -c 10240 -ngl 0 \
  -s 20260905 --temp 0

# candidate, in a separate quiet run after baseline persistence; same port
$BIN -m "$MODEL" --host 127.0.0.1 --port 18401 \
  -t 2 -tb 2 -b 256 -ub 256 -np 1 -c 10240 -ngl 0 \
  -s 20260905 --temp 0 \
  --spec-type ngram-mod --spec-draft-n-max 64 \
  --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64
~~~

The S1 spec_bench.py harness is a protocol/metrics reference, but it is
hard-coded to the held-out Qwen S1 trace set and b4 paths. Do not point it at
the current TRAIN fixture without an explicitly reviewed adapter. Its
measured-source request flags are /completion, stream:true,
cache_prompt:false for cold and true for warm, n_predict:64, temperature:0,
and stop:[">>>>>>> UPDATED"]; it records TTFT, prompt time, decode tok/s, wall
time, acceptance, and exact baseline parity.

## TRAIN-only fixture and acceptance

The smallest useful fixture is one paired 2K row and one paired 8K row, both
from split:"train", rendered under PRM-03. Each JSONL row should carry an
immutable id, split, ctx_class, full prompt, and expected target ending in the
exact terminal line >>>>>>> UPDATED. Its sidecar manifest records source package
IDs, row count, tokenizer/renderer IDs, fixture hash, and creation command. Do
not include final/evaluation rows or use a held-out trace cache. One cold and
one warm request per row is enough for wiring and latency collection; two rows
cannot support a promotion claim.

For each row, persist:

* baseline and candidate raw output and output SHA-256;
* native token IDs, tokens_evaluated, stop_type, canonical final token, and
  draft_n/draft_n_accepted fields when present;
* prompt token count, cold/warm TTFT, prompt milliseconds, predicted tokens,
  decode tok/s, wall milliseconds, errors, and host load;
* candidate_output == baseline_output, target terminal/parsing status, and
  binary/model/fixture hashes.

The TRAIN-only fixture must pass zero crashes, native PRM-03 acceptance (EOS 1,
no early native control token, exact terminal and parser status), and exact
greedy baseline/candidate output equality. The existing RUN-06 promotion gate
is separate: 20 frozen 2K plus 20 frozen 8K paired traces, one pass, unchanged
64-token cap, context 10240, exact parity, at least 1.4x paired end-to-end
speed, bootstrap lower bound above 1, and no p95 regression. That larger gate
is explicitly not run by this packet.

For a learned draft, the cache and training data must be TRAIN-only. The
smallest adaptation screen is one frozen target snapshot, target hidden-state
cache from TRAIN rows only, one tiny draft/export, then the same two-row
baseline/candidate native parity smoke. The existing audit estimates 30–90
minutes CPU for export/parity and permits up to two GPU-hours only by
reassignment from the already approved optional consolidation envelope. No
such cache or training was started here.

## Existing speculative-path status

These statuses reuse RUN-06-speculative-paths-audit.json and
SPECULATIVE-PATHS.md; no settled research was repeated.

| Path | Status for Midtrain/SFT theta0 |
| --- | --- |
| ngram-simple | Model-free baseline comparator. Prior loaded smoke was faster, but broader results were near 1x and host-load sensitive. Run only as a comparator if time remains. |
| ngram-mod | Highest-value model-free CPU scout. b10453 supports the exact flags above; no Sepalith result exists yet. It needs exact output parity and the registered wall/p95 gate. |
| Released MiniCPM5-2B-DSpark | Blocked for this target. The released draft is a separate exact pair for released post-trained MiniCPM5-2B: five draft layers, seven tokens, taps [1,10,20,30,39], 323,776,001 parameters. No bounded local DSpark target/draft files or MiniCPM converter recipe were admitted; its published acceptance does not transfer to Midtrain/SFT/RL bytes. |
| Matched lighter AR draft | Readiness hypothesis only. Existing MiniCPM1B layer-drop candidates share vocab/tokenizer JSON but have hidden size 1536 and 12/16/20 layers versus target 2048/42; existing pad metadata conflicts (130559 versus target HF pad 1). No target-matched draft is ready. |
| Native MTP/attached head | Blocked. PRE-05 finds no MTP/nextn/draft fields in Midtrain; the earlier Qwen head is a different architecture/tokenizer family. |
| EAGLE-3/DFlash | Reserve. b10453 advertises support, but no MiniCPM-compatible draft artifact, taps, converter, or paired target is present. |

Do not select a path from a model name or released-draft headline. The ordinary
baseline remains shippable while these gates are unresolved.

## Extension smoke and remaining gates

The source smoke path is
EXEC/extensions/vscode-sepalith/scripts/smoke.ts. It starts a tracked child
with the historical local abl_dropout-Q8_0.gguf and local server path, probes
real completion readiness, sends a synthetic zeta2 prompt with max_tokens:640,
and terminates only its child. It is a plumbing check, not a theta0/PRM-03
release check; its hard-coded paths must be replaced by a reviewed final-model
invocation or equivalent manual smoke.

Remaining build requirements on the EXEC extension:

~~~sh
cd /home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/extensions/vscode-sepalith
npm ci
npm run compile
npm run bundle
npx @vscode/vsce package --allow-missing-repository
~~~

Then install the VSIX in the Remote-SSH extension host, configure explicit
model/server paths for development or a reviewed HTTPS manifest for a primary
profile, and exercise in a real GUI window: Developer: Show Running Extensions
must confirm the extension host is m0pad; Sepalith reaches ready; one R
same-line suggestion completes; a no-op produces no edit; a changed document
cancels the stale request; and stop/deactivate leaves no owned child. SSH shell
access cannot establish editor acceptance. The EXEC worktree currently has no
node_modules and no dist; read-only checks gave npm run compile -> exit 127,
tsc not found, and npm run test:lint -> exit 1, 15 missing oxlint modules. No
dependency installation or build artifact was created by this packet.

## Shutdown, persistence, and no-retry rule

After every future run, persist inputs.sha256, command/flags, host load,
stdout/stderr log, per-request JSONL, summary, and terminal status in the run
directory before declaring the run complete. Keep the admitted model and
fixture immutable. If readiness, load, conversion, or a request fails, record
the failure and stop; do not retry with another model, port, thread count, or
draft mode under the same run ID.

For a manual tmux run, signal only the PID in server.pid:

~~~sh
kill -TERM "$(cat "$RUN_DIR/server.pid")"
for i in $(seq 1 50); do
  kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$(cat "$RUN_DIR/server.pid")" 2>/dev/null; then
  kill -KILL "$(cat "$RUN_DIR/server.pid")"
fi
test ! -e /proc/$(cat "$RUN_DIR/server.pid")
ss -H -ltn | grep -E ':18099$|:18401$' || true
tmux capture-pane -pt "$SESSION" -S -50 > "$RUN_DIR/tmux-tail.txt" || true
tmux kill-session -t "$SESSION" 2>/dev/null || true
~~~

The extension's lifecycle implementation uses the same ownership rule: TERM,
a 5-second grace period, KILL escalation, then a deadline while retaining
ownership if exit is not observed. This is source-backed lifecycle capability.
No termination was exercised in this packet, so no live termination result is
claimed.

## Acceptance and next action

RUN-01/RUN-06 remains partial/ready-for-lead-launch. Host identity and binary
readiness are live-observed; extension placement/lifecycle and speculative
flags are source-backed; model admission, GGUF load, native tokenization,
completion, latency, extension build, and GUI behavior remain unresolved. The
next concrete action is for the lead, after theta0 and the TRAIN-only two-row
fixture are hash-admitted, to run the remote input manifest and one quiet
baseline CPU completion smoke on m0pad, then persist and stop the tracked
process before any ngram-mod candidate. Estimated cost is $0 paid compute and
10–30 CPU minutes for the two-row baseline/spec wiring smoke on the six-core
host; a full 20+20 scout is 30–90 CPU minutes by the existing estimate. No
cloud, CUDA, or paid allocation was used.

