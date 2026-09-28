# RUN-01 theta0 notebook admission preparation

This packet prepares a root-owned managed notebook server and a three-case
native protocol probe for the transferred theta0 Q8/Q6 artifacts. The worker
did not start a server, open a model, read model tensors, use CUDA, or write
over SSH. The local checks use only newly authored synthetic R snippets and a
shape-only tokenizer; they make no native tokenizer, quality, or latency claim.

The exact profile is in `theta0-notebook-profile.json`. It binds the accepted
theta0 Q8/Q6 hashes from
`RUN-01-theta0-quant-c-lead-review.json` to the accepted b10453 Vulkan server
(`16000` bytes, SHA-256
`92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f`). The
CPU fallback hash is recorded as a separate identity. The profile is based on
the accepted long-transition renderer and the managed editor-G route; their
source pins are recorded rather than copied or modified.

Run the offline checks on the worker CPU:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python3 theta0_notebook_probe.py --self-test
```

The three checks are:

1. `/tokenize` uses `add_special=false`, `parse_special=false`, and
   `with_pieces=false`, and returns integer IDs without native control IDs.
2. Completion prompt construction prepends manual BOS `0` exactly once and
   accepts canonical terminal EOS `1` from native EOG `[1, 130073]`.
3. Native EOG `130073` is surfaced as noncanonical and rejected rather than
   being treated as canonical PRM-03 EOS.

The shape-only IDs in `--self-test` are deliberately not the pinned
tokenizer. Before a live check, root must create a private copy of
`token-reference.template.json`, use the pinned tokenizer revision
`8dc5f6055b90fe4b9422340810b270b9569f37f3` to fill all three
`expected_native_token_ids` arrays, and record each canonical-array SHA-256.
The live driver refuses null or mismatched references, so no hand-written
token IDs can pass the parity gate.

Root may then start one fresh managed endpoint on m0pad after the transferred
model has been admitted. The launch template checks the binary and candidate
model bytes/SHA-256, refuses an existing run directory, owns the process
group, and enforces a 1100-second hard timeout. Example (run by root on the
target host, with a new `RUN_ID` for each candidate):

```sh
ROOT=/home/m0hawk/.local/share/sepalith-campaign-20260915 \
  CANDIDATE=Q8_0 PORT=18404 \
  RUN_ID=run-01-theta0-q8-native-$(date -u +%Y%m%dT%H%M%SZ) \
  bash launch_theta0_notebook.sh
```

The launcher uses the exact managed profile: `-t 6 -tb 6
--threads-http 2 --parallel 1 -c 4096 -b 256 -ub 256 -ngl 99 -fa on -lv 4`.
It leaves the endpoint alive for the bounded probe and any separately
admitted follow-up. Use a new run directory and fresh output for a Q6 check or
for a later DEV admission; never append to an existing trace.

With an already-running endpoint and the populated reference, root runs the
three native checks with the normal five-second budget:

```sh
python3 theta0_notebook_probe.py \
  --server http://127.0.0.1:18404 \
  --reference /path/to/private/theta0-token-reference-q8.json \
  --candidate Q8_0 --deadline-ms 5000 \
  --out /path/to/fresh/run-01-theta0-q8-native.json
```

For an operational stall diagnosis only, repeat against a fresh server/run
with `--deadline-ms 60000`. The driver records request/response hashes, full
token arrays, prompt length, `tokens_evaluated`, terminal/EOG classification,
per-case timing, errors, and denominators. A timeout is an operational
failure, is excluded from any quality denominator, and is never converted to
a rejection or quality result.

The existing renderer contract can be checked before serving and replayed
only after root admits the endpoint:

```sh
node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/serving-transition-long-panel/check-long-transition-panel.ts

node --no-warnings=MODULE_TYPELESS_PACKAGE_JSON --experimental-strip-types \
  /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/serving-transition-panel/replay-transition-panel.ts \
  --fixture /home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/serving-transition-long-panel/long-transition-fixture.json \
  --server http://127.0.0.1:18404 --mode serial \
  --out /path/to/fresh/run-01-theta0-transition.json
```

Root acceptance requires all of the following for each candidate: exact
remote binary/model hashes and fresh process evidence; all three reference
arrays equal the native `/tokenize` arrays under the pinned tokenizer
identity; each completion prompt has one leading BOS and no automatic
special-token insertion; response IDs are integer and in-vocabulary; native
EOG is identified as `[1, 130073]`; canonical EOS `1` is the only successful
terminal; `130073`, truncation, malformed stop fields, and control IDs are
rejected; and the five-second and diagnostic sixty-second results remain
separate. The renderer's complete-buffer identity, parser/no-op evidence, and
fresh output paths must also remain intact. These gates establish protocol and
runtime readiness only. They do not establish model quality, editor
application, throughput, p50/p95, or a latency win.

The template cannot supply native token IDs without the pinned tokenizer, and
the worker did not verify the transferred Q6 file or any live endpoint. Root
must perform those checks, own SSH/tunnel/server cleanup, retain full logs,
and decide whether a separate DEV admission is warranted.
