# R2 serving screen follow-up

This packet records a CPU-only readout of the completed four-arm screen and a
larger paired panel prepared from accepted TRAIN rows. It does not load model
weights, start a server, use CUDA, or use DEV/final data.

The prior `pair-analysis.json` status is `fail` because its summary objects
look for top-level acceptance fields. The completed adapter records the fields
inside each request's `timings` object. `extract_screen_counters.py` uses those
fields as the primary record and cross-checks them against both the server's
`print_timing` lines and cumulative `statistics` lines. It never infers
acceptance from returned text, output length, or speed.

The verified screen counters are:

| arm | draft tokens | accepted tokens | acceptance | counter status |
| --- | ---: | ---: | ---: | --- |
| released-dspark | 392 | 102 | 26.0204% | JSON timings + server log agree |
| trained-dspark | 462 | 92 | 19.9134% | JSON timings + server log agree |
| model-free-ngram | 0 | 0 | undefined (0 denominator) | server verifies zero draft tokens |
| ordinary-baseline | n/a | n/a | n/a | speculation is not applicable |

For released DSpark, the eight JSON timing pairs are
`(56,3), (56,3), (56,22), (56,22), (49,12), (49,12), (35,14), (35,14)`.
For trained DSpark they are
`(49,4), (49,4), (77,19), (77,19), (49,12), (49,12), (56,11), (56,11)`.
The cumulative server terminal values are respectively `392/102` and
`462/92`. All four arms returned eight accepted protocol records and each
candidate had 8/8 greedy token parity in the existing pair analysis. Existing
screen medians were about 102 ms for ordinary baseline, 140 ms for released
DSpark, and 161 ms for trained DSpark. Those results do not establish a
promotion decision or justify more training; the paired panel below is the
next measurement.

Run the extractor against the immutable screen directory with:

```sh
python3 extract_screen_counters.py \
  --screen-dir /home/m0hawk/.local/state/sepalith/campaign-20260915/benchmarks/r2-four-arm-screen-v1 \
  --out screen-counters.json
```

The completed run produced `screen-counters.json` with status `pass`.

## TRAIN panel

`train-serving-panel.jsonl` contains 40 complete source rows selected by
stable SHA-256(row-id) order: 20 rows in a 1536..2048 prompt band labelled
`2k_band`, and 20 rows in a 2049..4096 band labelled
`4k_cap_long_band`. Selection used the corrected TRAIN source

`docs/campaign/work/lead/finish-corrected-train-v1/train-token-rows.jsonl`

with SHA-256
`e2408c5177e3134189c4b86f7247db41fc0c1ea5aab9d9260a3dea55906786fe`.
The source contains 11,526 TRAIN rows. After requiring a target body plus
terminal sequence of at most 191 tokens (the 192-ID serving cap reserves one
ID for protocol EOS), there were 2,548 eligible 2K-band rows and 1,238
eligible long-band rows. 2,031 source rows were excluded by that cap. The
selected panel has no DEV/final rows, no authored padding/EOS, and no
duplicate IDs; its prompt range is 1,558..2,765 tokens. The source's natural
long rows do not reach exactly 4,096, so `4k_cap_long_band` means long rows
under the 4K context contract, not fabricated 4K prompts.

`train-serving-panel.manifest.json` records source and row hashes, source
line numbers, families, selection counts, prompt/target geometry, and the
native BOS/EOS/EOG/PAD contract. The panel rows retain complete source fields
for audit; a serving client must send only `input_ids[1:target_start]` to
tokenize and `[0, *prompt_ids]` to completion. It must preserve generated
IDs, parse native EOG IDs `[1, 130073]`, and never append the source target
tail.

The existing `runtime_native_probe.py` admits prompts only through 2,048. The
packet-local `paired_panel_probe.py` is the reviewed client for this panel: it
admits the long band with `context_size=4096` while retaining the same
greedy/cold-warm protocol. `panel-run-command.txt` runs it once per admitted
arm and does not launch a server. A fair larger comparison must use identical
panel rows and request order across ordinary baseline, released DSpark,
trained DSpark, and any model-free comparator, then compare exact returned
token IDs and raw text before reporting protocol outcomes,
accepted/drafted denominators, and cold/warm latency. The client records
`draft_n` and `draft_n_accepted` only when both are present in the server's
final timings; it reports the denominator as unavailable otherwise.

The separate `eight-k-stress-spec.json` rejects 8K under the daily 4096-token
context. No 8K row is included and no truncation or padding is allowed. An 8K
measurement needs a separately approved model/server context and budget.

## Packet files

* `extract_screen_counters.py` — deterministic screen counter/log checker.
* `screen-counters.json` — actual screen readout.
* `select_train_panel.py` — hash-checked, TRAIN-only deterministic selector.
* `train-serving-panel.jsonl` — selected 40-row panel.
* `train-serving-panel.manifest.json` — source/panel manifest and contracts.
* `paired_panel_probe.py` — executable 4096-context cold/warm client.
* `test_panel_probe.py` — synthetic HTTP test (42 accepted requests).
* `panel-run-command.txt` — executable command plan for the client.
* `eight-k-stress-spec.json` — explicit daily-context rejection.
