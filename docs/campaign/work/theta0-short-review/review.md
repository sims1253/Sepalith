# RUN-04 theta0 short-trace independent review

This is a read-only comparison of the retained theta0 Q8 short trace against
the earlier SFT500 Q8 trace. It is a serving-observation audit only. It does
not claim a quality result, a causal speed result, or a promotion.

## Inputs and identity

| item | theta0 run | prior SFT500 run |
| --- | --- | --- |
| trace | `docs/campaign/work/lead/theta0-q8-transition-a/short-deadline5s.json` | `docs/campaign/work/lead/transition-panel-v2-q8-a/serial.json` |
| trace SHA-256 | `046dd49615e3b3f4c79d4bce518892c7143495e01dd4952864caeb46d9559549` | `e8be32ed7218a74f3e26fdadcb454d653e49e2d0d43cc795139a1740ed86b47a` |
| model SHA-256 | `22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559` | `f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256` |
| launch record | `docs/campaign/work/lead/theta0-q8-transition-a/launch.json` (`1c0bbd71fc100eb0f49541ad59b16696201c99515c3aa98c68b376b684d59438`) | `docs/campaign/work/lead/transition-panel-v2-q8-a/launch.json` (`806014b184372f1cc9775cbdc0c7a996bb8950b8c1f70b209c9ad3c8d6885e06`) |
| binary SHA-256 | `92a39a4fe4972653d096c26587e5f7f903f5ff5f38504f46ca1e7e5c0207095f` | same |
| endpoint/mode/deadline | `127.0.0.1:18405`, serial, 5000 ms | same |

Both traces report schema `sepalith.serving.transition-panel.trace.v2`, nine
serial rows, nine fresh responses, eight quality-denominator-eligible fresh
responses, one unchanged duplicate control, zero timeouts/cancellations, and
zero protocol rejects. The readout used for the earlier run is
`docs/campaign/receipts/RUN-04-transition-panel-q8-lead-readout.json` (SHA-256
`3725de7ec956ac39f375518b00a0e5395747e8d5a473f3c954cfca057eeada74`).

The native request profile and renderer source object compare equal in all
fields. The exact profile is:

```text
tokenize: /tokenize, add_special=false, parse_special=false, with_pieces=false
completion: /completion, integer_token_ids_with_one_manual_bos_0,
            n_predict=192, temperature=0, stream=false, cache_prompt=true,
            return_tokens=true, greedy=true
manual BOS=0, canonical EOS=1, vocab=130560, max_output=192, parallel=1
normal deadline=5000 ms, diagnostic deadline=60000 ms, launchesServer=false
```

Both launch records use the same native argv after the model path and port:
`-t 6 -tb 6 --threads-http 2 --parallel 1 -c 4096 -b 256 -ub 256 -ngl 99
-lv 4`. The binary is the same pinned Vulkan AVX2 build; only the model
artifact and launch time differ.

The pinned source object is also identical: `campaign_client.ts`
`0390fd1bf1af94197b6771fe2b96c8e833b13e9d0d123c92fc4bcd69fc9c1333`,
`campaign_protocol.py`
`5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156`,
`campaign_protocol.ts`
`ec4ab1529cb7ade10be28343600f26be027f24239009ec354a357a7caff4b824`,
`campaign_requests.ts`
`4fdc6a881b93ae8beb9614c2b210f80fd5b0a2d5592ae02c8174345612a17535`,
`campaign_selection.ts`
`70a03d86c8d1c2cd0e117edcbd69cfbac193382b20b555e2d0a327e7e006e8e3`,
`context_select.ts`
`a72edaf733395d7b4839f2ca9ed0c2092e4051359279f6839c93e2d5d04b9d59`, and
`history_provider.ts`
`b2763b7d1b146d398835f4a15eb84e3f0bccfa3bb07b2e9faf5ca9a4eda0a251`.
The tokenizer JSON/config pins are respectively
`3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81` and
`e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`.

## Per-event comparison

The prompt token ID and prompt text hashes are equal for all nine paired event
IDs. `eval` is the response top-level `tokens_evaluated`; `prefill` is
`responseMetrics.timings.prompt_n`; `decode` is
`responseMetrics.timings.predicted_n`; milliseconds are the corresponding
native timing fields. Values are `theta0 / SFT500`.

| event | eval | prefill | generated/decode tokens | elapsed ms | prompt ms | decode ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-a1 | 178 / 178 | 178 / 178 | 14 / 14 | 2645.170 / 1528.872 | 1219.245 / 751.342 | 1395.641 / 739.277 |
| typing-a2-number | 211 / 211 | 117 / 117 | 14 / 14 | 2262.035 / 1270.721 | 857.115 / 527.465 | 1377.807 / 737.265 |
| typing-a3-comment | 244 / 244 | 119 / 119 | 16 / 16 | 2781.429 / 1376.467 | 855.992 / 536.274 | 1598.349 / 854.417 |
| cursor-a3-label | 244 / 244 | 163 / 163 | 13 / 13 | 2571.950 / 1420.347 | 1215.667 / 716.121 | 1264.275 / 684.076 |
| history-a4-result | 277 / 277 | 186 / 186 | 14 / 14 | 2631.870 / 1499.646 | 1294.616 / 776.713 | 1383.196 / 743.980 |
| diagnostics-a4-refresh | 300 / 300 | 85 / 85 | 14 / 14 | 2225.560 / 1495.419 | 875.708 / 498.704 | 1418.494 / 744.186 |
| anchor-a4-result-use | 300 / 300 | 202 / 202 | 11 / 12 | 2842.883 / 1553.230 | 1714.674 / 950.035 | 1093.668 / 628.631 |
| file-switch-b1 | 148 / 148 | 89 / 89 | 13 / 19 | 2123.573 / 1499.041 | 888.217 / 508.363 | 1321.674 / 1020.425 |
| unchanged-repeat-b1 | 148 / 148 | 1 / 1 | 13 / 19 | 1387.206 / 1062.195 | 104.529 / 60.352 | 1297.969 / 1024.144 |

Generated token ID hashes are equal on the first six rows and differ on
`anchor-a4-result-use`, `file-switch-b1`, and `unchanged-repeat-b1`. Thus the
theta0 run generated one fewer token on the anchor row and six fewer on each
of the final two rows. The first six rows have identical generated counts.
Those are observed token-stream/count differences; they are not quality
measurements.

## Supported timing observations

Across all nine rows, theta0 elapsed time is 1387.206–2842.883 ms (mean
2385.742 ms, median 2571.950 ms), while the earlier trace is
1062.195–1553.230 ms (mean 1411.771 ms, median 1495.419 ms). The elapsed
mean difference is +973.971 ms. Excluding the unchanged control, the first
eight rows are 2510.559 ms versus 1455.468 ms on mean elapsed time.

The native fields move in the same direction for every row: all theta0
`prompt_ms` values are higher and all theta0 `predicted_ms` values are higher.
All nine `tokens_evaluated` and all nine `prompt_n` values are identical
between runs, so prompt length, measured prefill count, and the serving
profile do not explain the observed difference. The theta0 `predicted_ms` is
also higher on every row where generated counts are identical, and remains
higher on the three rows with fewer generated tokens. The retained evidence
therefore supports only this statement: this theta0 run recorded higher
client elapsed, native prompt, and native decode times under the same request
profile and prompt/token inputs while serving a different model artifact.

The traces do not contain matched host utilization, thermal state, driver
state, server startup state, or transport/Node overhead measurements. The
readout says client wall time includes campaign-host Node startup and
LAN/SSH-forward overhead; cached-prefix counts are inferred from
`tokens_evaluated - prompt_n`, not a native cache-hit flag. Host/backend and
run-to-run variation remain unresolved, as does any model-computation
contribution. No causal speed attribution or latency win is admitted.

The protocol evidence also differs on the final rows: theta0 records
`no_op` for `file-switch-b1` and `unchanged-repeat-b1`, whereas the prior
trace records `fresh_plan`. This is a renderer/applicability observation and
does not alter the timing comparison or establish editor application.

## Reproduction and limits

The CPU-only audit was run with Python against the two retained JSON traces;
it read no model bytes and launched no server, GPU, native client, SSH, or
network operation. The comparison asserted equal row IDs, native profiles,
source objects, prompt text hashes, prompt token hashes, and all nine equal
prefill/eval pairs, then computed the table and summary statistics.

This is a nine-event small synthetic panel. It has no model-quality,
losslessness, real-editor-application, cancellation, long-context, or
promotion claim. The prior readout remains the cited native baseline; this
independent review records the theta0 comparison only.
