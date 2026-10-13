# RUN-09 F16 versus Q8 native DEV review

This review compares the completed native Vulkan 75-case DEV runs at step 500:

- F16: `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-primary500-f16-dev-a/quality.json`
- Q8: `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-09-primary500-q8-dev-a/quality.json`
- Shared panel: `/mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl`

The raw run directories were read only. Their quality, terminal, launch, server, device-audit, client, and stderr artifacts remain at those paths. Their byte counts and SHA256 values are recorded in the receipt.

Both quality receipts are complete `sepalith.campaign.run09.native-dev-quant-quality.v2` evaluations over the same 75 IDs in the same order. The panel hash is `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`, and the ordered case-ID hash is `6e0c4c5bd7de4f0d9aa6eab602796c13da110f56edab6b0dd8a9e04fb66ee6bb` for both runs.

The prompt identity is exact for all 75 rows: HF prompt IDs, native prompt IDs without BOS, prompt hashes, prompt token identity, and complete request objects all match F16/Q8. The production request contract is also equal: manual BOS 0, context 4096, `n_predict=512`, temperature 0, `cache_prompt=false`, returned token IDs, stream enabled, and no server stop list. Every row has wire-to-HF text parity and no wire/HF decode error in both runs.

Generated output is byte/token exact for 66/75 rows. Nine rows differ in returned token IDs and raw/decoded text:

| ID | Family | F16 tokens | Q8 tokens | F16 terminal/cap | Q8 terminal/cap |
| --- | --- | ---: | ---: | --- | --- |
| `dat07-existing-152d62f7a57472f8abb30ca9` | roxygen_drafting | 79 | 80 | canonical / within | canonical / within |
| `dat07-existing-5b5e844d3a8795c52149f848` | roxygen_drafting | 104 | 106 | canonical / within | canonical / within |
| `dat07-existing-06097c12b0328d475be857e1` | roxygen_drafting | 72 | 74 | canonical / within | canonical / within |
| `dat07-existing-7e51067ae755fcd8dcda1765` | no_op | 307 | 512 | canonical / within | non-EOS / cap hit |
| `dat07-derived-783c8b64bbb90fb346e9933b` | no_op | 512 | 512 | non-EOS / cap hit | non-EOS / cap hit |
| `dat07p-a3532882e98a227a44eb5536c0fb` | format_propagation | 44 | 38 | canonical / within | canonical / within |
| `e623a61b5a4c066358a477f2` | finish_block | 512 | 512 | non-EOS / cap hit | non-EOS / cap hit |
| `157517ba47dbab157f7c361a` | finish_block | 54 | 73 | canonical / within | canonical / within |
| `f43de3e77f2d92ed7b223464` | finish_block | 512 | 203 | non-EOS / cap hit | canonical / within |

The nine changed ID-array SHA256 pairs are in the receipt. Seven rows differ in cap metadata; six of those changes do not change the stored classification (the three roxygen rows, `dat07-derived-783c8b64bbb90fb346e9933b`, `e623a61b5a4c066358a477f2`, and `157517ba47dbab157f7c361a`). All cap differences remain within the fixed inclusive 512-token request policy. Across all rows, each run has 70 canonical EOS responses, five cap hits, and 68/75 equal cap metadata objects; EOS metadata is identical on 71/75 rows.

Re-running the pinned `campaign_eval.classify` against each raw response, the shared panel `PromptContext`, expected region, and returned IDs matched the stored quality classification on all 75 F16 rows and all 75 Q8 rows. For edit rows, `classify.exact_region` was compared with the receipt’s edit-exact field. For no-op rows, predicted-no-op, strict-no-op-correct, and no-op false-positive fields were compared with their stored meanings. This avoids treating a no-op label as a prompt or request input.

The only three per-case classification changes are:

| ID | Family | F16 → Q8 change | Mechanical interpretation |
| --- | --- | --- | --- |
| `dat07-existing-7e51067ae755fcd8dcda1765` | no_op | valid canonical response with a no-op false positive → cap hit/non-EOS protocol failure | Q8 removes one false-positive count by failing mechanically; strict no-op-correct remains unchanged |
| `dat07p-a3532882e98a227a44eb5536c0fb` | format_propagation | accepted non-exact edit → accepted exact edit | one Q8 exact edit gain |
| `f43de3e77f2d92ed7b223464` | finish_block | cap hit/non-EOS protocol failure → accepted non-exact edit | one Q8 protocol-valid gain without exact edit |

Aggregate denominators are:

| Metric | F16 | Q8 |
| --- | ---: | ---: |
| Panel / attempted / responses | 75 / 75 / 75 | 75 / 75 / 75 |
| Protocol-valid rows | 70 | 70 |
| Mechanical/protocol failures | 5 | 5 |
| Cap-hit rows | 5 | 5 |
| Exact edit rows | 25 | 26 |
| Exact-region rows | 44 | 45 |
| Strict no-op cases / correct | 32 / 19 | 32 / 19 |
| No-op false positives | 12 | 11 |
| Transport failures / partial responses | 0 / 0 | 0 / 0 |

Family changes are localized. `finish_block` moves from 2 to 3 protocol-valid rows, 4 to 3 cap/mechanical failures, and remains 0 exact edits. `format_propagation` moves from 5 to 6 exact edits. `no_op` moves from 31 to 30 protocol-valid rows, 1 to 2 cap/mechanical failures, and 12 to 11 false positives while strict no-op correctness stays 19. `na_rm_propagation`, `pipe_rewrite`, `rename_propagation`, and `roxygen_drafting` have identical classification denominators; roxygen has 0 exact-region rows in both runs.

Runtime identity is matched at the harness level. Both receipts report tokenizer revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`, tokenizer JSON hash `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`, tokenizer config hash `e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b`, vocabulary size 130560, BOS 0, EOS/PAD 1, and vocabulary hash `9178ce39e267573fdcb234518b3bd60ffb4750c42fe3ec8afc60d95afeb4031d`. Static preflight is equal: the pinned `campaign_client.ts`, `campaign_eval.py`, and protocol hashes match; the legacy B4 renderer is absent; all prompts fit context with maximum 2619 tokens including BOS. The server build is `b10453-3cb7ffb1a`, `n_ctx=4096`, and both health checks report `ok`. The model path and provenance differ only by candidate: F16 `model-F16.gguf` / `50f523af997f2f77dcbd187adde8a36dbc41529932703fd016e506b172761725`; Q8 `model-Q8_0.gguf` / `f0be11a9215adc7eef68820ac907fc899e08db8ef8c72c27771e6f93d096b256`.

Both client and SSH supervisors exited zero, both remote servers exited zero, and both raw stderr files are empty. The remote device audits show the same render node `/dev/dri/renderD128`, Radeon Vulkan library, and `libggml-vulkan.so.0.20.0` path family. F16 peak server RSS was 3,486,636 KiB and Q8 peak server RSS was 4,094,432 KiB; this difference is an observation from the terminal receipts and is not a quality conclusion.

A bounded Q4 native 75-DEV run is mechanically admissible under the same harness, conditional on root creating a candidate-specific model provenance and repeating the terminal/server/device audit. The F16/Q8 evidence supports reusing the fixed panel, 4096 context, 512 inclusive cap, no-stop-list request, exact renderer/tokenizer contract, and current native client/evaluator/build identities. The recommendation does not promote Q4 or transfer F16/Q8 quality: Q4 still needs its own complete receipt, outputs, per-case protocol checks, and root acceptance. The 5 mechanical failures in each comparison run remain part of the diagnostic denominator.

Review command, run CPU-only with the pinned environment and without a model/framework load:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/sepalith/src:experiments/training /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python - <<'PY'
# read both known quality.json files and the pinned DAT-07 panel;
# invoke campaign_eval.classify for all 75 rows and compare identities
PY
```

The review receipt records the exact inline command, raw artifact hashes, comparison counts, changed IDs, and unresolved Q4 admission conditions.
