# DAT-04/RL-08 finish training lineage audit

This is a bounded lineage audit of the five `finish_block` rows sampled by the
prior boundary review. It answers whether the exact converted rows reached the
materialized SFT and RL inputs, and whether a downstream training or reward
consumer adds the missing outer `}`.

The audit used the five row IDs in
`docs/campaign/receipts/DAT-08-finish-boundary-source-audit.json` and matched
them by exact `id` through the accepted DAT-05 registry and provenance index,
the SFT train-token rows and 3000-step draw schedule, and the RL-02 eligible
rows, sidecar, selected-ID order, and v5 source-draw sequence. The packet body
hash, registry body hash, SFT body hash, and (where eligible) RL body hash were
compared. The source references were compared on file, line, source digest,
raw-line digest, group, and package. No source corpus line was opened.

## Input identities

| artifact | path | identity | rows/draws |
| --- | --- | --- | ---: |
| prior boundary receipt | `docs/campaign/receipts/DAT-08-finish-boundary-source-audit.json` | `1b8111c5ddae15e46f13b4fcc0d016e9b89f96313420369142c3508ceda0d8bd` | 5 sampled IDs |
| converted packet | `/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl` | `42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25` | 4346 reported; first 5 read |
| DAT-05 registry | `/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/registry.jsonl` | `ef8ca082be4699d52dab67cb9c42628df4cc935479fb769d880961e7c3c2cf3d` | 11839 |
| DAT-05 provenance | `/mnt/e/sepalith/campaign-20260915/data-work/DAT-05-registry-v1/provenance.jsonl` | `6883f0691301e79180164920414e4752f53edcfe555c73b1ea0b02dd599d2123` | 11914 |
| SFT selected rows | `/mnt/e/sepalith/campaign-20260915/data-work/SFT-inputs-v1/train-token-rows.jsonl` | `7641bbdc8f609aca1e0ad72177561edddfdbdae1b49a8444c15470652bf5ebb6` | 11764 train |
| SFT 3000-step schedule | `/mnt/e/sepalith/campaign-20260915/data-work/SFT-inputs-v1/draws-3000.json` | `a439fe8af68b4929a7747d4a82b9e0b26a05fcd3973f8a3c81148a84581d7f94` | 48000 draws |
| RL eligible rows | `/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/eligible-train-rows.jsonl` | `e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602` | 8440 |
| RL context sidecar | `/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/context-sidecar.jsonl` | `6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f` | 8440 |
| RL selected IDs | `/mnt/e/sepalith/campaign-20260915/data-work/RL-02-contexts-v1/selected-train-ids-lead-order-v1.json` | `24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d` | 8440 unique |
| RL v5 source draws | `docs/campaign/work/rl-pool/source-row-draw-sequence-v5-interleaved.json` | `2e75e10bf86f6b72e33fd72cfed96194bdd7c7c4741db8793681985292581574` | 24000; sequence `dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6` |

The v5 path above is the schedule admitted by
`docs/campaign/receipts/RL-02-v5-interleaved-admission.json` (receipt SHA
`8ca410ba0d8da94e0a8d1f7125888161643102b9ec08cda5ad96506e0d09852e`). The
earlier v4 schedule remains the source-draw path named by the initial RL-02
stage manifest (SHA `892991f89d063544d85902eba04c7607f3867897749d3eee4f4e2ac6f407ee48`,
sequence SHA `d44a076d0b850fd607e7c613caa096d04ed352292ceeeb2e17f736dc20a089f2`).
For these five IDs, v4 and v5 have the same per-ID draw counts; v5 is used for
the current selection result.

## Exact five-row lineage

Every row below matched the packet to the DAT-05 `source_ref` on source file,
line, source SHA, raw-line SHA, group, and package. The packet's
`target_body_sha256` equals the body text hash recomputed from the registry,
SFT row, and (for three eligible rows) RL row. The `registry/SFT/RL line SHA`
is the SHA-256 of the JSONL record bytes without its line ending; registry and
SFT line bytes are identical for all five.

| row ID | package; source file:line | packet canonical row SHA | source SHA | raw-line SHA | target body SHA | registry/SFT/RL line SHA |
| --- | --- | --- | --- | --- | --- | --- |
| `2b578e5b4936158eaab12ee3` | `author:epidemiology`; `finish_block_authored.jsonl:3` | `0430a187c195dde007afe25a1a088456deaa2de4b634d09e23077f87a1358eed` | `c9476a8f9d7c36a9cf91e769434db9245a9a9309e9cd215090884a0131b78734` | `9791eef5d7db2d0325dfa82e2745fec1d6e44e52687a84106e8441aa6bffc757` | `e317be98d55f7257e01dcbfcc445bdd61bae40b120744c6dd9d75c290d6f860f` | `748fbd825bb25a2c1ed8570635fb9448b73834d6ce08c7a1832b57d655a27192` |
| `000b491aed3d64dbf6587c98` | `author:chemometrics-spectroscopy`; `finish_block_authored_agy.jsonl:525` | `20292adcc96bff52f4b0560c5361534f49b67e47c551451fbb00a343786d3cf4` | `3f298f0362ab31f69e570105c224cabebe49f523be45e3f13b1baeda43cb0d44` | `ebd34926d1dee6611d78268784b6b1ca9687d8d3454f8c118d0830fdb170305a` | `314fe4e94c91e732b81e2c35b11e6e45743286fc03cefb75a8004aebb5bfb35e` | `7c89f96dc98b922f9197578b21c367a55919fb17b8002cf996f6efab9b3dc563` |
| `0002bc91539cd90ea3089201` | `author:insurance-actuarial`; `finish_block_authored_gpt56sol.jsonl:2257` | `9389751ed600ce01f65d9cbebcfff498f0501c1c50c44d272f294b4269facb4e` | `dd2ed2c724cae0d7363bb68a46196c75cbb0f69581b53f5e44635ecad1dcd636` | `d02c2933d48c3958807d9f144509943329aeb9310da952689c424bdec5a83d30` | `f1a951b1448a53942caf62443131646ba61b194b66d82293575707323deb5f1c` | `45b45e192930220793add17b5a0abb64ae783ca0e6482b409c6ca8e54c226f9d` |
| `0388bd30853cc3ce66901daa` | `author:agriculture`; `finish_block_authored_oxalpha-nous.jsonl:15` | `77163a185871fd9e38c4ff3ecc9b92857f1e290502c3a3ad2949df86bf05521c` | `47416e87fd66e2cb5758bea350c72444802bbf7f8edaa9d74644d22876109071` | `5fca77c5ca79a03eb486080964ca84d93c4db0c4307c0e0a66dcae50e7a1ba87` | `8702862d56a21130b867b07d074b9d4a96d0ee325924bc6361f9094e32409b36` | `46c1b9a1f8c9199d992ebce4e5c4efd642cbe4a9e0d41f2d8ae0789dfc9bc842` |
| `001093150ddd485f0255be66` | `author:agriculture`; `finish_block_authored_zai.jsonl:986` | `7b353dd7edccb750e4f10f1937191b1239fd16a5b769e68f0e9d8ba00474eec1` | `afa095a752a7a40c5c937fd29239ae8fda13b60e4f53eef3efa3bcfbbe8462e6` | `2bbb781d92a58a65e27263ff743b5f017382f1600dd58e340130c4986caa4168` | `ca6b59f83f38b2e37958a44b24d305f9ba960e2e92d1a6b0e60f387d1c74f338` | `23b0a89bc1c98541e7727d693d4cfed9a724a074988268a3333a2f3e7207c57b` |

The DAT-05 provenance `target_sha256` values also match the registry
`target_text` hashes (the body plus the protocol terminal marker):

| row ID | full target text SHA | DAT-05 provenance prompt SHA |
| --- | --- | --- |
| `2b578e5b4936158eaab12ee3` | `0ba0d5ee332f857462a5ce760b513e5c5f2fa09d85b08e927c59c9470a8051ce` | `17b3ca89f0caabf9aec249307d79df4c75a3521bcb378154ca587b65d488e960` |
| `000b491aed3d64dbf6587c98` | `8d5585ab32e62c0e57e091b6793da5e7f9051c9226dfc19ca3acf4195ad96363` | `e068a29460ca2d30cf0c73e3e5b524d6ad0ae540aa34db1b146546f5f1471223` |
| `0002bc91539cd90ea3089201` | `adfb89bdd790c604211365268e16ed13344b931aed968541fd6e7999b70e4e82` | `5ffbeca039f8900c1f75b1d8b523c21f594de7be67977753aba053039279e7ae` |
| `0388bd30853cc3ce66901daa` | `9d77765dae20fbe8398aa2fbc7b920587b8e5d74cc9657a2a8666350f8eda8a4` | `04d7447470b3ed87ba15641c9c91d71203352941c34001f928c77a0c54d85426` |
| `001093150ddd485f0255be66` | `80cc454b56aa0b7980d841431a90a3e3af1e8888e812920eb3032abfd385d6f1` | `f840d7f1acd0ee35dc19ea955ab71c3bba9b926a42fa67878e79d797ec139905` |

## SFT and RL selection results

`DAT-05-sft-registry.json` (receipt SHA
`6f0855ac891855622e6cc6e846e6be21ec041a2fa7c2eeb41bba2582fbe09b66`) admits
11764 train rows, including all five IDs. The SFT materializer's pinned
`train-token-rows.jsonl` contains the exact same JSON record bytes for all five.
The frozen 3000-step schedule contains each ID three times, for 15 scheduled
presentations in the 48000-draw plan:

| row ID | `target_start` / full sequence length / RL completion length | SFT scheduled draw indices (zero-based) | RL eligible row | RL selected-ID position (zero-based) | v5 source-draw positions (zero-based) |
| --- | ---: | --- | :---: | ---: | --- |
| `2b578e5b4936158eaab12ee3` | 267 / 294 / 27 | 1470, 18626, 35781 | yes | 6416 | 2160, 17320 |
| `000b491aed3d64dbf6587c98` | 166 / 587 / 421 | 7794, 24950, 42106 | no: `train_completion_only_over_cap` | — | — |
| `0002bc91539cd90ea3089201` | 169 / 564 / 395 | 10593, 27750, 44906 | no: `train_completion_only_over_cap` | — | — |
| `0388bd30853cc3ce66901daa` | 185 / 282 / 97 | 4001, 21157, 38313 | yes | 6411 | 5588, 20747 |
| `001093150ddd485f0255be66` | 348 / 379 / 31 | 12153, 29310, 46466 | yes | 5665 | 11033 |

The RL completion value is `len(input_ids) - target_start`, including the
stored final protocol EOS. The pinned RL-02 length filter computes exactly this
value and rejects values above 192. Thus the two missing RL rows are excluded
for their complete response lengths (421 and 395), while their prompt starts
(166 and 169) are below 2048. Their absence is a length admission result, not
evidence that the brace was repaired or that a source identity failed.

The three eligible IDs occur in the RL sidecar and selected-ID order. Each
sidecar row reports `context_has_target_or_reward_keys=false` and
`offline_static_source=true`; its prompt SHA equals the DAT-05 provenance and
its replacement-range document hash is the packet context hash. For example,
the sidecar rows are JSONL lines 6025, 6026, and 6027 for IDs
`2b578e5b4936158eaab12ee3`, `0388bd30853cc3ce66901daa`, and
`001093150ddd485f0255be66`, respectively. The sidecar carries context and
source/geometry identity; the target remains in the separate RL row file.

## What the consumers do with the boundary

The exact SFT source used by the accepted SFT attempt is pinned by the attempt
copy and has the same hashes as the execution source:

| source | SHA-256 | relevant behavior |
| --- | --- | --- |
| `experiments/training/campaign_sft_data.py` | `31e34c5a3a662724cf50dce1ea9e6a0d248c5f44bc0f5eb6b0a5b45894433104` | validates stored full IDs and the target-body prefix (lines 21–50), maps schedule IDs to row indices (51–64), and labels the stored IDs positionally (81–95) |
| `experiments/training/campaign_sft.py` | `671a1da97939b21962cfaedda0dc7c2c6f2f01369cb2834ef351fd26c38541f9` | uses a sequential sampler (198–206), creates the training dataset from `row["input_ids"]` selected by the schedule (334–335), and passes the full-text collator (363–372) |
| `experiments/training/campaign_rl_contexts.py` | `80b50f472de109f49fc77e43b64565b4528bd65fe127f12d3667c0859363fcb3` | applies the complete 2048/192 profile (399–446), then emits sidecar rows with target labels out of context (630–748) |
| `experiments/training/campaign_rl_data.py` | `86ebc3fe80a976e3c35a29057234b341c9e06060135a87382bbf8c915b1db91b` | joins selected rows to sidecar contexts and exposes `target_body_text` separately in the TRL envelope (138–158, 545–615) |
| `experiments/training/campaign_rl_train.py` | `78d27aa98cebc80292d1871a39821eee5a1a705b5a8f412ce270d1707d724443` | reward expected text is `target_body_text.split("\\n")` (1003–1010); generated output is parsed and compared with that body (1074–1096) |

None of these consumers reconstructs and applies a complete R document. The
SFT path carries the pre-tokenized sequence as labels, so it preserves the
stored target body. The RL path carries the stored target body as reward
metadata and compares parsed output to it; it does not add an outer brace.
The known completion adapter also applies `source_prefix + target_body` without
that brace, as established by the prior boundary receipt. No live editor
application was executed here, so an external editor layer could still have a
separate close-brace policy; there is no evidence for one in the reviewed
trainer/reward paths.

## Consumption status, limits, and recommendation

The accepted SFT-10 step-1000 review (`SFT-10-step1000-review.json`, receipt
SHA `f76d9b6d60af3edc2fbfc29aba6eb68bc778ca4794650bd2a7c8834fd1372b69`) proves
the same train-row and schedule identities reached a terminal full checkpoint
with `sampler_consumed_draws=16000`. SFT-08 selected that checkpoint (receipt
SHA `ca42502d9f2d140e243be2c4dec7287a5db4348b5b5470903d4fdba137137bf9`). All
five first scheduled positions above are below 16000, while every second and
third position is above it. Given the pinned sequential schedule and trainer
cursor, this supports the bounded inference that all five IDs had one intended
presentation consumed by step 1000 (5/5 first presentations; 0/5 second or
third); it is still not a per-row execution ledger.

RL-08-e did run and reached an accepted full step-140 checkpoint (receipt
`RL-08-e-terminal-full140-lead-acceptance.json`, SHA
`8f56c4028c856e9c50a262cd19042e56ddbda09262a5a11f5855333c20af3b33`). Its
verified source-draw cursor is 1120. The active v6 sequence retains the v5
sequence SHA above, and the three sampled eligible IDs have first v5 positions
2160, 5588, and 11033. Therefore none of these sampled occurrences falls in
the consumed positions 0–1119 at step 140 (0/3 sampled eligible IDs reached by
that cursor; the two length-excluded IDs have no RL draw). The checkpoint
receipt reports aggregate reward/generation counts, but no per-row reward-call
ledger, so this is a deterministic schedule/cursor inference rather than a
claim about an observed reward record for each ID.

The brace result is strong for this bounded chain: all five packet bodies,
DAT-05 registry rows, and SFT rows retain the same body hash and no outer `}`;
three RL rows retain it as well. It is not a population statement about all
4346 converted packets, and it is not a claim of training contamination.

Root should retain finish completion/application coverage as unresolved and
perform one bounded corrective action before relying on finish semantic or
reward coverage: either repair the shared finish boundary and regenerate the
dependent token/sidecar artifacts, or explicitly demonstrate a real editor
application that supplies the outer brace. Re-run the same five-row identity,
UTF-16 application, parser, and downstream hash checks after that decision.

This audit launched no source, data, model, trainer, editor, CUDA, SSH, network,
CLI, DEV, or final operation and changed no original artifact.
