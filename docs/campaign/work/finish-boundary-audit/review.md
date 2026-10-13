# DAT-08/RL-08 finish-boundary audit

This bounded review checks whether the missing outer closing brace seen in the
synthetic raw-family v3 case is specific to that fixture constructor or is
also present in converted TRAIN finish packets. It samples exactly the first
five packets of the retained DAT-04B converted TRAIN packet, asserting each
row is `split=train_group` and `family=finish_block`. The packets are
source-derived simulations; they are not evidence that a model training run
consumed them.

## Assembly boundary

The pinned finish extractor at
`/home/m0hawk/.t3/worktrees/Sepalith/tuesday-execution-20260912/experiments/synthetic-data/finish_block.py`
(SHA-256
`47759356b85eb9b48e0461e9265a9fa5a272df14de87747a63e62304be48a3fb`) defines
the boundary as follows:

- Signature rows build `prefix` through the opening `{`, then set
  `body_lines = node_text(src, body)[1:-1]` (lines 89–97). Both braces are
  stripped from the body text; the outer closing `}` is absent from `target`.
- Mid-body rows set `tail = src[cut:body.end_byte - 1]` (lines 99–107), which
  again ends immediately before the outer closing `}`.

The pinned batch assembly
`experiments/training/campaign_completion_batch.py` (SHA-256
`a7a989f491f168154a3274b6678595c15b4d434be6e7f346521483e0d76035f4`) performs
an R fragment check as `prefix + target_text + "}"` (lines 338–348). Its
summary explicitly states that the outer closing brace is never added to
`target_body` (lines 632–637). That appended brace is parser/reconstruction
framing, not an emitted completion character.

The current pinned adapter
`experiments/training/campaign_admission_completion.py` (SHA-256
`fafb852f31e5e59f003c3ccc0fb34919752020fb6c231a09d77b3ea158f173cc`) preserves
the same boundary. `_finish_replacement` joins the target lines without
adding a brace (lines 798–805), and `_verify_finish_literal_splice` accepts
only `source_prefix + source_target_text` (lines 808–837). The suffix route
then emits that replacement through the whole-current-line range (lines
977–1035). Thus the brace omission is a shared finish-block conversion
contract, not a condition introduced only by raw-family-v3.

## Five converted TRAIN packets

Input packet:

`/mnt/e/sepalith/campaign-20260915/data-work/DAT-04B-completion-batch.jsonl`
(SHA-256
`42743e70dbde54dd0f4adab42ebd0c2b0a0e7d3a4c4596752ffca272b20adc25`). The
batch receipt is
`docs/campaign/receipts/DAT-04B-completion-batch.json` (SHA-256
`7cea4a7d2e43d66eebb77a483b7bdd5b29c2363aea8ce8164f2b32fe8009bad5`). It
reports 4346 converted finish packets from `train_group`; no model or
tokenization admission was performed.

For each sampled packet, the exact emitted replacement was applied to its
source-derived simulated pre-edit document using the packet's UTF-16 range.
The pinned tree-sitter R parser then checked the resulting document. `after+}`
is a diagnostic reconstruction check; it is not counted as application
validity.

| packet line | row ID | variant | post bytes/chars | post SHA-256 | post parses | post + `}` parses |
| ---: | --- | --- | ---: | --- | :---: | :---: |
| 1 | `2b578e5b4936158eaab12ee3` | `mid_body` / authored | 475 chars | `a37b3824eecf624aaf62710c7b28f38478ad4977c5c3e7c475fa8b86465e8b5e` | no | yes |
| 2 | `000b491aed3d64dbf6587c98` | `signature` / authored_agy | 1729 chars | `445f08e4a877c7f93df94b6e3178e0de70d1fcb3d0d17efd7888d29dd5def889` | no | yes |
| 3 | `0002bc91539cd90ea3089201` | `signature` / authored_gpt56sol | 1587 chars | `6722b2ccff8ba14875faaa88372b44e7752b52b02941db359e58cd1a493e99a0` | no | yes |
| 4 | `0388bd30853cc3ce66901daa` | `mid_body` / authored_oxalpha-nous | 555 chars | `6139a683bbbc31245535d35a17b76569a244a47bd98d46374be05ada6f8800ea` | no | yes |
| 5 | `001093150ddd485f0255be66` | `mid_body` / authored_zai | 646 chars | `d56b4d972bffa680c016f8943301156e761787bb79057abc1768772000894a89` | no | yes |

All five packets report `source_constructor=finish_block_v5_prefix`,
`target_convention=suffix`, `literal_source_splice_verified=true`, and
`outer_closing_brace_in_label=false`. Their emitted target bodies end with the
adapter's terminal empty line, but none carries the outer brace. The first
five sample therefore directly shows the same missing brace in both
`signature` and `mid_body` converted TRAIN packet output.

The sampled source identities are retained in the packet and were not used to
read the source corpus:

| row ID | source file:line | source SHA-256 | raw-line SHA-256 |
| --- | --- | --- | --- |
| `2b578e5b4936158eaab12ee3` | `finish_block_authored.jsonl:3` | `c9476a8f9d7c36a9cf91e769434db9245a9a9309e9cd215090884a0131b78734` | `9791eef5d7db2d0325dfa82e2745fec1d6e44e52687a84106e8441aa6bffc757` |
| `000b491aed3d64dbf6587c98` | `finish_block_authored_agy.jsonl:525` | `3f298f0362ab31f69e570105c224cabebe49f523be45e3f13b1baeda43cb0d44` | `ebd34926d1dee6611d78268784b6b1ca9687d8d3454f8c118d0830fdb170305a` |
| `0002bc91539cd90ea3089201` | `finish_block_authored_gpt56sol.jsonl:2257` | `dd2ed2c724cae0d7363bb68a46196c75cbb0f69581b53f5e44635ecad1dcd636` | `d02c2933d48c3958807d9f144509943329aeb9310da952689c424bdec5a83d30` |
| `0388bd30853cc3ce66901daa` | `finish_block_authored_oxalpha-nous.jsonl:15` | `47416e87fd66e2cb5758bea350c72444802bbf7f8edaa9d74644d22876109071` | `5fca77c5ca79a03eb486080964ca84d93c4db0c4307c0e0a66dcae50e7a1ba87` |
| `001093150ddd485f0255be66` | `finish_block_authored_zai.jsonl:986` | `afa095a752a7a40c5c937fd29239ae8fda13b60e4f53eef3efa3bcfbbe8462e6` | `2bbb781d92a58a65e27263ff743b5f017382f1600dd58e340130c4986caa4168` |

## Interpretation

The missing brace is present in the accepted converted TRAIN packet contract
for the five sampled rows, so it is not confined to the new raw-family-v3
construction. The evidence supports a structural application-validity gap in
the shared `finish_block_v5_prefix` / suffix route:

```text
actual emitted application:       source_prefix + corpus_target
R reconstruction diagnostic:     source_prefix + corpus_target + "}"
```

The five post-application documents all fail the pinned R parser, while the
diagnostic reconstruction with one appended brace parses. This is independent
of model output and does not imply training contamination: the packet receipt
marks these rows as source-derived simulated pre-edit windows, and no model
training run was inspected.

The sample is bounded. It does not establish that every one of the 4346
converted packets is parser-invalid after an editor splice, and it does not
establish whether any downstream trainer or serving layer separately appends a
brace. It does establish that the current converted packet representation and
the adapter's own application-shaped result omit the brace in representative
TRAIN rows. Root should treat finish completion/application coverage as
unresolved until a real editor application contract is demonstrated or the
boundary is repaired and revalidated.

## Validation and scope

The CPU-only driver
`audit_train_finish.py` (SHA-256
`98f852a87f88bafb6c6a8e6d567b1d22256b2eaedf9a1d4ac7937167d095a73a`) passed
all five row identity, constructor, UTF-16 splice, and parser assertions. It
read only the first five lines of the retained converted TRAIN packet. No
model bytes, tokenizer, final/DEV content, source corpus, GPU, SSH, network,
or CLI state was accessed. No implementation or data files were changed.
