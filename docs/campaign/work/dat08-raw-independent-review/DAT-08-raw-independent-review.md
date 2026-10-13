# DAT-08 raw-source independent review

This is an independent CPU review of the two DAT-08 preparation components.
It uses only the four explicit fixtures hard-coded in
`raw_source_tokenization.py` and the three actual source fixtures named by
the builder tests. It did not open candidate, DEV, held-out, sealed-final, or
campaign row files, and it did not launch a model or write a training dataset.

The reviewed inputs matched the worker receipt identities:

- `raw_source_case_builder.py`: SHA-256
  `2e617f5c14b04a3a329362665d12f18ff836f88a4f029719a9a5d0ab0e93999b`
- `raw_source_tokenization.py`: SHA-256
  `81821054e3c22a9d8f80cdc07d95b7350b4e35eefa3713fd69ac368aa1304c39`
- builder tests: SHA-256
  `4a3efe83c506bf2135141351814b61f2ff0eed52906704a07d4ab15c08ee3a41`
- tokenizer tests: SHA-256
  `1f2b5859d78f74bfdcfc22f372454c6aa31413b938fddf591a3f372f236f0927`
- pinned canonical scenarios: SHA-256
  `cccf8ddfff0ae1f64a0113c9612386227df66f9fff4701608c2320bd8eb0250c`
- pinned PRM-03 protocol: SHA-256
  `5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156`

The local tokenizer was loaded by `/usr/bin/python3` 3.10.12 with
`transformers` 5.13.0.dev0 and `tokenizers` 0.22.2. Its two pinned files were
hashed before `AutoTokenizer.from_pretrained(..., local_files_only=True,
trust_remote_code=False)` loaded them. No model weights were imported.

## Independent results

The sequential commands below passed with `CUDA_VISIBLE_DEVICES=''` and
`PYTHONDONTWRITEBYTECODE=1`:

```text
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest -v docs/campaign/work/final-constructor-preparation/test_raw_source_case_builder.py
PASS: 6 tests, 2.287 s

env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest -v docs/campaign/work/final-constructor-preparation/test_raw_source_tokenization.py
PASS: 3 tests, 9.566 s
```

The independent four-row run called the real builder, pinned protocol, local
tokenizer, `build_training_row`, and `validate_training_row`. It confirmed
the following measured geometry. Counts are one explicit row per family;
they are not population estimates.

| family / case | source bytes | event old→new lines | target old lines | prompt tokens without/with BOS | target body + terminal = total | input tokens | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rename_propagation` / `heplots-rename` | 2,117 | 1→1 | 1 | 335 / 336 | 10 + 5 = 15 | 352 | valid |
| `pipe_rewrite` / `timetk-pipe` | 7,703 | 1→1 | 1 | 339 / 340 | 19 + 5 = 24 | 365 | valid |
| `format_propagation` / `rempsyc-format` | 3,426 | 3→1 | 6 | 359 / 360 | 38 + 5 = 43 | 404 | valid |
| `na_rm_propagation` / `lares-na-rm` | 20,109 | 1→1 | 1 | 261 / 262 | 12 + 5 = 17 | 280 | valid |

The source SHA-256 values were checked from the bytes read by the independent
run:

```text
heplots  cb44daa9d2cc33cf0456f7a175b7716d5e8d97687eb946701c612c57ccc0db2b
timetk   2b95fc5bde00be623a7108edb0cf60e8309c44d524ffaaddbf479bcae1c4adf0
rempsyc  723b0c365bcb9346a321f7107cd31c71e4a5c54b5baffd6ff9023b06a56b7a6c
lares    7914aa2aaf7c689083007825ed071422ad4872f21d8570b883cd287a6ded4e2a
```

For `rempsyc`, the named archive member and normalized counterpart were
checked against the worker receipt hashes. All four rows had one BOS (0), one
terminal EOS (1), exact target start, and no padding or truncation. The
aggregate was `attempted=4`, `full_row_contract_valid=4`,
`prompts_over_4096=0`, `prompts_padded_or_truncated=0`, and
`final_admission_allowed=0`.

The target-free check compared each emitted prompt with its separately
derived target. For all four rows, the serialized target text was absent from
the prompt, `region_new` was absent from the context mapping, and the full
source byte string was not embedded in the prompt. This checks the current
four fixtures; it is not a semantic guarantee for arbitrary caller-supplied
fixtures.

## What is actually checked

The builder computes and verifies the source and normalized-source hashes,
loads the pinned canonical scenario and protocol sources by hash, parses the
source before and after the event and target edits, locates the event and
target regions uniquely, preserves EOL/terminal-newline geometry, constructs a
`PromptContext`, renders with the pinned PRM-03 renderer, and round-trips the
derived target through the pinned output parser. The tokenizer component
checks both tokenizer file hashes and tokenizer identity, calls the real
`build_training_row` and `validate_training_row`, and rejects rows over the
4,096-token cap without truncating them.

The result booleans are outcome labels after those checks. Several are literal
`True`/`False` fields written only after a failure would already have raised:
`before_r_parse`, `after_event_r_parse`, `post_edit_r_parse`, geometry flags,
`prompt_rendered_exact`, `target_roundtrip_valid`, `row_contract.valid`, and
the scope/admission flags. They should be read as fail-closed control flow,
not as independently replayable evidence. In particular,
`final_admission_allowed`, `dev_admission_allowed`, and `dat03_verified` are
deliberately hard-coded false; no disjointness or final semantic admission is
performed here.

## Findings and remaining coverage

1. **Positive source-root gate is missing.** `_safe_input_path` and
   `_guarded_path` require an absolute regular file and reject path components
   such as `final`, `sealed-final`, `heldout`, and `candidate-references`, but
   they do not require the path to be below an approved source root. The
   public builder API can therefore read any caller-supplied absolute regular
   file whose path avoids the denylist; the tokenizer API can likewise accept
   custom fixture mappings. The default four-fixture CLI path is bounded by
   its constant list, but this is insufficient as a future held-out/final
   boundary. A release caller should use a positive allowlist of exact source
   roots and fixture identities, or reject custom fixture overrides.

2. **Output path is caller-controlled.** Both `_write_json` helpers create
   parent directories and replace the requested output path without a
   positive output-root check or `fsync`. This did not mutate campaign state in
   this review because no CLI output was directed there, but a later caller
   should constrain output to a fresh preparation directory and make the
   durable/no-overwrite policy explicit.

3. **Admission is intentionally incomplete.** The four rows cover four of
   the seven admitted family concepts one row each. `no_op`, `finish_block`,
   and `roxygen_drafting` have no raw-source construction in this packet.
   The builder unit tests directly exercise rename, pipe, and format; the
   tokenizer run also exercises the explicit `na_rm` row. No final readiness
   or quality claim follows from this coverage.

The safe release disposition is **engineering fixture contract verified;
final admission pending**. Before any held-out or final construction, close
the positive source-root and output-root gates, recompute disjointness and
semantic source-case evidence, then obtain the separate tokenizer/model/
harness receipts required by the campaign.
