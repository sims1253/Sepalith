# RL-07 full5 independent review

Review time: 2026-09-12. The review is CPU-only and framework-free. It reads the known RL-primary-p2-mb4-full5-a artifact paths, hashes checkpoint files in streaming mode, parses the 160 generation/reward rows and five gradient rows, and skips the 52 MiB identity record in telemetry. It does not import a model or torch and does not read sealed final artifacts.

The verifier is [`verify_rl07_full5.py`](verify_rl07_full5.py). It was syntax-checked and run with:

```text
python3 -m py_compile docs/campaign/work/rl-full5-review/verify_rl07_full5.py
python3 docs/campaign/work/rl-full5-review/verify_rl07_full5.py \
  --run-root /home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-a \
  --recipe docs/campaign/work/main-rl-preparation/primary-mb4-full5-a.recipe.json \
  --sequence docs/campaign/work/rl-pool/source-row-draw-sequence-v6-interleaved-mb4.json \
  --panel /mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl \
  --json-out docs/campaign/work/rl-full5-review/RL-07-full5-independent-review.json
```

The final mechanical result is `pass_mechanical_quality_pending`. The full checkpoint, training records, load audit, development panel, and terminal receipt all pass the independent checks. This result does not promote the model; root retains the quality decision.

## Identity and load

- Frozen source snapshot: `be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9`.
- Canonical recipe SHA-256: `c85b7e3e9fd7927d429a6d45085e98d3f5bb8b06047e468f5bd7dc9190939969`.
- Manifest identity: `48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2` (identity fields independently checked against the frozen contract).
- Source schedule/sequence: `2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132` / `dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6`.
- Parent is the complete merged-SFT manifest `1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12`, with merged weights `499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d` and base revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`.
- Load audit passes BF16, model-load request/observed capacity 4096, allocator fraction 0.75, LoRA rank/alpha 16/16, 294 attachments, and 25,116,672 trainable parameters. The tokenizer contract is verified as BOS 0, EOS/PAD 1, native EOG `[1, 130073]`, vocabulary 130,560.
- The checkpoint-5 adapter SHA is `ab2caa03ed9ab682d64af220f648fe9be6f1ac8f9c1da28f3e980d29da36be3c`, byte-equal to the prior 4/8 light5 adapter supplied for comparison. This is a byte comparison only and carries no quality conclusion.

## Full checkpoint and state

`archive/full/checkpoint-5` is `full=true`, `step=5`. Every one of the 12 manifest-listed files was streamed and matched its recorded byte count and SHA-256. Every corresponding file in `output/checkpoint-5` matched the archive byte-for-byte by size and SHA; the two checkpoint manifests also match. The full state contains optimizer, scheduler, RNG, sampler, adapter, tokenizer, trainer state, and training args. The independent state checks found:

- `source_draw_cursor=40`, `consumed_rows=1280`, `current_index=1280`, `consumed_prompt_copies=320`, `selected_id_index=5460`;
- sampler `shuffle=false`, candidate count 4, buffer reuse 8, 8 prompt groups, and `source_draws_bound=true`;
- sampler geometry batch 32, per-device batch 4, gradient accumulation 8, and 8 steps per generation;
- trainer state global step 5, max steps 3000, train batch size 4, and save/evaluation steps 5/500.

The logical footprint is 384,311,155 bytes for each full checkpoint directory, including its 74,009,711-byte manifest and 148,011,281-byte campaign state. The light adapter archive is 258,566,442 bytes. The known output tree is 760,798,598 bytes; full archive plus output checkpoint plus light archive is approximately 1.03 GB before filesystem deduplication. These are metadata measurements only.

## Training records

The verifier found 5 updates × 32 generation rows, 160 reward rows, and 40 source IDs with four contiguous candidates each. Reward IDs collapsed every four rows match the first 40 rows of the frozen v6 sequence. Generation output hashes and generated-token counts match the ordered reward rows.

Generation geometry is four calls × two groups per call, eight groups × four candidates, with no truncation after terminal beyond the recorded right-padding counts. Terminal reasons are 151 `eos` and 9 `length`. Reward records contain 151 canonical-EOS/protocol-valid rows, 9 `missing_canonical_eos` rows, and one additional canonical-EOS row that also reports a cap hit; aggregate cap-hit count is 10.

The actual reward stream has 119 `replace`, 32 `no_op`, and 9 missing operation values caused by those canonical-EOS failures. Expected operation denominators are 128 replace candidates and 32 no-op candidates. Family counts over 160 candidates are format propagation 40, rename propagation 40, no-op 32, finish block 24, pipe rewrite 16, roxygen drafting 4, and na.rm propagation 4.

All five gradient records are finite and nonzero. Each reports 588 trainable tensors and 588 present gradients; norms are `0.0843905563`, `0.0685153340`, `0.17677003597`, `0.10188919489`, and `0.12224085149`.

## Development panel

The exact pinned 75-case panel SHA is `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`. The complete step-5 case file has the exact panel ID order and denominators of 43 edits and 32 no-ops. Raw result counts are:

- 26 predicted edits/exact suggestions, 26 predicted no-ops, 51 exact regions;
- 68 protocol-valid rows and 7 `missing_exact_terminal` rows;
- 7 cap hits;
- 25 strict no-op correct and 5 strict no-op false suggestions.

The maximal retained DEV case `dat07-existing-719cd49683667d0fb86fb2fa` has 2,619 prompt tokens, 12 generated tokens, canonical protocol, and no cap. The evaluator interpretation remains diagnostic and explicitly makes no R semantic-validity or final-release claim.

## Terminal and remaining gate

The immutable supervision receipt remains separate at `supervision.json` and still records that command acceptance/completion requires review. The distinct `supervision.entry-result.json` reports `status=lead_decision`, `step=5`, and points to the full checkpoint. The supervised process is gone and telemetry independently records optimizer step 5 with `stop_reason=lead_decision`.

The artifact passes RL-07 mechanical review. A same-identity continuation/resume to step 25 still requires its own byte/semantic comparison of adapter, optimizer, scheduler, RNG, sampler, source draws, outputs, and rewards. The current receipt does not claim that proof or make a quality/promotion decision.
