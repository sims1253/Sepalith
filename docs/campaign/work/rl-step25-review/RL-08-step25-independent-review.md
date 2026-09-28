# RL-08 update-25 independent review

This CPU-only review is mechanical artifact evidence for the completed
`RL-primary-p2-mb4-full5-b` continuation. It does not promote the adapter. The
verifier is [`verify_rl08_step25.py`](verify_rl08_step25.py); it projects the
74 MiB/148 MiB checkpoint JSON through a small `jq` allowlist, hashes files in
streaming mode, and never imports a model, framework, or CUDA.

The exact command was:

```text
python3 -m py_compile docs/campaign/work/rl-step25-review/verify_rl08_step25.py
python3 docs/campaign/work/rl-step25-review/verify_rl08_step25.py \
  --run-root /home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-b \
  --recipe docs/campaign/work/main-rl-preparation/primary-mb4-full5-b.recipe.json \
  --sequence docs/campaign/work/rl-pool/source-row-draw-sequence-v6-interleaved-mb4.json \
  --panel /mnt/e/sepalith/campaign-20260915/data-work/DAT-07-final-evaluator-cases.jsonl \
  --json-out docs/campaign/work/rl-step25-review/RL-08-step25-independent-review.json
```

The command returned exit 0 with `pass_mechanical_quality_pending`; checkpoint,
training records, DEV reclassification, and runtime all passed with zero
failed checks. The result JSON is the detailed receipt for this review.

## Identity and checkpoint

- Frozen source snapshot: `be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9`.
- Recipe SHA-256: `ea27572c81d46c7a36a54a5ed05def7fb9decd044abec103586d5bd2cbcde56d`.
- Manifest identity contract: `48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2`.
- Parent: merged-SFT manifest `1230e35f3d4adc3a8b23ba4cec0c8b55620d8e83e671cde8c66c3dcd77c5af12`, weights `499b7fdadfa701c04a4fba8f1717eb3ce02acc453237718562f002486d09840d`, revision `8dc5f6055b90fe4b9422340810b270b9569f37f3`.
- Full checkpoint manifest: `archive/full/checkpoint-25/campaign-manifest.json`, 74,009,713 bytes, SHA-256 `7c135c7b80ed90d1d893b4ad8165fd2490346868fa5d697868f580709fee3ca4`.
- Full checkpoint state: 148,011,284 bytes, SHA-256 `a1d5a7f2f097779ebe38e0e38559f7fc0492ebdf8d83d46c2688d043a974db35`.
- `archive/full/checkpoint-25` and `output/checkpoint-25` each contain the same 12 manifest-listed files; every recorded byte count and SHA-256 matched, and both manifests are byte-equal. Each full directory is 385,861,710 logical bytes. The detailed JSON records all 12 file hashes.
- The light adapter archive is complete at step 25 (`full=false`), 258,566,444 logical bytes; its manifest SHA-256 is `b4b02a4037de0fc34347538fc88ec77a0b9b3580dd3f9a24430ae039b4a379e5`, and the adapter bytes are SHA-256 `cfd933d48bcc1e9e56ec681a64d870777985b615a3a2075a31551f119feae03d`.
- Full-state sampler is at `source_draw_cursor=200`, `consumed_rows=6400`,
  `current_index=6400`, `consumed_prompt_copies=1600`, and
  `selected_id_index=1198`; geometry is generation batch 32, per-device batch
  4, accumulation 8, steps per generation 8. The source schedule and sequence
  hashes are `2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132`
  and `dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6`.

The known selected-ID JSON has 8,440 unique rows and SHA-256
`24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d`.
The eligible rows and context sidecar independently hashed to
`e54bca71d96cf29f3a75d3e084b601e25edce8fa513cdb2bb61c730a7385f602` and
`6996f89d399e5e4dd5a5dafa49e92e8a49178fbecf15b2d6e769532f1afe703f`.

## Continued training records

The resumed continuation has 640 generation rows and 640 reward rows over
`global_step` 5 through 24, 32 rows per buffer. Each buffer has four calls,
two groups per call, eight groups, and four candidates per group. Its 160
collapsed source IDs exactly match frozen sequence rows 40 through 199, with
four contiguous candidate records per source ID. Generation/reward output
hashes and token counts join exactly.

Candidate reward family counts are format propagation 160, rename propagation
160, no-op 128, finish block 104, pipe rewrite 64, roxygen drafting 16, and
na.rm propagation 8. Expected operations are 512 `replace` and 128 `no_op`;
the realized operation fields are 471 `replace`, 130 `no_op`, and 39 unset
because 39 length-capped rows lack a canonical EOS. Training termination is
601 EOS and 39 length; reward records agree with 601 EOS/protocol-valid and 39
`missing_canonical_eos`/cap rows.

The 20 durable gradient records map to optimizer updates 6 through 25. All are
finite, with 588 present gradients and 588 trainable tensors. Update 11 has
zero nonzero adapter tensors and norm 0.0; its trainer metric also has
`frac_reward_zero_std=1.0` and reward standard deviation 0.0. This is retained
as an observation, not hidden as a nonfinite failure. The other records have
positive norms. Telemetry independently records optimizer steps 6 through 25,
with `lead_decision` at step 25.

## DEV75 reclassification

The complete artifact is
`archive/evaluations/cases-step-25.json`, SHA-256
`fdfd724326485b0d5f92e3ad3b3b255bdb072735858a15d1593f9e3b1402dd96`. The
pinned panel has exactly 75 IDs in exact order and SHA-256
`b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`. Running
the frozen `campaign_protocol` parser over every stored raw output and token
sequence produced no disagreement with the stored row flags.

The step-25 denominators are 43 edits and 32 strict no-ops. Counts are 50
exact regions, 69 protocol-valid rows, 27 predicted no-ops, 25 exact edits,
42 suggestions, 6 caps, 25 strict no-op correct, and 5 strict no-op false
suggestions. The maximal case
`dat07-existing-719cd49683667d0fb86fb2fa` is retained at 2,619 prompt tokens,
12 generated tokens, canonical protocol, and no cap.

Against the same-identity step-5 DEV artifact (SHA-256
`8a84b9737934cfb6c2d4b5374040f2193da903d665c36d83acbc9ba164b695e6`), exactly
two case outcomes changed:

| ID | family | step-5 → step-25 change |
| --- | --- | --- |
| `dat07-existing-c54ed09efae0ca38cff22981` | roxygen drafting | protocol invalid/cap → protocol valid/no cap; exact remains false |
| `dat07p-48b35fcee94a6d767a1acf49e495` | format propagation | exact edit → predicted no-op and not exact |

All no-op family counts remain 25 exact/predicted no-ops, while format exact
edits change 5→4 and roxygen protocol-valid rows change 7→8.

## Process and host evidence

The immutable `supervision.json` remains separate from
`supervision.entry-result.json`. The distinct entry result is `lead_decision`
at step 25 and points back to the supervision receipt. Host supervision ended
at `2026-09-12T20:49:38.909634+00:00` with child exit 0, no reason, and
2,087.2955662379973 seconds.

The host recorder captured 130 samples. Available memory reached a minimum of
14,245 MiB and ended at 18,062 MiB; the final committed bytes were
77,564,342,272 of a 137,353,273,344-byte commit limit. Historical samples
contain page-read/input spikes (max 1,509 page reads/s and 11,025 pages
input/s) but zero page output and no recorded driver events. This review does
not attribute causality to those samples.

The next same-identity resume from cursor 200 to update 100 remains a separate
root-owned state/output equality gate. No quality or promotion claim is made
by this receipt.
