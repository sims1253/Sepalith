# RL-08 independent review through update 50

The c continuation passes the bounded mechanical review for the completed DEV50 artifact and the first 25 post-step-25 training updates. The accepted status is `pass_mechanical_quality_pending`; this is an artifact and parser review, with no RL quality promotion claim. The continuation was still active after the reviewed window, so records after global step 49 were excluded from update-26 through update-50 acceptance.

The verifier is [`verify_rl08_step50.py`](verify_rl08_step50.py), and its complete compact result is [`RL-08-step50-independent-review.json`](RL-08-step50-independent-review.json). It uses only Python standard-library code plus the pinned pure campaign parser. It does not import torch, CUDA, Transformers, TRL, Unsloth, or a model, and it does not read the 52 MiB live telemetry file.

## Inputs and checkpoint

The run is `/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-c`, with recipe SHA256 `fc1de4c4b4e9b1da9617edbc99162d1db1df31080de95eed2bc75c695d5725c3`. The frozen source snapshot is `be406557c23d368fbb9869c0edf0c0c1099c7939d650a6bc32190fce53c330b9`; projected identity fields match the supplied identity `48028a843276d3ebe32b0b07fe9beb77e465dfc8aef920586eb468a8dbf9a0f2`. The canonical identity serialization is not present as a standalone artifact, so the supplied identity SHA is recorded rather than re-derived.

Both `archive/full/checkpoint-50` and `output/checkpoint-50` are present, marked `step=50`, `full=true`, and have identical manifest and state projections. The full manifest inventory contains 12 files. Every non-model file listed by the manifest was streamed in both locations and matched its recorded byte count and SHA256. `adapter_model.safetensors` (100,544,848 bytes) was deliberately not hashed under the review scope. Manifest SHA256 is `cfb1108fe415b547e8a4d19302f42f832fcd5563fb9871e5d68db355778f94f0`; campaign-state SHA256 is `34d40283eba95b562a9b530237ea74515757aa07d19dc81b03b7510f716eb76e`.

The checkpoint state is at source draw cursor 400, consumed rows 12,800, selected ID index 8,243, with sampler geometry `batch=32`, `per_device=4`, `gradient_accumulation=8`, and `steps_per_generation=8`. Its `resume_validation` text still requires the matched interruption/resume receipt; the state itself is complete and identity-bound, but this review does not turn that text into a new resume proof.

## Training window

The review selected generation rows with `global_step` 25 through 49. These correspond to optimizer updates 26 through 50 after restoring step 25: 25 updates × 32 rows = 800 generation records and 800 reward records. Every update has four generation calls, eight groups, four candidates per group, and the expected stored microstep `(global_step - 25) * 8`. Generated ID hashes and token counts join the reward records exactly; all IDs are within the 130,560-token vocabulary and all counts are within the 192-token limit.

The 200 source IDs obtained from every fourth reward row equal source sequence entries `[200:400]`, so the continuation joins the step-25 cursor to the step-50 cursor exactly. Each source has four candidate records, all 200 IDs resolve in the admitted rows file, and family/expected-operation metadata agrees for every candidate. The window has 751 canonical EOS records and 49 length-cap records; those are retained as runtime denominators, not hidden as failures.

Gradient records for rows 25 through 49 map to finite optimizer updates 26 through 50. Every record has 588 present gradients and finite norm; nonzero tensor counts are either 588 or zero. The zero-gradient updates are 30, 33, 34, 36, 45, 46, and 47. The checkpoint trainer history independently reports `reward_std=0`, `frac_reward_zero_std=1`, and `grad_norm=0` for exactly those updates. Zero-variance updates are recorded for root review and are not classified as nonfinite or protocol failures.

## DEV50

The pinned panel has 75 unique IDs and SHA256 `b16c5892635d6e9a5e9e789677057c85a5e875c907c163e91d39f25bc6ad3d21`. The complete cases artifact is SHA256 `c12f784366172c7dda63d38fd6ade3f072e19221678497474a3a2788ccb05833`; its IDs exactly equal the frozen panel and it reports `step=50`, `status=complete`.

Re-running the pinned parser produces the stored classifications exactly:

| Measure | Count |
|---|---:|
| Cases | 75 |
| Edit denominator | 43 |
| Strict no-op denominator | 32 |
| Protocol-valid | 70 |
| Exact region | 50 |
| Strict no-op correct | 25 |
| Predicted no-op | 27 |
| Suggestions | 43 |
| Cap hits | 5 |
| Strict no-op false suggestions | 5 |
| Exact edits | 25 |

Compared with step 25, the only classification change is `f43de3e77f2d92ed7b223464` in `finish_block`: `protocol_valid false → true`, `cap_hit true → false`, and `failure missing_exact_terminal → null`. There is no exact-region gain. Per-family counts are: finish block 3/6 protocol-valid, 3 caps; no-op 30/32 protocol-valid, 25 predicted no-ops, 2 caps; format propagation 7/7 and 4 exact; na.rm propagation 6/6 and 5 exact; pipe rewrite 8/8 and 8 exact; rename propagation 8/8 and 8 exact; roxygen drafting 8/8. The maximal retained DEV prompt has 2,619 prompt tokens and is protocol-valid, exact, and uncapped.

The recipe and entry audit bind the expected `model_load_max_seq_length=4096`, RL prompt/context/completion limits `2048/2240/192`, BF16 load, CUDA fraction `0.75`, 294 adapter attachments, 25,116,672 trainable parameters, renderer `zeta2-prm03-v1`, and the explicit `campaign_eval:development_evaluator` wrapper with 75 cases, 512 new-token cap, and 4096-token parameters.

Run command:

```text
python3 docs/campaign/work/rl-step50-review/verify_rl08_step50.py
```

The only remaining gates are the continuing c terminal and later same-identity resume/DEV75 review, plus root's quality decision. This packet does not promote the checkpoint or infer R semantic validity from the exact-region counts.
