# RL-03 supervised entry preparation

This packet adds the CPU-verifiable boundary that turns the admitted
`campaign_rl_train.build_live_trainer` seam into one supervised RL attempt.
The entry module remains framework-free at import and preflight.  It performs
all file, identity, parent, schedule, output, archive, and filesystem checks
before importing Torch, Transformers, TRL, Datasets, or Unsloth.

## Parent contract

The RL recipe must contain this exact wrapper:

```json
{"parent_manifest": {"path": "/absolute/merged-sft-parent.json", "sha256": "<64 lowercase hex>"}}
```

The referenced JSON must have
`schema_version=sepalith.merged-sft.parent-manifest.v1`, `status=accepted`,
`kind=merged_sft`, and these fields:

```json
{
  "merged_model_path": "/absolute/local/merged-sft-directory",
  "base_model_revision": "<nonempty pinned revision>",
  "sft_identity": {"<accepted SFT identity>": "<value>"},
  "merged_weights_sha256": "<64 lowercase hex>",
  "weight_inventory_sha256": "<64 lowercase hex>",
  "weight_inventory": [
    {"path": "model.safetensors", "bytes": 123, "sha256": "<64 lowercase hex>"}
  ],
  "tokenizer": {
    "tokenizer_json_sha256": "<64 lowercase hex>",
    "tokenizer_config_sha256": "<64 lowercase hex>",
    "vocab_size": 130560,
    "bos_id": 0,
    "eos_id": 1,
    "pad_id": 1,
    "native_eog_ids": [1, 130073]
  }
}
```

`weight_inventory` is sorted by relative path and must enumerate every
non-symlink file under `merged_model_path` with a weight suffix
(`.safetensors`, `.bin`, `.pt`, or `.pth`).  Every byte count and SHA256 is
recomputed.  The complete inventory digest is the SHA256 of the sorted JSON
entries with exactly `path`, `bytes`, and `sha256`; it must equal
`weight_inventory_sha256`.  For one weight file, `merged_weights_sha256` is
the conventional file-content hash (the inventory hash is also accepted for
fixtures); for sharded weights it is the complete inventory hash.  The model directory
must also contain `config.json`, `tokenizer.json`, and `tokenizer_config.json`,
whose tokenizer hashes are checked.  The recipe identity must repeat the
manifest hash, aggregate weight hash, base revision, and SFT identity.  This
keeps the profile-only Midtrain path from satisfying the RL parent gate.

## Live sequence

`campaign_rl_entry.run` first requires the launcher's soft and hard UTC
deadlines, repeats the CPU preflight, checks the one-device `nvidia-smi`
occupancy guard (one GPU, at most 4096 MiB used), and creates fresh output and
archive directories.  Both destinations and the telemetry file must resolve
to native ext4; v9fs/NAS paths such as `/mnt/h` are rejected before training.

The live loader calls `FastLanguageModel.from_pretrained` only after that gate,
with the verified merged model path, `dtype=torch.bfloat16`,
`load_in_4bit=False`, and `trust_remote_code=False`.  It loads a separate local
reference tokenizer and runs `restore_pinned_tokenizer_contract` against every
stored selected prompt.  It then attaches LoRA r16/a16 to the seven pinned
modules and requires 294 attachments and 25,116,672 trainable parameters.

The existing builder supplies the fixed-ID generation adapter, BNPO/group
reward, deterministic selected-ID sampler, control callback, and full-state
checkpoint callback.  The entry adds an exact post-generation tokenizer/config
assertion, an fsynced per-attempt reward JSONL sink, and the train-begin
`campaign_sft.restore_trainer_eog_alignment` callback.  The latter repairs the
known Transformers EOS alignment (`[1,130073]` becoming `1`) immediately before
the first optimizer update and records its audit.  A post-trainer assertion is
performed before this callback is installed.

Resume accepts only an existing `campaign_checkpoint` full checkpoint whose
identity exactly matches the recipe.  A resumed attempt still requires fresh
output, archive, telemetry, and launcher receipt paths.  After `trainer.train`
the entry verifies the terminal archived full checkpoint and final tokenizer
contract before writing `terminal.json`.

The installed TRL 0.24.0 GRPO path was checked against the profile's policy
memory assumption. It calls the model with `labels=None`, sets
`logits_to_keep = completion_ids.size(1)`, and, when the model advertises that
keyword, passes `logits_to_keep + 1` into the forward before dropping the final
position. It then computes `selective_log_softmax` over the trailing padded
completion width. This avoids a full FP32 `[batch, context, vocab]` tensor,
but it is not identical to the profile helper: the profile boolean-selects
only actual response positions and then converts those rows to FP32, while
TRL retains padded completion columns and masks them later in the loss. If
MiniCPM does not expose `logits_to_keep`, its model output can still be full
context width in BF16 before TRL slices it; the live model signature and peak
memory remain a lead-owned gate. TRL's selective helper uses FP32
`logsumexp` only when the sliced logits are already FP32; BF16 logits use
row-wise BF16 `log_softmax`.

`campaign_rl_launch.py` performs the same CPU preflight, computes the absolute
hard cutoff and soft checkpointing cutoff, writes a supervision receipt, and
executes GNU `/usr/bin/timeout` with `TERM` plus `--kill-after`.  The entry
receives `SEPALITH_CAMPAIGN_SOFT_DEADLINE`,
`SEPALITH_CAMPAIGN_HARD_DEADLINE`, and the checkpoint reserve through the
environment.  It never treats a process exit as a promotion decision.

## CPU verification and limits

The focused entry tests use real parent-manifest and weight-inventory bytes,
the actual ext4/v9fs filesystem probe, mocked model/trainer seams, resume and
deadline rejection, and a train-begin callback invocation.  The existing RL
trainer tests continue to exercise the pinned TRL constructor and fixed-ID
generation adapter.  No model weights, CUDA process, optimizer step, cloud
resource, or data admission was used by this packet.

Live acceptance remains lead-owned: an admitted RL-02 sidecar and selected-ID
order, a merged-SFT parent manifest satisfying the contract, a quiet single
CUDA lease, real BF16 load/PEFT counts, finite nonzero gradients, and a full
interruption/resume comparison are still required.
