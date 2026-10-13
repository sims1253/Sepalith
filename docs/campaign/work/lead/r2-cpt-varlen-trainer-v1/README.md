# Full-weight CPT variable-length Trainer candidate v1

Status: **CPU mechanics prepared; GPU backend and model parity unproven; no
launch admitted**.

This packet copies the frozen `r2-cpt-prefix-extension-v1` source closure and
changes only four production files: a new `varlen_update_adapter.py`, the
Trainer wiring, backend evidence binding, and recursive identity support. The
hybrid optimizer, full checkpoint, scheduler, RNG, tokenizer repair, streaming
cache, heldout evaluation, milestone stop, and prefix contracts remain copied
byte-for-byte.

Each dataset item is one logical optimizer update containing exactly 16
consecutive frozen schedule draws. Rows are greedily packed in their original
order under the admitted complete-row context cap. Packing never crosses an
optimizer window, removes no token or label, resets positions at every member,
omits the dense attention mask, and supplies `packed_seq_lengths` for
block-diagonal attention.

Transformers uses internal batch size 1 and gradient accumulation 1 because
the dataset item already represents the reviewed micro1/acc16 scientific
update. `VarlenUpdateTrainerMixin.training_step` performs each physical
forward/backward sequentially with the same pre-packing supervised-token
denominator. Every graph is released after its backward. Transformers then
performs one hybrid optimizer and scheduler step. Checkpoint sampler cursors
remain 16 draws per global optimizer step.

The real Transformers 5.5 CPU regression starts from an ordinary checkpoint
whose Trainer used gradient accumulation 4, transitions to the logical-update
seam, stops after one destination update, and resumes. The resumed result is
bit-exact for model, optimizer, scheduler, and destination cursor versus the
uninterrupted destination run. A separate full-update test compares one pack
against several packs with the same denominator.

Before a GPU run, root must fill the backend kernel and actual-model parity
receipts in `root-admission.template.json`. The binder accepts only `xformers`
or `flash_varlen`; runtime rechecks the selected backend and refuses SDPA. For
the current xFormers candidate, `PYTHONPATH` must point to the reviewed cu130 E
overlay in `root-commands.json`.

This is a single-device candidate. It rejects `world_size != 1`. Eight-GPU
DDP/FSDP needs a distinct design that globally reduces the supervised-token
denominator and coordinates rank pack counts while preserving each 16-row
window; this packet must not be launched unchanged on p4d.

CPU verification:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/cpt-varlen-trainer-v1 \
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  source/experiments/training/full_weight_cpt_trainer.py \
  preflight-template --recipe recipe.template.json

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/cpt-varlen-trainer-v1 \
/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B \
  -m unittest discover -s tests -v
```
