# Full-weight varlen feasibility v1

Status: **prepared, not admitted**. No CUDA or production process was touched.

The installed Unsloth patch has a real isolation route: explicit
`packed_seq_lengths` becomes cu-sequence metadata and reaches either
FlashAttention varlen or xFormers `BlockDiagonalCausalMask`. Reset
`position_ids` separately resets RoPE. FlashAttention is not installed; the
actual RTX backend can only be established while holding the root CUDA lease.

`varlen_contract.py` forms each logical optimizer update from exactly 16
consecutive schedule rows. It packs only consecutive members into physical
groups capped at 16,384 tokens, copies every token and label, preserves the
masked first label of every document, and records the independent supervised
token denominator. This denominator must be passed explicitly: the installed
automatic packed counter subtracts internal boundaries although these rows
already mask them.

Run the CPU source and cursor checks:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/varlen-feasibility \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B verify_environment.py
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/varlen-feasibility \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python -B -m unittest discover -s tests -v
```

The root-only CUDA command is in `actual-model-command.json`. It compares the
same fixed 16 rows standalone and packed: hidden states, sampled logits, a
cross-document mutation control, globally weighted loss, and sampled
q/k/v/o/MLP gradients. It performs no optimizer step and writes no model.

A passing probe is still not a throughput claim. Production integration needs
an actual Trainer seam that consumes one logical 16-row update, performs its
variable number of physical backward calls with one shared denominator, then
performs exactly one optimizer/scheduler step. It also needs a timed comparison
and a full-state stop/resume proof through that seam. The current production
trainer remains unchanged.

`findings.json` records the checkpoint I/O audit. Current save publication is
already a sealed same-filesystem rename. Startup still rehashes and rereads the
5 GB model several times. Removing any pass requires a receipt plus stable-file
metadata and an actual restoration test; no unverified stream-hash replacement
is proposed here.
