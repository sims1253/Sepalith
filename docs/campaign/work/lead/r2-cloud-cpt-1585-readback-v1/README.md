# CPT checkpoint 1585 private readback

This packet downloads the exact private Hub tar pinned by the root metadata
receipt. It checks the tar hash and member table before extracting each member
through an exclusive file descriptor and independent SHA-256 stream.

The verifier does not import torch, load a model, or deserialize pickle state.
It reads only the JSON header of `adapter_model.safetensors`; optimizer,
scheduler, RNG, and training-argument files remain opaque and byte exact.

Run from the plan worktree with cached Hugging Face authentication:

```sh
ionice -c3 nice -n10 taskset -c 0,2 env PYTHONNOUSERSITE=1 \
  /home/m0hawk/Documents/Sepalith/.venv-sft/bin/python \
  docs/campaign/work/lead/r2-cloud-cpt-1585-readback-v1/readback.py \
  --metadata-receipt docs/campaign/receipts/PAR-R2-CPT-tar1585-root-metadata-review.json \
  --root317 /mnt/e/sepalith/campaign-20260915/cloud-readback/CPT-v4-checkpoint317-tar-v1/extracted \
  --destination /mnt/e/sepalith/campaign-20260915/cloud-readback/CPT-v4-tar1585-full-v1
```

The command refuses an existing destination. Root review and comparison with
checkpoint 1902 remain required before selection.
