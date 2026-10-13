# Full-weight Trainer resume proof

This preparation runs two deterministic updates over eight exact 2,048-token
TRAIN rows in three root-owned lanes: uninterrupted through step 2, stopped
cleanly after step 1, and resumed from the sealed full-weight checkpoint-1
through step 2. It uses Transformers 5.5 `Trainer`, a sequential sampler, the
reviewed full-weight checkpoint v2 helper, and the runtime-observed Aurora mix
dispatch for all 381 trainable parameter tensors. After the Unsloth-first load,
it applies the reviewed full-vocabulary tokenizer restoration and requires
BOS=0 and EOS=PAD=1 before training.

Each `on_save` callback seals the actual Trainer checkpoint after model,
optimizer, scheduler, trainer state, and RNG state are written. Resume performs
`verify_checkpoint(..., require_full=True,
expected_checkpoint_kind="full_weights")` before model/framework loading.
The final comparison requires exact model, optimizer, scheduler, CPU RNG, CUDA
RNG, optimizer dispatch, and draw-cursor fingerprints.

The proof uses the optimizer's smoke learning rates and only two updates. It
proves checkpoint and resume mechanics; it is not a long-run schedule or a
quality result. Root must run each live lane under the exclusive CUDA/resource
guard and preserve fresh output paths. The three lanes can create several full
5 GB model checkpoints, plus FP32 optimizer state, on the E: output volume.

`root-commands.json` gives the exact order and argv. The preflight is CPU-only.
