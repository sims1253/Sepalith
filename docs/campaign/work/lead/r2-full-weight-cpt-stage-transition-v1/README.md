# Full-weight CPT stage transition v1

This packet prepares a full-state transition from an admitted representative CPT checkpoint into a distinct admitted full-corpus draw schedule. It authorizes no launch. The selected source metadata is checkpoint 66: manifest `6ae1d9ce…`, dense weights `aac456d2…`, 85 saved FP32 tensors (174,080 elements), 296 BF16 tensors, and the exact source recipe and source closure recorded in `selected-source-checkpoint66.preparation.json`.

The initial transition verifies the original checkpoint against its original representative recipe identity with the unchanged full-checkpoint verifier. It then lets Transformers load the source model, hybrid optimizer FP32 state, scheduler, and CPU/CUDA RNG from that checkpoint. The destination corpus is not represented as the source sampler. Its streaming dataset starts at draw 0 and `ignore_data_skip=true` delegates every later resume cursor to the explicit stage-local dataset view.

Global optimizer steps remain monotonic. Checkpoint 66 gives `global_optimizer_step_offset=66`; destination stage step `n` is global step `66+n`. New checkpoint sampler state records both the stage-local cursor `n*16` and global step. A later same-stage resume verifies the destination checkpoint identity and starts the dataset view at the recorded stage-local cursor, with no source-corpus skip and no replay of destination rows.

V1 admits only an unchanged `constant_with_warmup` optimizer schedule after the source warmup is complete. Optimizer configuration, hidden/side learning rates, scheduler type, and warmup steps must exactly match the source recipe. The loaded scheduler state retains its source step, so warmup does not run a second time. Cosine or changed-horizon reinterpretation is rejected.

Checkpoint cadence is stage-local. The callback forces a full-state save at each selected cadence, mandatory quality stop, and terminal stage step rather than relying on divisibility of the offset global step. The first mandatory gate and terminal must be named in the admitted evaluation and preservation lists. Any later destination checkpoint may resume only with a fresh root continuation admission binding the unchanged destination recipe, exact checkpoint manifest, exact global step, and stage-local cursor.

Root command order is in `root-commands.json`. The initial command requires `--stage-transition-admission`; later same-stage resumes instead require `--continuation-admission`. Root must bind the terminal admitted union/cache and scientific cadence before either command becomes executable.
