# Native CPT checkpoint194 continuation preparation

This packet prepares the already selected checkpoint194 continuation through the next scheduled stage-local checkpoint at global step322. It does not admit or launch training.

The fresh runtime source fixes the frozen native-v2 overlay's source-schema mismatch. It also makes the lifecycle contract explicit: Transformers may transiently retain two native checkpoints, while the callback removes the predecessor only after the current checkpoint is atomically published and verified on the fresh E archive. The scientific recipe, optimizer, scheduler, RNG, sampler, rows, draw order, and global offset66 do not change.

The root must materialize fresh admitted copies of the four `prepared/*.json` records in dependency order. `root-commands.template.json` then creates a runtime recipe with fresh native/E output roots, verifies every staged input and all 17.25 GB of checkpoint194 under held file descriptors, performs the actual resume preflight, and finally provides the child command for a separately guarded root launch.

Capacity accounting reserves the existing 9.33 GB immutable staged data plus two 18 GiB transient checkpoint slots (47.98 GB total). It leaves 104.01 GB from the measured post-staging free space, above the required 70 GiB floor. The existing E archive and its checkpoint194 run result remain immutable evidence.
