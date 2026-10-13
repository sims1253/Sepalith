# RL-08 notebook reward replay

This packet replays the frozen `protocol_exact_reference_v1` scorer over the
320 completions already produced by `RL-R2-step500-exact-tenstep-v1`. It does
not generate completions, load a model, touch CUDA, or execute generated R.

`materialize_inputs.py` verifies all large source artifacts at their admitted
hashes and writes only the 80 referenced TRAIN contexts, 320 output ID rows,
320 recorded reward rows, and pinned tokenizer. `replay_rewards.py` verifies
the reduced artifact hashes and frozen scorer source, decodes the stored token
IDs, recomputes every reward record, and requires exact full-record equality.

The notebook replay returned exact parity for all 320 rows in 0.342285 seconds.
The ten 32-row scorer batches took 0.003117 to 0.004028 seconds each (0.035601
seconds total). Input loading and source/tokenizer initialization account for
the rest. Staging 21,443,268 bytes took 0.58 seconds.

For future runs, the GPU generation process can seal each existing completion
batch with its source schedule, tokenizer, output-ID, and scorer hashes. A
CPU worker can consume that immutable batch, return its replay-row digest and
parity result, and gate optimizer acceptance on exact parity. This is an
offloaded verification seam; it does not implement distributed on-policy RL.
