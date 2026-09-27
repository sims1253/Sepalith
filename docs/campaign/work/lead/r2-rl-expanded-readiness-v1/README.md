# RL-11 expanded signal readiness

This packet reviews and extends the repaired TRAIN-only signal driver. It closes one executable authorization gap: a model-bound pending config can no longer enter `run` until a separate root admission is pinned and replayed through `admit_signal_driver.py`.

The pilot remains inference and reward measurement only. It has 112 prompt groups, four candidates per group, and no optimizer updates. It does not authorize sustained RL, select a model, or limit a future admitted RL pool to these rows.

Root uses `root-commands.json` in order: create the exact pending model binding, issue a root admission for that binding, materialize a fresh admitted config, run CPU preflight, then use an exclusive CUDA lease for the pilot. Every resume rehashes the admitted config, source, model/tokenizer inventory, pilot spec, TRAIN rows, contexts, reward buffers, and each committed group.
