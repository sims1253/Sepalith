# RL-08 cap80 replay independent review

Status: **pass for the bounded replay-105 audit; the long run remains Root-owned**.

The audit compared the known d run at cap `.75` with the e run at cap `.80` using only the named run directories. It did not launch, stop, or modify the run, recipe, source, CLI state, or checkpoints.

The e load audit records allocator fraction `0.8`, total CUDA memory `34,190,458,880` bytes, and cap `27,352,367,104` bytes. E completed optimizer steps 101 through 105. The formerly failing d boundary completed in e as step 104, with finite gradient norm `0.2059539258480072`, reward `0.8146708607673645`, and peak allocated `21,208,753,664` bytes. Step 105 also completed and produced a sealed full checkpoint.

For each source update 100, 101, 102, and 103, d and e had 32 generation rows. Generation records matched exactly after removing the expected `elapsed_sec` field, including source schedule hash, prompt hash sequence, generated ID hash sequence, group geometry, and terminal accounting. The 32 reward records per update matched exactly, including source row ID order, output ID hashes, and reward vectors.

The optimizer metrics for successful steps 101, 102, and 103 matched exactly for gradient norm, loss, reward, reward standard deviation, completion length, and token count. Gradient records for d/e matched exactly for pre-update global steps 100, 101, and 102. D has no gradient record for global step 103 because it failed during the corresponding update-104 backward pass; e has one new finite record there, so no equality claim is made for that previously failing gradient.

The matching digest evidence is stored by update in the checker output. The source schedule hash for every compared row was `2f0d2d03a62412644c05e4a9ec2dd1d07d056b1e36284635ae7623c35a436132`.

E full checkpoint 105 is:

`/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RL-primary-p2-mb4-full5-e/archive/full/checkpoint-105`

Its manifest identity is the accepted cap80 identity `ebf964b456eff1c2df2b3064277afe346db9a358eb47b5a902a203822b5b4818`; the manifest is full step 105. Frozen production `campaign_checkpoint.verify_checkpoint(..., require_full=True)` passed for this checkpoint.

At the audit snapshot, e had no `output/failure.json`, `output/terminal.json`, or host `terminal.json`; its owned process was still running. Host supervision last reported 18,343 MiB available, with no driver event in the sampled line. The telemetry and log files can continue to grow after this audit, so their recorded hashes are snapshot hashes.

The CPU-only checker is [check_cap80_replay.py](check_cap80_replay.py). It imports only the frozen checkpoint verifier and performs no writes. It passed with `INDEPENDENT_CAP80_REPLAY_PASS`.

This result establishes a deterministic cap80 replay through update 105 and checkpoint integrity. It does not establish bitwise optimizer equivalence after the cap change, future capacity, quality, or promotion readiness.
