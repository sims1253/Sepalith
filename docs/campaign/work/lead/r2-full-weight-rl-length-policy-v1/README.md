# RL-11 full-weight RL length policy preparation

The frozen production driver still encodes a 2,048-token prompt and a
192-token completion.  That profile excludes 4,106 of the 15,006 currently
accepted TRAIN rows.  In particular, 2,878 rows have complete gold
completions longer than 192 tokens.  Treating 192 as a production cap would
discard supervised signal and create length-capped generations.

`rl_length_policy.py` validates a recipe-bound length envelope against the
complete stored row.  Prompt length includes the manual BOS; completion length
includes terminal EOS.  It returns the original row unchanged when admitted
and raises a named error when any limit is exceeded.  It contains no slicing
operation.  Its streaming census hashes the bytes it reads, checks file
identity before and after the pass, validates one-BOS/one-EOS geometry, and
records a digest of every rejected row ID.

The 4,096-context candidate uses a 3,072-token prompt allowance and a
1,024-token completion allowance.  It admits all 15,006 current repaired rows:
the observed maxima are 3,000 prompt tokens, 933 completion tokens including
EOS, and 3,064 total tokens.  This establishes coverage for this exact pool.
It does not establish full-weight GRPO memory fit or a universal limit for
later TRAIN additions.  Production admission still requires a resource probe
with the exact number of simultaneous candidates and a new census whenever
the bound pool changes.

The active full-weight CPT recipe independently binds 16,384-token sequences
(`r2-cpt-full-corpus-root-launch-v1/bound-recipe.json`, SHA-256
`88f5d6b1348802a8f0efcccd858f01c62269c771512c154b30ca7029c5a88498`).
That corroborates 16K CPT execution support.  CPT uses one all-token sequence
per microbatch, so it is not evidence that multi-candidate RL generation and
backpropagation fit at 16K.

Reproduce the census:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 rl_length_policy.py \
  --rows /mnt/e/sepalith/campaign-20260915/data-work/DAT10-finish-source-repair-v3/train-token-rows.jsonl \
  --policies length-policies.json --output length-census.json
```

Run CPU tests with temporary files on E:

```sh
TMPDIR=/mnt/e/sepalith/campaign-20260915/tmp/rl-length-policy-v1 \
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

This packet prepares policy and integration evidence only.  It does not admit
RL training, load a model, use CUDA, or access DEV/final content.
