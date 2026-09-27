# C00. Secure code, data and weights

- **Where:** PC, daytime, network only
- **Needs:** nothing
- **Produces:** code in git; weights and data on public Hugging Face
- **State at writing:** mostly done in the planning session on September 27

## Goal

Everything later cards need must exist outside this one machine: code in
GitHub, and data and weights on public Hugging Face. The sealed final set must
not be uploaded.

## Done in the planning session

- A verbatim snapshot of the planning worktree's code and small manifests is in
  `docs/campaign/`: 5,455 files after removing DEV-case data and vendored code, including the state-dir supervisors under
  `docs/campaign/state-snapshot/resume-20260921/`. The snapshot excludes
  row-level data, vendored libraries, checkpoints and every path tied to the
  final set. One test file with a token-shaped fixture was left out.
- Secret scan: none of the ten secrets in `~/.zshrc`, the old Hermes key or the
  Kaggle key occurs in the snapshot. Pattern scans for HF, GitHub, AWS, Kaggle
  and private keys found no live token.
- Upload script: `docs/training/plan-20260927/receipts/upload_secure_20260927.py`.
  It uses an allow-list and refuses sealed-final paths. It ran from
  `/mnt/e/sepalith/hf-staging/`, which was cleaned up afterwards. Its receipt
  is `receipts/C00-upload-receipt.json`. Targets:
  - `scholzmx/sepalith-2b-cpt` (public model): `checkpoint-11586/` and
    `checkpoint-11649/`, holding weights, config, tokenizer, chat template,
    campaign manifest and trainer state, plus a model card. No optimizer
    state.
  - `scholzmx/sepalith` (public dataset), under `campaign-20260915/`: SFT TRAIN
    rows and sidecars, roxygen queue, schedules, DEV75, CPT cohort rows and
    draw schedule, CPT 2K validation panel, and the provenance tarball.

## Remaining steps

All steps were completed on September 27; see `status/C00.json`. Both weight
hashes were verified. The dataset holds 27 files (5.93 GB) under
`campaign-20260915/`, with no forbidden paths. The dataset README now says the
repository is public and documents the folder. To repeat or extend an upload,
reuse the script with `HF_TOKEN` loaded. Uploads deduplicate, so reruns are
cheap.

## Acceptance

- Both weight hashes match.
- Every allow-listed path exists on the Hub.
- No sealed-final file is present.
- The dataset README is accurate.

## Stop and ask if

- Any upload would include a path outside the allow-list.
- A hash mismatch persists after a rerun.
