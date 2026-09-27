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
- Upload script: `/mnt/e/sepalith/hf-staging/upload_secure_20260927.py`. It
  uses an allow-list and refuses sealed-final paths. Its log is
  `/mnt/e/sepalith/hf-staging/upload-20260927.log`, and it writes
  `upload-receipt.json` next to the script. Targets:
  - `scholzmx/sepalith-2b-cpt` (public model): `checkpoint-11586/` and
    `checkpoint-11649/`, holding weights, config, tokenizer, chat template,
    campaign manifest and trainer state, plus a model card. No optimizer
    state.
  - `scholzmx/sepalith` (public dataset), under `campaign-20260915/`: SFT TRAIN
    rows and sidecars, roxygen queue, schedules, DEV75, CPT cohort rows and
    draw schedule, CPT 2K validation panel, and the provenance tarball.

## Remaining steps

1. Read `upload-20260927.log` and `upload-receipt.json`. If the run did not
   finish, run it again: `uv run --no-project --with huggingface_hub python
   upload_secure_20260927.py all` from `/mnt/e/sepalith/hf-staging/`, with
   `HF_TOKEN` loaded. Uploads deduplicate, so a rerun is cheap.
2. Verify both `model.safetensors` sha256 values against CONTEXT.md. The
   script's `verify` stage does this.
3. List `campaign-20260915/` in the dataset and compare it with the upload
   list in the script. Check that no path contains `final` except
   `combined-draw-schedule-final-v1.json`, which is a CPT schedule and not
   evaluation data.
4. Update the dataset README. It still says "Private" at the top. Mark it
   public and add a short `campaign-20260915/` section listing the folders
   above, with a note that the sealed final set is intentionally absent.
5. Write `status/C00.json` with the repository URLs, file counts and
   verification results.

## Acceptance

- Both weight hashes match.
- Every allow-listed path exists on the Hub.
- No sealed-final file is present.
- The dataset README is accurate.

## Stop and ask if

- Any upload would include a path outside the allow-list.
- A hash mismatch persists after a rerun.
