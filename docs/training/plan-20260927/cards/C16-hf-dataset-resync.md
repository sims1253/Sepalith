# C16. Re-sync the public HF dataset

- **Where:** PC, daytime, network and CPU
- **Needs:** C00
- **Produces:** an up-to-date projection of the NAS corpus, families and
  mixtures in `scholzmx/sepalith`
- **Effort:** one agent-day

## Goal

The public dataset's older top-level folders (`corpus/`, `families/`,
`mixtures/`, `provenance/`, `datasets/`, `pretraining/`) were last updated on
September 1. The user wants the data secured and current.

## Steps

1. Read `experiments/post-processing/push_cases.py`. It builds the unified
   projection and makes batched directory commits, because per-file uploads
   exhaust HF's limit of 128 commits per hour. Run it in dry-run or listing
   mode against the NAS (`/mnt/h/sepalith/datasets`), and compare with the
   Hub's current tree.
2. Report what changed since September 1: new families, new mixtures, and
   corpus ledger updates.
3. Check eval protection before uploading. Nothing from the sealed final set,
   the eval-protected mirrors, or any `final`-named DEV or evaluation file may
   be included. Use the script's existing exclusions, and add a guard like
   C00's forbidden-path list.
4. Run the sync with batched commits. Verify with a tree listing and spot
   hashes.
5. Update the dataset README's layout table if folders changed.

## Stop and ask if

- The projection would upload more than 50 GB, or anything outside the
  documented folders.
- `push_cases.py` no longer matches the NAS layout.
