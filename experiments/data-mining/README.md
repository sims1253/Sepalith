# Data mining

This stage collects R code and real edit history. It writes everything to the
NAS store at `/mnt/h/sepalith`, with provenance on every record. Data never
enters git. The published corpus lives at
[huggingface.co/datasets/scholzmx/sepalith](https://huggingface.co/datasets/scholzmx/sepalith).

Products, in the order this stage builds them:

- CRAN package shards: one JSONL file per package, plus tarballs, normalized
  source trees, and provenance.
- A selection of active GitHub R repos, then shallow clones of them.
- Edit-pair datasets (`examples.jsonl`, `eval.jsonl`) built from commit diffs.
- PR-review edit pairs built from reviewer comments and the fixes they caused.
- Hidden R rows from general code datasets. See `hidden_r_harvest/README.md`.

## Before you start

- Run `uv sync` from the repo root. Run every Python script as
  `uv run python <path>`.
- Tools on PATH: `git`, `air`, `jarl`, `gh` (log in first with `gh auth login`).
- Environment variables: none required. `SEPALITH_ROOT` is optional, and only
  `ingest_cran.py` reads it. Every other script hardcodes `/mnt/h/sepalith`.
- If your storage lives elsewhere, make `/mnt/h/sepalith` a symlink to it.
  That is the only way to move the data root without editing scripts.
- `select_repos.py` reads two inputs that no script in this repo builds:
  - the r-universe cran-to-git mapping, JSON files under
    `/mnt/h/sepalith/meta/cran-to-git/`;
  - a mirror-log ranking at `/mnt/h/sepalith/ranked/2026-08-15.counts.txt`,
    one `count package` pair per line.
  You must place both yourself.

## Run it

Run the steps in order. Step 1 honors `SEPALITH_ROOT`. Steps 2 to 7
hardcode `/mnt/h/sepalith`.

1. Ingest CRAN packages.

   ```bash
   uv run python experiments/data-mining/ingest_cran.py 100
   ```

   Consumes: the CRAN `PACKAGES` index and the cranlogs top list (100 max per
   call). Produces under the data root: `tarballs/`, `normalized/`,
   `provenance/`, `datasets/packages/<pkg>.jsonl`, `datasets/manifest.jsonl`,
   and `logs/`. For more than 100 packages it reads the newest
   `ranked/*.counts.txt` instead of cranlogs.

2. Select repos.

   ```bash
   uv run python experiments/data-mining/select_repos.py
   ```

   Consumes: the cran-to-git mapping, the ranking file, and the GitHub API
   through `gh`. Produces: `/mnt/h/sepalith/meta/selected_repos.json` and the
   GraphQL cache `/mnt/h/sepalith/meta/pushed_cache.json`.

3. Clone the selection.

   ```bash
   bash experiments/data-mining/clone_repos.sh /mnt/h/sepalith/meta/selected_repos.json
   ```

   Consumes: `selected_repos.json`. Produces: shallow clones at
   `/mnt/h/sepalith/git/<owner>__<repo>` and logs under
   `/mnt/h/sepalith/logs/`.

4. Mine edit pairs.

   ```bash
   uv run python experiments/data-mining/mine_edit_pairs.py \
     --repos-dir /mnt/h/sepalith/git \
     --spool /mnt/h/sepalith/datasets/edit_pairs_v1/spool \
     --per-repo 30
   ```

   Consumes: the cloned repos. Produces: one JSONL file per repo in the spool
   plus the progress file `_progress.jsonl`. Each row is one next-edit example
   in the format `../eval/run_eval.py` scores.

5. Finalize the edit-pair dataset.

   ```bash
   uv run python experiments/data-mining/finalize_edit_pairs.py \
     --spool /mnt/h/sepalith/datasets/edit_pairs_v1/spool \
     --out /mnt/h/sepalith/datasets/edit_pairs_v1
   ```

   Consumes: the spool. Produces: `examples.jsonl`, `eval.jsonl` (5% of repos,
   repo-level split), `stats.json`, and `repos.json` in the output directory.

6. Mine PR-review pairs (optional).

   ```bash
   uv run python experiments/data-mining/mine_pr_review_pairs.py \
     --max-prs-per-repo 6
   ```

   Consumes: the cloned repos and the GitHub API through `gh`. Produces:
   `/mnt/h/sepalith/datasets/pr_instructed_v1/pilot.jsonl`, `stats.json`, and
   an API cache under `_cache/`.

7. Expand the selection (optional).

   ```bash
   uv run python experiments/data-mining/expand_repos.py
   ```

   Consumes: `pushed_cache.json`, `selected_repos.json`, and the clone
   directory. Produces: `/mnt/h/sepalith/meta/selected_repos_v2.json`. Pass
   that file to step 3, then repeat steps 3 to 5.

## How it works

| Script | What it does | Usage |
|---|---|---|
| `ingest_cran.py` | Ranks CRAN packages by downloads, fetches tarballs, runs `air format` + `jarl check --fix`, writes one shard per package with license provenance. | `uv run python ingest_cran.py 500` |
| `select_repos.py` | Picks active GitHub R repos: cran-to-git mapping crossed with download rank, GraphQL activity check, per-owner cap. | reads `/mnt/h/sepalith/ranked/` |
| `clone_repos.sh` | Shallow-clones the selection since 2026-05-01, six at a time. | `bash clone_repos.sh <selection.json>` |
| `mine_edit_pairs.py` | Builds next-edit examples from commit diffs: the parent state becomes prefix/region/suffix, a sibling hunk becomes the event. | `--repos-dir --spool --per-repo` |
| `mine_waves.sh` | Mines newly cloned repos in waves until the spool drains. Safe to restart. | run after cloning |
| `finalize_edit_pairs.py` | Merges the per-repo spool into train/eval files with a repo-level split. | `--spool --out` |
| `mine_pr_review_pairs.py` | Pairs reviewer comments with the fixes they caused. Flags Copilot-authored comments. | `--max-prs-per-repo --probe-cap` |
| `expand_repos.py` | Grows the selection from cached lookups and GitHub search. | |
| `hidden_r_harvest/` | Detectors that found about 74k R rows hiding in general code datasets. Two stages: regex prefilter, then fence-aware confirmation. | see its README |

## Notes

- `run_mining.sh` and `mine_waves.sh` do not run as committed. They point at
  `experiments/pipeline/`, the old name of this directory, and they hardcode
  `/home/m0hawk/Documents/Sepalith/.venv/bin/python`. Run
  `mine_edit_pairs.py` directly (step 4), or fix those paths yourself.
- `clone_repos.sh` also hardcodes `/mnt/h/sepalith/git` and the `.venv`
  Python path. It needs the repo cloned at exactly that path.
- Skip-list: `BH` (C++ headers only, no R).
- The cranlogs API caps at 100 per call. Rankings beyond 100 come from the
  mirror-log file described above.
- Resumability: `ingest_cran.py` skips packages that already have provenance
  and a shard; `mine_edit_pairs.py` skips repos listed in `_progress.jsonl`;
  `clone_repos.sh` skips clones that exist; `select_repos.py` caches every
  GraphQL lookup.
- Politeness: clones run six at a time under `nice -n 10`; GraphQL lookups
  are batched at 100 slugs per query; the PR miner paces one request per
  `--min-interval` seconds (default 2) and caches every API body on disk.
- `finalize_edit_pairs.py` does not drop flagged rows. It tags them, and
  downstream mixtures choose.
