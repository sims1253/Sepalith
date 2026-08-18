# Data mining

Collects R code and real edit history. Everything writes to the NAS store at
`/mnt/h/sepalith` with per-record provenance. Never to git.

| Script | What it does | Usage |
|---|---|---|
| `ingest_cran.py` | Ranks CRAN packages by mirror-log downloads, fetches tarballs, runs `air format` + `jarl check --fix`, writes one shard per package with license provenance. | `uv run python ingest_cran.py 500` |
| `select_repos.py` | Picks active GitHub R repos: cran-to-git mapping crossed with download rank, GraphQL activity check, per-owner cap. | reads `/mnt/h/sepalith/ranked/` |
| `clone_repos.sh` | Shallow-clones the selection since 2026-05-01, six at a time. | `bash clone_repos.sh <selection.json>` |
| `mine_edit_pairs.py` | Builds next-edit examples from commit diffs: parent state becomes prefix/region/suffix, a sibling hunk becomes the event. Same format `eval/run_eval.py` scores. | `--repos-dir --out --per-repo` |
| `mine_waves.sh` | Mines newly cloned repos in waves until the spool drains. Safe to restart. | run after cloning |
| `finalize_edit_pairs.py` | Merges the per-repo spool into train/eval files with a repo-level split. | `--spool --out` |
| `mine_pr_review_pairs.py` | Pairs reviewer comments with the fixes they caused. Uses the authenticated `gh` CLI. Flags Copilot-authored instructions. | `--max-prs-per-repo --probe-cap` |
| `expand_repos.py` | Grows the selection from r-universe and GitHub search. | |
| `hidden_r_harvest/` | Detectors that found ~74k R rows hiding in general code datasets. Two-stage: regex prefilter, then fence-aware confirmation. Bluespec and Verilog also use `<-`, so the second stage matters. | `filter_codex.py`, `filter_ling.py` |

Skip-list: `BH` (C++ headers only, no R). The cranlogs API caps at 100 per
call, so rankings beyond 100 come from one full mirror log.
