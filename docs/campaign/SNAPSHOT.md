# Campaign snapshot (September 27, 2026)

This directory is a verbatim snapshot of the code and small manifests from the
planning worktree's campaign directory:
`/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/`, on branch
`t3code/best-token-model-plan`. The full-weight CPT runtime, the editing-SFT
and evaluation-gate packets and the RL driver existed only there. They had
never been committed.

- **Included:** Python, R and shell sources, Markdown, packet manifests,
  recipes and templates, `tasks.json` and `state.json`, the campaign tool, and
  the supervisor scripts from `~/.local/state/sepalith/resume-20260921/`
  (under `state-snapshot/`).
- **Excluded:** row-level data (`*.jsonl`), large JSON (above 256 KB), logs,
  checkpoints, vendored libraries and caches. Also excluded is every path
  tied to the sealed final evaluation set.
- **Removed:** one test file with a token-shaped fixture was left out.
- **Where the rest is:** receipts, logs and full manifests are in the public
  dataset `scholzmx/sepalith` at
  `campaign-20260915/provenance/campaign-docs-snapshot-20260927.tar.gz`, with
  the same exclusions.

Files keep their original relative paths, so receipts that cite
`.../t3code-a8153bbb/docs/campaign/<path>` resolve to `docs/campaign/<path>`
here. Their contents are unchanged, and the hashes in packet source manifests
still apply.

Do not edit files here. New work goes into
`packages/sepalith/src/sepalith/training/`. See card C01 in
`docs/training/plan-20260927/`.
