# DAT-10 r2 data expansion audit

This directory contains a CPU-only, candidate-only audit of the corrected
task rows and the short-task packet. Existing accepted packets are untouched.
The audit joins corrected rows to DAT-05 using the exact row ID, then looks up
the group in the DAT-02 global split. A row is eligible for the task candidate
only when its DAT-02 group is `train_group`; the CPT partition is used to
reserve its explicit `cpt_validation` groups. The CPT map was built for the
raw CPT source and covers 10,719 groups, so requiring every task row to appear
in that map incorrectly removed task-authored train groups.

The materialized candidate has 11,505 complete rows. It excludes all 432
explicit CPT-validation rows and has no sequence over 4,096 tokens. It retains
the 1,963 complete rows whose target tail is over the current 192-token task
gate in a separate cap-deferred candidate view. The current trainer-compatible
view has 9,542 rows. Compared with the existing 8,526-row primary packet, the
current view recovers 1,016 rows; the full bounded view recovers 2,979 rows.
The full view adds 2,792 finish rows and 184 roxygen rows. These are source and
token candidates only. Root admission must still repeat source, license,
tokenization, duplicate, and quality checks.

The existing task builder records the failure at
`r2-task-mixture-v1/prepare_task_mixture.py:167-182`: it returns
`group_not_in_cpt_partition` before it evaluates the DAT-02 split. Its target
gate is at lines 289-300. The repaired pipeline keeps the global held-out
guards, reserves `cpt_validation`, and permits a `task_train_unmapped_cpt`
status. It never adds those task groups to the CPT partition.

The reviewed task trainer still enforces `development_max_new_tokens == 192`
and checks the complete target tail. The 9,542-row view can use that identity
after root admission. The largest retained target tail is 933 tokens including
EOS, so the 11,505-row view needs a new task-stage identity whose declared
target cap is at least 933 (1024 is one possible engineering bound), a new
data/trainer identity, fresh schedule and row hashes, and a new DEV gate for
the long-target path. The target limit for training is a separate field from
`development_max_new_tokens`: retain the paired 192-token DEV baseline, and
use 512/1024 generation only as separately measured diagnostics when needed.
The report compares target gates at 192/256/384/512/1024 and full sequence
gates at 2K/4K/8K/16K; all 11,505 task candidates fit 4K, so this CPU audit does
not choose a context or output cap as a quality optimum. No target truncation
or silent label change is permitted. The DEV cap sweep already shows that
raising the cap alone is not a quality fix.

`rl-coverage.json` measures source-prompt consumption from the current RL
schedule. It has 8,246 eligible source IDs, 24,000 scheduled source draws, and
8,171 distinct source IDs (99.09%); all 75 omitted IDs are roxygen. Four
declared completions per source draw give 96,000 generated-candidate slots.
`buffer_reuse=8` repeats policy exposure, so the declared exposure count is
768,000 policy rows. No actual rollout output is present in this candidate
directory, so actual rollout coverage remains unevidenced.

Run the synthetic checks with:

```text
python3 -m unittest discover -s docs/campaign/work/lead/r2-data-expansion-audit-v1 -p 'test_*.py'
```

The expansion script is `audit_expansion.py`. It refuses to overwrite its own
artifacts and writes row/provenance hashes, an exclusion ledger, metadata-only
schedule inputs, and the report consumed by the root agent.
