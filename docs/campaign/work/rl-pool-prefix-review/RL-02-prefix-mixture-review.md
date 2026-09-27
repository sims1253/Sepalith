# RL-02 v5 prefix-mixture review

Status: **verified candidate for root review; no source, data, model, or launch state was changed.**

The recorded v4 audit reproduces the reported defect. The allocator in
`campaign_rl_pool.py` iterates `family_targets` and appends each family's
complete row queue before moving to the next family
(`_allocate_row_exposures`, lines 621–665). With the 24,000 source-draw
targets, the first 25/100/250 draws contain 200/800/2,000 `finish_block`
presentations. The first family transition is at draw 3,825.

The proposed correction is a family-level integer deficit scheduler. At
1-based position `i`, for every family with remaining quota, compute:

```text
priority(f) = i * target(f) - used(f) * 24000
```

Choose the greatest priority. Resolve ties with the root v5 policy's
`sha256(seed:interleave-v1:family)` in ascending order, then family name in
ascending order. The chosen family consumes the next row from its existing
per-family queue. Build those queues with the current `_rotation_order` and
the existing `_row_capacity`/replay-cap checks. This changes family order only;
it preserves the row multiset, each family's row order, and every row cap.

Root's v5 artifact is the concrete application of that policy:

| identity | value |
| --- | --- |
| schedule artifact SHA256 | `2e75e10bf86f6b72e33fd72cfed96194bdd7c7c4741db8793681985292581574` |
| ordered source-draw sequence SHA256 | `dc052dc99347eef5e348fe4621b73c3a1fb0137d1aca765a02edaf5b486cfea6` |
| selected-ID SHA256 | `24ec16f5ce5343539d31a44aaa843fb4be0de7c6831ec977432affc53fa77a9d` |
| ordered-ID SHA256 | `7e994a3e74a642c149908b0737dfd2befed055e336f7b94400ee9773a6a88d2d` |
| row-identity SHA256 | `d3187195f0ebe401b69df244bbbff0f2a28fb0500185428701f933e31e8eab60` |
| generator SHA256 | `f756e580999d3e9c4cb4913c2aa16474c314065a158d9929602648a334d41f75` |

The v5 final family multiset is exactly:

```json
{"finish_block": 3825, "format_propagation": 6000,
 "na_rm_propagation": 375, "no_op": 4800, "pipe_rewrite": 2400,
 "rename_propagation": 6000, "roxygen_drafting": 600}
```

Independent verification streamed only `id`, `family`, and `split` metadata
from the approved train rows. It found every draw in the selected train set,
equal v4/v5 row-draw multisets, and equal within-family row subsequences for
all seven families. The production `load_source_draw_schedule` accepted the
artifact with 24,000 draws, 8 source groups per update, and buffer reuse 4.

Prefix and block evidence from the independent check:

| source draws | `finish_block` | `format_propagation` | `na_rm_propagation` | `no_op` | `pipe_rewrite` | `rename_propagation` | `roxygen_drafting` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 4 | 6 | 0 | 5 | 3 | 6 | 1 |
| 100 | 16 | 25 | 2 | 20 | 10 | 25 | 2 |
| 250 | 40 | 63 | 4 | 50 | 25 | 62 | 6 |
| 24,000 | 3,825 | 6,000 | 375 | 4,800 | 2,400 | 6,000 | 600 |

The maximum running prefix discrepancy from the final quota proportions is
0.8125 draws, at prefix 300. The first 25-draw window contains six families;
`na_rm_propagation` has a quota of only 1.5625%, so zero in that short window
is within the integer quota error. All 100-draw and 250-draw blocks contain
all seven families. The actual 8-draw optimizer-update blocks have 4–6 family
types because the two small families cannot appear in every 8-draw block;
their per-block ranges are finish 1–2, format 2, na_rm 0–1, no-op 1–2, pipe
0–1, rename 2, and roxygen 0–1. No adjacent same-family run occurs in v5.

The framework-free sampler seam was exercised against the production
`campaign_rl_train.py` import. `load_source_draw_schedule` validated the
artifact and `CampaignRepeatSampler` expanded each 8-source-group update to
32 completion rows and four reuse passes: 384,000 dataloader rows for 24,000
source draws. The first 128-row update matched the expected
group→candidate→reuse order. `indices_from` matched the uninterrupted suffix
at optimizer updates 1, 25, and 100, with source cursors 8, 200, and 800.

The sampler loader already fails closed on schedule hash changes, incomplete
update groups, geometry mismatch, IDs outside the unique selected set, and
sequence/identity mismatch (`campaign_rl_train.py`, lines 416–505). The
sampler expands the bound stream before candidate repetition and buffer reuse
(`__iter__`, lines 827–865), while `indices_from` reconstructs only complete
prompt-group and generation-buffer boundaries (lines 917–925).

The remaining root action is to bind the v5 schedule path, artifact SHA, and
sequence SHA into the live recipe/stage identity and run the existing parent,
tokenizer, and admission gates. This review does not authorize launch or make
a training-quality claim.
