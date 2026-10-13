# DAT-04A structured adapter v2 geometry correction

This v2 packet supersedes the DAT-04A structured reference bundle only for
zero-width insertion geometry. The frozen PRM-04 context remains unchanged:
an empty insertion uses `region_old=[]` and
`Cursor(-1, None, None)`. The physical source builder image now retains one
empty anchor line at that range. This line is included in the source hash and
selection range, so a replacement is applied between the captured prefix and
suffix and a no-op leaves the full source bytes unchanged.

The correction covers the actual DAT-03 train references
`00003e7d884439e973648de0` (no-op, source line 30),
`0010114ca8c9a3e1dfb7b161` (comment drafting, source line 28), and
`0002ac3116b2f408c9cf792b` (roxygen drafting, source line 30). Their builder
windows retain `physical_region_old=[""]`; the canonical prompt context does
not expose that sentinel. The adapter verifies the complete post-edit source
image from the physical range and records its hash and byte count. The
independent v2 checker verifies that every canonical empty range maps to one
in-buffer empty physical line, including the no-op unchanged-source invariant.

The v2 bundle remains a bounded, non-training probe: one deterministic
`train_group` row per requested family, with no final or TU3 rows opened. The
lead’s scientific exclusions remain in force: the sampled v1 `doc_sync` row
and the `edit_pairs` architecture claim are not admission candidates. All
structural rows remain pending family source validation, exact rendered
envelope collision checks, and tokenizer gates.

## Reproduction

Run the focused model-free tests:

```text
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 -m unittest discover -s experiments/training \
  -p test_campaign_admission_structured.py -q
```

Build v2 into separate non-training staging paths:

```text
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python3 experiments/training/campaign_admission_structured.py --bundle \
  --audit /mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-candidates-1-lowest.jsonl \
  --source-audit /mnt/e/sepalith/campaign-20260915/data-work/DAT-03-row-audit.jsonl \
  --source-audit-sha256 9878a70d5d822f1192ccfc2956346738d668dc38175871556d1af1e16dbccadd \
  --output /mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-reference-bundle-v2.json \
  --source-staging /mnt/e/sepalith/campaign-20260915/data-work/DAT-04-structured-source-snapshots-v2
```

Run the independent v2 source and physical-range checks with
`python3 docs/campaign/work/data/DAT-04-structured-source-checks-v2.py`.
The result is metadata-only verification; it does not admit rows or establish
final-set feasibility.
