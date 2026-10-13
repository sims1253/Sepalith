# SFT-11 CPT hot-retention audit

This is a read-only storage audit. It records the named native checkpoint
copies, their owners, the campaign capacity arithmetic, and a conditional
retention recommendation. It performs no removal, rename, cache drop, or
checkpoint mutation.

## Findings

The ordinary reference checkpoint 330 on C is an exact durable duplicate of
the E archive. Its 12 manifest payload files plus `campaign-manifest.json`
were hashed on both paths with an 8 MiB buffer, and every size and SHA-256
matched. The scan took 190.352 seconds. The ordinary C root has no symlinks.
The durable E path is:

`/mnt/e/sepalith/campaign-20260915/checkpoints/SFT11-native-varlen-canary-322-ordinary_reference-v1/full/checkpoint-330`

The corresponding inactive C hot copy is:

`/home/m0hawk/.local/state/sepalith/campaign-20260915/training/SFT11-native-varlen-canary-322-ordinary_reference-v1/runtime/checkpoint-330`

The E publication receipt is
`publish-receipts/checkpoint-330.json`, status
`durable_atomic_complete`, with manifest SHA-256
`f7bd7b819584a8abeadc37c677914bd8775f350bd097479d326425292ba8e943`.
The existing ordinary continuation instructions remain at
`docs/campaign/work/lead/r2-ordinary330-continuation-preparation-v1/README.md`
and its receipt. They identify this C checkpoint as immutable and preserve
checkpoint 322 as fallback.

At the owner probe, the live SFT-11 trainer (PID 2474029, supervised by
2473594/2473526) held all 13 files of the packed-330 C checkpoint. It held no
ordinary-330 C or E file. The trainer must remain untouched. PID 2473524 is
the root retry controller and PID 2473526 is its host guard. The owner probe
is a point-in-time observation; repeat it immediately before any future
eviction.

## Capacity interpretation

The measured regular-file sum of the named native bundle and C copies of
checkpoint 322, ordinary 330, and packed 330 is 61,083,174,943 bytes
(56.888139754 GiB). Adding the recipe's declared next full-checkpoint reserve
of 19,327,352,832 bytes gives 80,410,527,775 bytes (74.888139754 GiB),
5,248,600,095 bytes above the 70 GiB cap of 75,161,927,680 bytes. Using the
historical observed checkpoint size of 17,250,917,475 bytes still gives
78,334,092,418 bytes (72.954308631 GiB), 3,172,164,738 bytes over the cap.

After conditionally releasing the ordinary C hot copy (17,250,975,992
regular-file bytes), the measured named set plus the declared next reserve
would be 63,159,551,783 bytes (58.821916378 GiB), below the cap by
12,002,375,897 bytes. The current `/home` free-space floor is a separate
check and has substantial headroom; the receipt records the contemporaneous
`df` result.

The cap's implementation scope is narrower than a recursive campaign disk
quota. `native_capacity.py` receives `bundle_bytes`, `existing_hot_bytes`,
and `next_checkpoint_bytes`; the trainer computes `existing_hot_bytes` only
from `Path(args.output_dir).glob('checkpoint-*')`. Therefore sibling prior
stage directories such as 322 and ordinary/packed 330 are not automatically
counted by the runtime guard. This audit reports both that guard's documented
scope and the physical named campaign occupancy so a root admission does not
mistake one for the other.

## Conditional recommendation

Do not remove anything as part of this packet. Root may consider removing the
ordinary C hot copy only after all of these gates pass in the same admission:

1. an immediate owner/FD probe shows no trainer, guard, verifier, or archive
   process has any ordinary C path open or mapped;
2. the E publication receipt and all 12 payload entries remain present;
3. a fresh full ordinary C/E hash comparison still passes, including the
   manifest and exact file inventory;
4. root records that the E ordinary checkpoint is the retained restore source
   and that ordinary C is not the selected packed-330 or checkpoint-322 input;
5. root preserves the E publish receipt, ordinary continuation instructions,
   and this audit before the C path is removed, then re-runs the capacity and
   free-floor checks.

The selected packed-330 C/E pair and checkpoint-322 C copy are explicitly
outside this recommendation. The future 450 output path is also absent at
audit time; its save must pass a fresh capacity guard before publication.
