# PRE-03 root storage relief audit

This packet audits one campaign-owned immutable parent checkpoint and a
path-preserving relocation plan. It performs no copy, move, deletion, link
creation, model load, or training action.

The candidate is:

```text
/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-CPT-global-a-250-merged/model.safetensors
```

The observed file is regular, owned by `m0hawk:m0hawk`, mode `0600`,
5,033,557,128 bytes, and SHA256
`5f12810692a90ebb32f589a25e5adcedac53d9fb6edd9166b8245fdec6e3dd1c`. Its
seven-file parent directory is also regular and currently has no symlink
components. The hash matches the accepted merge receipt, task recipe, native
selection recipe, and merged-parent manifest. The E destination
`/mnt/e/sepalith/campaign-20260915/models/SFT11-CPT-global-a-250-merged` is
absent on a different device with about 1.58 TB available.

At the final audit observation, `lsof` found no open descriptor for the exact
root weight file and no process command line mentioned it. During an earlier
observation, the host-memory guard mentioned the path as its release-cache
argument but did not hold an FD; that guard has since exited. The active
full-weight trainer was observed with its E checkpoint-66 weight open, not this
root parent. Root must repeat the exact FD check immediately before any move.

The symlink compatibility result is narrow and tested with synthetic tiny
safetensors files:

- A directory symlink at the parent path passes the current dense CPT layout
  validator because the final `config.json` and `model.safetensors` remain
  regular files.
- A symlink at the individual weight-file path is rejected by that validator.
- The native final-row path guard rejects a directory symlink component.

Therefore, the safe scoped option for a future CPT/full-weight training load is
to copy the complete seven-file parent directory to E, verify every explicit
file size and hash plus `validate_dense_weights`, then replace the original
directory with a directory symlink after all readers are stopped. Keep a root
backup until the original path has passed parent preflight and load checks. A
fresh recipe bound directly to the E directory is safer for native/final
evaluation and avoids symlink-sensitive guards. Do not use the directory link
for native source closures, final-row paths, or any validator that rejects a
symlink component.

The exact move/link/delete sequence remains root-owned because it is destructive
and must be coordinated with checkpoint I/O. The audit leaves the source
untouched and records the command stages in `candidate-audit.json`.

Run the read-only audit with:

```text
python3 audit_candidate.py --output candidate-audit.json
python3 -m unittest -v test_relocation_semantics.py
```
