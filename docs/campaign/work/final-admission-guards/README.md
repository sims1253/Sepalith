# DAT-08 raw-source admission guard

The raw_source_case_builder.py component is intentionally a derivation
component.  It accepts bytes and produces a raw fixture, but it does not
authorize a sealed split.  Its path helper also reads before a caller can
enforce a positive allowlist, and its JSON helper uses replacement semantics.
admission_guard.py is a caller-side boundary for the missing checks.

The guard takes an explicit list of case metadata, frozen weight and harness
receipts, semantic case bindings, training and DEV memberships, exact source
roots, and a write-once output path.  Its admit() method first validates all
of those values and records source lstat identity.  Only after the complete
preflight does it open source files.  The source is opened by walking the
selected root with O_NOFOLLOW directory descriptors, and the expected source
SHA-256 and inode identity are checked before the bytes are returned to the
raw builder.

The required case shape is:

~~~json
{
  "row_id": "final-row-001",
  "split_identity": "repository::package::group::final-row-001",
  "package_id": "package-001",
  "repository_id": "public-repository",
  "group_id": "history-group-001",
  "family": "rename_propagation",
  "source_path": "/authorized/root/package/file.R",
  "source_sha256": "<64 lowercase hex characters>",
  "context_sha256": "<64 lowercase hex characters>",
  "target_sha256": "<64 lowercase hex characters>"
}
~~~

split_identity is required to be stable across split construction.  The guard
rejects duplicate final row or split identities and compares row, package,
group, and split tokens with the supplied row_ids, package_ids, group_ids,
and split_identities memberships for TRAIN and DEV.  The two memberships must
also be nonempty and disjoint.

The freeze receipt must have status: "frozen", weights_frozen: true,
harness_frozen: true, and final_access_unlocked: true.  Its weight and
harness hashes must equal the caller-pinned expected_*_sha256 values, and
both timestamps must be no later than the explicit observed_at.  The content
freeze itself still requires observed_at to be at or after
2026-09-14T10:00:00Z; weights and the harness may have been frozen earlier.
The guard does not use the machine clock implicitly; a production launcher
should pass its recorded observation time.

The semantic receipt must use this proposed extension to the existing
component receipt:

~~~json
{
  "schema": "dat08.semantic-source-case-receipt.v1",
  "status": "source_cases_verified",
  "split": "final_locked_v1",
  "source_lock_sha256": "<caller-pinned source lock>",
  "content_access": "post_freeze_source_bytes",
  "verified_case_ids": ["final-row-001"],
  "constructor_sha256": "<raw semantic constructor hash>",
  "case_bindings": {
    "final-row-001": {
      "split_identity": "repository::package::group::final-row-001",
      "source_sha256": "<same source hash as case>",
      "context_sha256": "<same context hash as case>",
      "target_sha256": "<same target hash as case>"
    }
  }
}
~~~

The binding keys and verified IDs must exactly equal the requested cases.
Every source, context, and target hash is compared field by field before any
source bytes are opened.  This intentionally rejects the older receipt shape
that only lists IDs; the older component receipt remains evidence for the
TRAIN raw constructor and is not silently promoted to final authorization.

admit() returns an AdmissionResult containing the source bytes in memory for
the subsequent raw builder call and writes only a metadata manifest.  The
manifest records the receipt hashes, validated per-case hashes, source byte
lengths, allowlist roots, and the no-follow read policy.  It sets
final_content_opened to false; it never writes prompts, targets, or source
contents.  The file is created through a temporary O_EXCL file, fsync, an
exclusive hard-link, parent-directory fsync, and cleanup.  Existing outputs
and races that create an output are rejected, and filesystem errors are
returned to the caller.

The CLI consumes JSON metadata files and requires all of the same pins:

~~~text
python3 admission_guard.py \
  --cases cases.json \
  --freeze-receipt freeze.json \
  --semantic-receipt semantic.json \
  --train-identities train-membership.json \
  --dev-identities dev-membership.json \
  --allowed-root /authorized/root \
  --weights-sha256 ... \
  --harness-sha256 ... \
  --source-lock-sha256 ... \
  --observed-at 2026-09-14T10:02:00Z \
  --output admission-manifest.json
~~~

The tests use only temporary synthetic files and metadata.  They cover the
freeze-before-read order, positive root and symlink rejection, split overlap,
stale semantic bindings, source replacement after preflight, CLI wiring, and
write-once output preservation.  They do not open a final, DEV, TRAIN, or
package inventory path.
