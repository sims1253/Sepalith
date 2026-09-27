# Semantic9534 post-render preparation v3

This packet supersedes the v2 verifier and preserves v2 unchanged. The v2
terminal and row-ID checks remain in place. V3 corrects the input-manifest
binding to the actual `sepalith.dat10.semantic9535.provider_inputs.v1` schema:
the provider denominator is 9,534, the one geometry hold is explicit, and the
9,535 upstream supported denominator is stored at
`upstream.semantic_supported`. The actual manifest has no top-level
`supported_denominator`; v3 derives that value as 9,534 + 1 and accepts an
optional derived field only when it agrees.

The same binding is used by fallback preparation, context-policy finalization,
and review-only materialization. They preserve these denominators and reject a
changed upstream value, provider shard sum, manifest hash, status, or
target-free marker. No input file or denominator is rewritten.

The completed CPU retry is the fresh root
`/mnt/e/sepalith/campaign-20260915/data-work/Semantic9535-provider-preparation-v1/render-16k-retry-rich-v1`.
The failed historical root
`.../render-16k-v1` remains preserved. Root must run the verifier first. It
then may prepare the 32K fallback from verified 16K policy holds, finalize the
context policy, and run the existing root-bound materializer. Each downstream
step remains review-only, target-free, and fresh-output guarded.

Tests include synthetic terminal controls plus two bounded real checks: the
17 KiB input manifest is validated against its pinned SHA and actual nested
denominator, and shard 0027 validates its pinned provider hash, terminal
binding, 746 input/output rows, and exact ordered IDs. The real shard files are
read only for this first-shard check; no full render verification or cohort
scan is run by preparation. Root may run `commands/verify_render16.sh` after
review when full terminal verification is desired.

No model, optimizer, target, label, CUDA, renderer launch, or frozen input
mutation was performed. Terminal output hashes remain root-owned evidence from
the completed retry; this packet records them only through the verifier after
root chooses to run it.
