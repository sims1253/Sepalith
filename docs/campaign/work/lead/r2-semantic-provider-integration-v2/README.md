# Semantic provider integration v2 (review only)

This is an isolated production-integration candidate. It preserves the frozen
v1 provider and fixes three review blockers:

1. A standalone or newly created `.R` file without package metadata receives
   `not_applicable` namespace evidence and continues through the ordinary
   full-document selector.
2. Package `NAMESPACE` bytes are re-read and SHA-256 checked after tokenizer
   boundaries and again immediately before a suggestion may be emitted.
3. Both bounded R parse-only helpers run with `Rscript --vanilla` and a minimal
   `PATH` environment.

The freshness callback closes over the exact namespace path and SHA parsed for
the request. The extension invokes it after selection and after the model
response, before applying or caching a suggestion. It also retains the existing
document URI, version, and content-hash checks.

This packet does not edit the active extension, launch a model, or admit data.
The running 4,551-row preparation remains bound to frozen provider v1. A v2
rerender/delta proof is required before any v1-rendered rows can represent v2
prediction-time behavior.

