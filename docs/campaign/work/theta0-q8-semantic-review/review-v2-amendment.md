# RUN-09 theta0 Q8 semantic DEV review v2 amendment

This amendment preserves review.md and the v1 receipt unchanged. It changes
only the semantic judgments identified during root review. The frozen Q8
quality artifact, corrected DEV panel, scores, and training labels remain
unchanged.

## Corrected roxygen judgments

| ID | Judgment in v2 | Correction |
| --- | --- | --- |
| dat07-existing-06847c759fd32bdf16f2e598 | uncertain, with data-support flag | The visible context is R/RcppExports.R and begins with the generated-file warning from Rcpp compileAttributes: do not edit by hand. In addition to the unresolved strings-versus-nested-list return shape, this is an explicit concern against admitting a generated-file documentation edit. |
| dat07-existing-81ddc4c6ddd226b04dcf0910 | useful_partial, with data-support flag | The matrix description is useful, but the output adds author Guangchuang Yu, which is absent from the visible function/context. That attribution is unsupported and prevents a fully useful/valid judgment. |
| dat07-existing-d38a792699b4cccded445dc0 | semantic_invalid | The visible code creates s^2 test treatments and returns Parameters with Number of Test Treatments (vt) = s^2; generated param s Number of test treatments is materially wrong. The description also falsely calls TCpRep1 a wrapper for itself. |

The corrected v2 roxygen totals are 3 useful_valid, 1 useful_partial, 1
uncertain, and 3 semantic_invalid. The v1 finish judgments are unchanged:
four protocol-valid finish outputs fail reconstructed R parsing and two are
mechanical cap/no-EOS failures.

## Separate data-support flags

- dat07-existing-06847c759fd32bdf16f2e598:
  generated_source_do_not_edit_context=true; visible evidence is the
  Rcpp-generated header in R/RcppExports.R.
- dat07-existing-81ddc4c6ddd226b04dcf0910:
  unsupported_author_attribution=true; no author appears in the visible
  function or context, so the generated attribution is not supported by this
  artifact.
- dat07-existing-d38a792699b4cccded445dc0:
  parameter_semantics_mismatch=true; visible output computes vt=s^2,
  while the generated parameter text calls s the number of test treatments.

These flags are diagnostic provenance findings. They do not repair output,
alter the panel, or create labels.
