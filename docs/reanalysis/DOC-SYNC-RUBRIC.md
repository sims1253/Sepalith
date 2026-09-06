# Retrospective doc_sync rubric v1

Frozen before inspecting model outputs for this reanalysis (2026-09-06).
This rubric supplements historical exact/validator scores; it does not replace them.
The adjudicator has read aggregate historical claims. Model labels and scores will
be hidden in deduplicated, shuffled case/output packets. This is partial blinding,
not an independent or prospectively blinded experiment.

Judge the entire replacement region after the historical response parser:

- Coverage: every argument introduced by the visible signature edit has exactly
  one nonempty @param description; no invented argument tags.
- Preservation: existing documentation remains intact in meaning, with no lost
  tags or unrelated code changes.
- Placement: parameter documentation belongs to the same roxygen block, before
  @return/@export for insertion cases. Exact line adjacency is not required.
- Formatting: valid roxygen comment lines, no explanatory prose, fences, edit
  markers or executable additions inside the replacement region. Trailing space
  and capitalization differences alone do not fail.
- Factual content: supported by the visible prompt (pass), contradicted by it
  (fail), or plausible but not established (unknown). A default can establish a
  default, but an argument name alone cannot establish implementation behavior.
  Report target-semantic compatibility separately: a faithful paraphrase of the
  intended target meaning can pass this even when that meaning is not entailed
  by the visible prompt. Merely naming an environment or a flag is incomplete.
- Canonical wording: retain original exact/validator scores unchanged. Wording
  differences alone must not fail target-semantic compatibility.

A target-semantic usable output passes coverage, preservation, placement,
formatting and target-semantic compatibility. A prompt-supported usable output
also requires factual content = pass. Unknown is never silently counted as pass.
Report each dimension, case-level counterexamples, and arm totals. Cases are
reused development tasks, not independent evidence for a general capability.
