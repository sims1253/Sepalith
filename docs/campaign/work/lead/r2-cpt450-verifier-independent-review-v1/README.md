# SFT-11 checkpoint-450 verifier independent review

This packet independently reviews the frozen root-owned
`r2-cpt450-review-root-v1/prepare.py` against the bound ordinary-stage recipe.
It reads only small JSON/source metadata and does not load checkpoint payloads,
run CUDA, or create a future checkpoint/evaluation artifact.

The current retry remains preterminal: guard PID 2473526 and trainer PID
2474029 are live, and the host/controller terminal files are absent. The
root verifier therefore fails closed before reading the future checkpoint.

The review checks the 120-update arithmetic (steps 331..450 and draws
4224..6143), the exact 12-file full-checkpoint contract, the source packed-330
lineage, ordinary execution identity, and the existing matched 2K/8K/16K
evaluation route. Tiny-fixture tests exercise missing/extra/mutated/symlinked
checkpoint files and non-contiguous draw controls.

This review does not grant checkpoint admission or evaluation launch. The
root verifier must be run only after the retry has written a terminal success
receipt and both training handles are gone.
