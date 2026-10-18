"""Card C13: offline post-SFT steps (RFT, then DPO) on the full-weight runtime.

The editing-SFT modules under ``training/sft`` are bound by hash into the SFT
arm recipes (``recipe_v2.SOURCE_GLOBS``), so this package lives beside them and
only imports from them:

- ``rules``: R1-style offline gates, R5 adoption, the DPO and no-edit guards;
- ``data``: the RFT cohort and the DPO pair file built from a C12 census;
- ``dpo``: completion log-probabilities, the sigmoid DPO loss, pair batches;
- ``recipe``: parent-free templates, parent binding and the fail-closed loader;
- ``trainer``, ``reference``, ``dev_generation``: GPU processes (night only);
- ``night``: the night-queue driver; ``jobs``: night-queue job files.
"""
