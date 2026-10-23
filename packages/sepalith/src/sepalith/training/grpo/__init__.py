"""Cards C14 (step 1, prompt pool) and C15 (GRPO pilot) on the full-weight runtime.

- ``pool``: the GRPO prompt pool from a C12 census (TRAIN only) and its bundle;
- ``algorithm``: group advantages without std scaling, dynamic sampling with
  refill, the seeded pool sampler and update assignment (no torch);
- ``loss``: the token-normalised DAPO loss with asymmetric clipping and the
  masking of truncated completions (torch);
- ``rules``: R6 for GRPO evaluations (the R1 later-gate table plus the GRPO rows);
- ``recipe``: the bound pilot recipe, its pre-registered values and the loader;
- ``trainer``, ``dev_generation``: GPU processes (night only, explicit ``cuda:0``);
- ``night``: the night-queue driver (pilot and smoke); ``jobs``: job files.
"""
