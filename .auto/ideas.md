# Ideas backlog (post-session; all major directions resolved)

Resolved this session — see `docs/validation/2026-10-17-quant-autoresearch.md`:
- attn_output q6_K/q8_0 boosts: measured (q6_K = efficient point).
- attn_k/attn_v q8_0: measured (cheapest per-MB boost).
- ffn_down q6_K all-layers: measured (fidelity, −30% decode tax).
- token_embd/output precision ladder q8→q6→q5→q4: measured (q5_K = floor).
- IQ4_XS direction: closed (dominated by the mixes).
- Search-panel + DEV confirmation of both finalists: done.

Not pursued (candidates for a future session):
- Layer-band sensitivity for ffn_down q6 (spend on the ~10 worst layers only) — could keep most of mixB's KLD gain without its decode tax; needs a KLD probe ladder over layer subsets (~6 runs).
- attn_q q8_0 (+11 MB) — unprobed cheap boost; marginal expected gain.
- Q6_K with q5_K/q6_K emb+head (~2.07 GB "safe small tier") — not needed to beat the baseline; Q6_K is already the quality fallback.
- imatrix variants (1M tokens, prompts-only, ctx 2048) — the current calibration is fine at these fidelities; revisit only if a future tier targets KLD < 0.01 at < 1.8 GB.
- A new llama.cpp build with faster Q5_K/Q6_K head matvec would directly move mixB's decode; user decision (upgrade is out of scope here).
