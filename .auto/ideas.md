# Ideas backlog (post-session; all major directions resolved)

Resolved this session — see `docs/validation/2026-10-17-quant-autoresearch.md`:
- attn_output q6_K/q8_0 boosts: measured (q6_K = efficient point).
- attn_k/attn_v q8_0: measured (cheapest per-MB boost).
- ffn_down q6_K all-layers: measured (fidelity, −30% decode tax).
- token_embd/output precision ladder q8→q6→q5→q4: measured (q5_K = floor).
- IQ4_XS direction: closed (dominated by the mixes).
- Search-panel + DEV confirmation of both finalists: done.

Not pursued (candidates for a future session):
- ~~Layer-band sensitivity for ffn_down q6~~ RESOLVED 2026-10-18: imatrix-top-11 band got 43% of the gain for 52% of the size; a 17-layer accidental variant (0.023730 @ 1679.0) shows the gain is ~linear in upgraded-layer count and NOT concentrated by imatrix importance. Interior points only; closed.
- attn_q q8_0 (+88.1 MB — it is 2048×2048): RESOLVED — dominated by mixB.
- Q6_K with q5_K/q6_K emb+head (~2.07 GB "safe small tier") — not needed to beat the baseline; Q6_K is already the quality fallback.
- imatrix variants (1M tokens, prompts-only, ctx 2048) — the current calibration is fine at these fidelities; revisit only if a future tier targets KLD < 0.01 at < 1.8 GB.
- ~~Selective IQ4_XS on ffn_gate/ffn_up~~ RESOLVED 2026-10-18: +0.0035 KLD for −33 MB on mixA-e4 (fails gate); the IQ4_XS base's poor KLD is largely the FFN halves at iq4_xs; size floor for gate-passing recipes = mixA-e4 1590.4 MB.
- A new llama.cpp build with faster Q5_K/Q6_K head matvec would directly move mixB's decode; user decision (upgrade is out of scope here).
- Complete-category override idiom for llama-quantize --tensor-type (q6 list first, catch-all q4_K second) — now proven; note it whenever partial attn_v/ffn_down mixes are needed, since ANY unmatched tensor re-enters the default mix at shifted counter positions (bit twice this session: night-run Q4_K_M+, and the first top-11 probe).
