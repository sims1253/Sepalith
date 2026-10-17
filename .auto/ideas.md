# Ideas backlog

- attn_output q8_0 only on flagged layers (imatrix flagged 39/41; HINT4) — saves most of the +88 MB; test q8_0 on {37..41} vs all layers vs every-2nd layer.
- token_embd q6_k / q5_k / q4_k: biggest untested size lever (−65/−92/−135 MB).
- IQ4_XS on ffn_gate/ffn_up only (keep rest Q4_K_M): −33 MB, KLD cost unknown.
- Q6_K with head q6_k + emb q6_k (−130 MB → ~2.07 GB): the "safe small tier", decode 21 tok/s.
- attn_v q8_0 all layers (Q4_K_M+ had attn_v q8_0 → its valid_drop was best among q4s; maybe not causal).
- imatrix variants: 1M-token calibration (2× chunks), prompts-only vs prompts+gold; ctx 2048.
- Layer-band sensitivity for attn_output and ffn_down (0–13 / 14–27 / 28–41).
- Measure changed-case / cap-rise proxy on TRAIN search panel — direct valid_drop predictor without DEV.
- Night-queue job JSON for CUDA confirmation of finalists (comparability with night receipt).
