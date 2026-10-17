# Ideas backlog

- Region-copy drafter (brief direction 2): draft continuation tokens from the
  aligned CURRENT-region position in the prompt, re-align after mismatch
  (suffix-decoding style). Needs C++ patch in a llama.cpp fork — only if the
  offline oracle clearly beats the best n-gram mode AND n-gram leaves warm-e2e
  headroom after accounting for prefill.
- ngram-map-k / k4v: persistent n-gram map with count voting — may handle the
  most-recent-match failure of ngram-simple better. Sim first.
- DSpark released drafter: only if 1-2 leave headroom; expect low acceptance
  (weights moved). Converter at ~/.local/opt/llama.cpp-b10453/venv-convert.
- Trained drafter proposal (EAGLE-3/DSpark-adapt/MTP): report-only, no training.
  Night-queue job JSON format at docs/training/plan-20260927/c09/jobs/*.json.
- 4-thread (laptop) numbers for top-3 configs at the end.
- Q4 tier cross-check for the winning config (spec results can differ by quant).
