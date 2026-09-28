# E750 notebook offline timeout-recovery pilot

The prior two-thread cap-192 run failed closed on the single maximum-prompt row after 111.880 seconds to first token and 69 generated tokens by the 120-second deadline. This packet replays exactly that row only, never a subset score, at 4 physical cores, 6 physical cores, and 6 physical cores plus 2 SMT siblings. Each arm uses a fresh server, cap 192, context 4096, identical prompt IDs, temperature zero, no prompt cache, and a 300-second offline deadline.

The result distinguishes prefill and decode timing and identifies a practical thread setting for a later fresh 75-row × 3-cap quality run. It is not production latency evidence. The 21 completed rows from the failed run remain diagnostic and must not be combined with later rows under a changed runtime binding.

If the pilot demonstrates every row can finish well within 300 seconds, a later full run may use the selected CPU setting, all 75 rows in every arm, deadline 300 per case, and an independent eight-hour wall watchdog. The five-second production profiles remain unchanged.
