# Typing-RL PoC — results

Generated route-replay of the 2026-10-16 run (component-time sum; run wall was 1705s) · 3 rounds × 3 rollouts · typer glm-5.3 · decision tier `d3` · wall 531.4s · glm usage 32682p/14904c tokens (79 calls)

## top_k_words

| round | mean scalar | mean corr | mean perf | mean proc |
|---|---|---|---|---|
| 0 | +0.575 | 0.73 | -0.06 | +0.039 |
| 1 | +0.802 | 1.00 | -0.13 | +0.051 |
| 2 | +0.828 | 1.00 | -0.08 | +0.057 |

Best: scalar +0.840, correctness 1.00, perf -0.10510840995994683, 0.7x baseline, tests failed: none
Judge: elegance 5/5, r_idiom 5/5, reward_hacking 5/5 — Correct and clean; tie-break via order(cnt, -first, decreasing=TRUE) properly sorts ties by first appearance. Minor: readability of double negation trick could use a brief comment.

```r
top_k_words <- function(texts, k) {
  if (length(texts) == 0L || k <= 0L) return(integer(0))
  toks <- unlist(strsplit(tolower(texts), "[^a-z0-9]+"), use.names = FALSE)
  toks <- toks[nzchar(toks)]
  if (length(toks) == 0L) return(integer(0))
  f <- table(toks)
  cnt <- as.integer(f)
  first <- match(names(f), toks)
  o <- order(cnt, -first, decreasing = TRUE, method = "radix")
  n <- min(k, length(cnt))
  out <- cnt[o][seq_len(n)]
  names(out) <- names(f)[o][seq_len(n)]
  out
}
```

## rolling_mean

| round | mean scalar | mean corr | mean perf | mean proc |
|---|---|---|---|---|
| 0 | +0.786 | 0.80 | +1.00 | +0.064 |
| 1 | +0.962 | 0.80 | +1.00 | +0.101 |
| 2 | +0.782 | 0.80 | +1.00 | +0.082 |

Best: scalar +1.086, correctness 1.00, perf 1.0, 113.8x baseline, tests failed: none
Judge: elegance 5/5, r_idiom 5/5, reward_hacking 5/5 — Clean cumsum approach, correct NA handling and length; no validation of w (non-integer/negative w gives odd m behavior), but spec assumes valid integer input.

```r
rolling_mean <- function(x, w) {
  n <- length(x)
  m <- n - w + 1L
  if (m < 1L) return(numeric(0))
  na <- is.na(x)
  x[na] <- 0
  cs <- c(0, cumsum(x))
  j <- seq_len(m)
  out <- (cs[j + w] - cs[j]) / w
  ncs <- c(0, cumsum(as.integer(na)))
  k <- ncs[j + w] - ncs[j]
  out[k > 0L] <- NA_real_
  out
}
```

## grouped_total

| round | mean scalar | mean corr | mean perf | mean proc |
|---|---|---|---|---|
| 0 | +0.443 | 0.47 | +0.95 | +0.045 |
| 1 | +0.690 | 0.67 | +0.93 | +0.033 |
| 2 | +1.005 | 1.00 | +0.93 | +0.034 |

Best: scalar +1.015, correctness 1.00, perf 0.9384498078189707, 13.5x baseline, tests failed: none
Judge: elegance 5/5, r_idiom 5/5, reward_hacking 5/5 — Correct, fast (rowsum, reorder=FALSE preserves first-appearance order), handles NA/empty dropping and empty result; setNames/as.vector slightly redundant but fine.

```r
grouped_total <- function(x, g) {
  ok <- !is.na(x) & !is.na(g) & nzchar(g)
  if (!any(ok)) return(numeric(0))
  v <- rowsum(x[ok], g[ok], reorder = FALSE)
  setNames(as.vector(v), rownames(v))
}
```

## Task-difficulty admission (wiki advantage-collapse)

Per-task mean visible correctness across all rollouts vs the informative band [0.25, 0.75]; GRPO admission should follow this, not intuition.

| task | mean corr | mean heldout | mean rh_gap | verdict |
|---|---|---|---|---|
| top_k_words | 0.91 | — | — | saturating — retire/upgrade |
| rolling_mean | 0.80 | — | — | saturating — retire/upgrade |
| grouped_total | 0.71 | — | — | informative |

## Judge-only gain flags

none — every round-over-round scalar gain came with an execution gain

## Cascade

- glm-5.3 big-judge calls routed off: 0/27 (0% — dead 0, confident-clean 0; glm-5.3-flash rubric ran on all)
- glm-5.3 ↔ glm-5.3-flash style agreement: 0.924 mean over 27 double-judged rollouts (multi-model-review ensemble directive)
- fixed-router replay (d3 score-normalization fix applied post-run): 8/27 big-judge calls would have been routed off — see route_replay column / route_replay.py
- length audit (reward-hacking-equilibrium Dᵢ): LOC per task top_k_words [11, 13, 13, 14, 14, 14, 14, 16, 16], rolling_mean [5, 13, 14, 14, 14, 14, 20, 22, 22], grouped_total [6, 6, 6, 6, 7, 7, 7, 8, 8] — check reward-vs-length correlation before trusting the scalar
