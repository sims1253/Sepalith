#' Composition statistics of one sequence
#'
#' @param x A single DNA sequence.
#' @param stats The statistics to compute, any of `"gc"`, `"at"`, `"gc_skew"`,
#'   `"at_skew"`, `"purine"`, `"keto"`, `"amino"` and `"entropy"`.
#' @return A named list with one element per requested statistic.
#' @export
composition_stats <- function(x, stats = c("gc", "at", "gc_skew", "at_skew", "purine", "keto",
                                           "amino", "entropy")) {
  stats <- match.arg(stats, several.ok = TRUE)
  chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  n_a <- sum(chars == "A")
  n_c <- sum(chars == "C")
  n_g <- sum(chars == "G")
  n_t <- sum(chars == "T")
  total <- n_a + n_c + n_g + n_t
  out <- list()

  if ("gc" %in% stats) {
    out$gc <- if (total > 0) (n_g + n_c) / total else NA_real_
  }
  if ("at" %in% stats) {
    out$at <- if (total > 0) (n_a + n_t) / total else NA_real_
  }
  if ("gc_skew" %in% stats) {
    out$gc_skew <- if (n_g + n_c > 0) (n_g - n_c) / (n_g + n_c) else NA_real_
  }
  if ("at_skew" %in% stats) {
    out$at_skew <- if (n_a + n_t > 0) (n_a - n_t) / (n_a + n_t) else NA_real_
  }
  if ("purine" %in% stats) {
    out$purine <- if (total > 0) (n_a + n_g) / total else NA_real_
  }
  if ("keto" %in% stats) {
    out$keto <- if (total > 0) (n_g + n_t) / total else NA_real_
  }
  if ("amino" %in% stats) {
    out$amino <- if (total > 0) (n_a + n_c) / total else NA_real_
  }
  if ("entropy" %in% stats) {
    p <- c(n_a, n_c, n_g, n_t) / total
    p <- p[p > 0]
    out$entropy <- if (total > 0) -sum(p * log2(p)) else NA_real_
  }

  out
}

#' GC content with a choice of ambiguity handling
#'
#' @param x A character vector of DNA sequences.
#' @param method `"strict"` counts only G and C over A, C, G and T;
#'   `"ambiguous"` also counts S as GC and W as AT; `"weighted"` gives every
#'   IUPAC code its expected GC fraction.
#' @return A numeric vector of GC fractions, one per sequence.
#' @export
gc_content <- function(x, method = c("strict", "ambiguous", "weighted")) {
  method <- match.arg(method)
  gc_weight <- c(
    A = 0, C = 1, G = 1, T = 0, S = 1, W = 0, R = 1 / 2, Y = 1 / 2,
    K = 1 / 2, M = 1 / 2, B = 2 / 3, D = 1 / 3, H = 1 / 3, V = 2 / 3, N = 1 / 2
  )
  out <- numeric(length(x))

  for (i in seq_along(x)) {
    chars <- toupper(strsplit(x[i], "", fixed = TRUE)[[1]])
    switch(method,
      strict = {
        gc <- sum(chars %in% c("G", "C"))
        at <- sum(chars %in% c("A", "T"))
        out[i] <- if (gc + at > 0) gc / (gc + at) else NA_real_
      },
      ambiguous = {
        gc <- sum(chars %in% c("G", "C", "S"))
        at <- sum(chars %in% c("A", "T", "W"))
        out[i] <- if (gc + at > 0) gc / (gc + at) else NA_real_
      },
      weighted = {
        known <- chars[chars %in% names(gc_weight)]
        out[i] <- if (length(known) > 0) mean(gc_weight[known]) else NA_real_
      }
    )
  }
  names(out) <- names(x)

  out
}

#' GC content at each codon position
#'
#' @param x A single coding DNA sequence.
#' @param frame The reading frame, 1, 2 or 3.
#' @return A named numeric vector with elements `gc1`, `gc2`, `gc3` and `gc`.
#' @export
codon_position_gc <- function(x, frame = 1L) {
  if (!frame %in% 1:3) {
    stop(sprintf("`%s` must be 1, 2 or 3.", "frame"), call. = FALSE)
  }
  chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  chars <- chars[frame:length(chars)]
  n_codons <- length(chars) %/% 3L
  if (n_codons == 0L) {
    return(c(gc1 = NA_real_, gc2 = NA_real_, gc3 = NA_real_, gc = NA_real_))
  }
  chars <- chars[seq_len(3L * n_codons)]
  pos <- rep(1:3, times = n_codons)
  is_gc <- chars %in% c("G", "C")
  is_base <- chars %in% c("A", "C", "G", "T")

  # gc1
  sel1 <- pos == 1L & is_base
  gc1 <- if (any(sel1)) sum(is_gc[sel1]) / sum(sel1) else NA_real_
  # gc2
  sel2 <- pos == 2L & is_base
  gc2 <- if (any(sel2)) sum(is_gc[sel2]) / sum(sel2) else NA_real_
  # gc3
  sel3 <- pos == 3L & is_base
  gc3 <- if (any(sel3)) sum(is_gc[sel3]) / sum(sel3) else NA_real_
  # gc
  gc <- if (any(is_base)) sum(is_gc[is_base]) / sum(is_base) else NA_real_

  c(gc1 = gc1, gc2 = gc2, gc3 = gc3, gc = gc)
}

#' Composition in sliding windows
#'
#' @param x A single DNA sequence.
#' @param width The window width.
#' @param step The step between window starts.
#' @param stat The statistic per window: `"gc"`, `"at"`, `"purine"` or `"entropy"`.
#' @return A data frame with columns `start`, `end` and `value`.
#' @export
sliding_composition <- function(x, width = 50L, step = 10L,
                                stat = c("gc", "at", "purine", "entropy")) {
  stat <- match.arg(stat)
  chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  n <- length(chars)
  if (width > n) {
    stop(sprintf("`%s` must not be longer than the sequence.", "width"), call. = FALSE)
  }
  starts <- seq(1L, n - width + 1L, by = step)
  value <- numeric(length(starts))

  for (i in seq_along(starts)) {
    win <- chars[starts[i]:(starts[i] + width - 1L)]
    win <- win[win %in% c("A", "C", "G", "T")]
    if (length(win) == 0L) {
      value[i] <- NA_real_
      next
    }
    value[i] <- switch(stat,
      gc = {
        mean(win %in% c("G", "C"))
      },
      at = {
        mean(win %in% c("A", "T"))
      },
      purine = {
        mean(win %in% c("A", "G"))
      },
      entropy = {
        p <- table(win) / length(win)
        -sum(p * log2(p))
      }
    )
  }

  data.frame(start = starts, end = starts + width - 1L, value = value)
}

#' Compare the base composition of two sequences
#'
#' @param x,y Single DNA sequences.
#' @return A list with elements `x_freq`, `y_freq`, `difference` (y minus x)
#'   and `distance` (half the sum of absolute differences).
#' @export
compare_composition <- function(x, y) {
  bases <- c("A", "C", "G", "T")
  # Frequencies for `x`
  x_chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  x_chars <- x_chars[x_chars %in% bases]
  if (length(x_chars) == 0L) {
    stop(sprintf("`%s` contains no unambiguous bases.", "x"), call. = FALSE)
  }
  x_freq <- as.numeric(table(factor(x_chars, levels = bases))) / length(x_chars)
  names(x_freq) <- bases
  # Frequencies for `y`
  y_chars <- toupper(strsplit(y, "", fixed = TRUE)[[1]])
  y_chars <- y_chars[y_chars %in% bases]
  if (length(y_chars) == 0L) {
    stop(sprintf("`%s` contains no unambiguous bases.", "y"), call. = FALSE)
  }
  y_freq <- as.numeric(table(factor(y_chars, levels = bases))) / length(y_chars)
  names(y_freq) <- bases

  difference <- y_freq - x_freq
  list(
    x_freq = x_freq,
    y_freq = y_freq,
    difference = difference,
    distance = sum(abs(difference)) / 2

  )
}

#' Composition summary for many sequences
#'
#' @param seqs A named character vector of DNA sequences.
#' @return A data frame with columns `id`, `length`, `gc`, `at`, `n_frac` and
#'   `entropy`, one row per sequence.
#' @export
composition_table <- function(seqs) {
  n <- length(seqs)
  ids <- if (is.null(names(seqs))) paste0("seq", seq_len(n)) else names(seqs)
  out <- data.frame(
    id = ids,
    length = integer(n),
    gc = numeric(n),
    at = numeric(n),
    n_frac = numeric(n),
    entropy = numeric(n)
  )

  for (i in seq_len(n)) {
    chars <- toupper(strsplit(seqs[[i]], "", fixed = TRUE)[[1]])
    known <- chars[chars %in% c("A", "C", "G", "T")]
    # length
    out$length[i] <- length(chars)
    # gc
    out$gc[i] <- if (length(known) > 0) mean(known %in% c("G", "C")) else NA_real_
    # at
    out$at[i] <- if (length(known) > 0) mean(known %in% c("A", "T")) else NA_real_
    # n_frac
    out$n_frac[i] <- if (length(chars) > 0) mean(chars == "N") else NA_real_
    # entropy
    p <- table(known) / length(known)
    out$entropy[i] <- if (length(known) > 0) -sum(p * log2(p)) else NA_real_
  }

  out
}
