#' Count bases per sequence
#'
#' @param x A character vector of nucleotide sequences.
#' @param alphabet Either `"dna"` or `"rna"`.
#' @param ambiguous How to treat IUPAC codes: `"ignore"` skips them, `"other"`
#'   counts them in an extra column and `"split"` shares each code equally
#'   between the bases it stands for.
#' @return A numeric matrix with one row per sequence and one column per base.
#' @export
count_bases <- function(x, alphabet = c("dna", "rna"), ambiguous = c("ignore", "other", "split")) {
  alphabet <- match.arg(alphabet)
  ambiguous <- match.arg(ambiguous)
  bases <- if (alphabet == "dna") c("A", "C", "G", "T") else c("A", "C", "G", "U")
  last <- bases[4]
  iupac <- list(
    R = c("A", "G"), Y = c("C", last), S = c("C", "G"), W = c("A", last),
    K = c("G", last), M = c("A", "C"), B = c("C", "G", last), D = c("A", "G", last),
    H = c("A", "C", last), V = c("A", "C", "G"), N = bases
  )
  cols <- if (ambiguous == "other") c(bases, "other") else bases
  out <- matrix(0, nrow = length(x), ncol = length(cols), dimnames = list(names(x), cols))

  for (i in seq_along(x)) {
    chars <- toupper(strsplit(x[i], "", fixed = TRUE)[[1]])
    for (b in bases) {
      out[i, b] <- sum(chars == b)
    }
    codes <- chars[chars %in% names(iupac)]
    if (length(codes) == 0L) {
      next
    }
    switch(ambiguous,
      ignore = {
        next
      },
      other = {
        out[i, "other"] <- length(codes)
      },
      split = {
        for (code in codes) {
          share <- iupac[[code]]
          out[i, share] <- out[i, share] + 1 / length(share)
        }
      }
    )
  }

  out
}

#' Count dinucleotides
#'
#' @param x A single DNA sequence.
#' @param observed_expected Whether to divide each count by the count expected
#'   from the composition of the two bases.
#' @return A named numeric vector with the 16 dinucleotides `"AA"` to `"TT"`.
#' @export
dinucleotide_counts <- function(x, observed_expected = FALSE) {
  bases <- c("A", "C", "G", "T")
  pairs <- as.vector(outer(bases, bases, paste0))
  pairs <- sort(pairs)
  chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  n <- length(chars)
  out <- setNames(numeric(length(pairs)), pairs)
  if (n < 2L) {
    return(out)
  }

  # Count adjacent pairs that use only the four bases
  left <- chars[-n]
  right <- chars[-1]
  keep <- left %in% bases & right %in% bases
  observed <- table(factor(paste0(left[keep], right[keep]), levels = pairs))
  out[pairs] <- as.numeric(observed[pairs])
  if (!observed_expected) {
    return(out)
  }
  # Scale by the product of single-base frequencies
  freq <- table(factor(chars[chars %in% bases], levels = bases)) / sum(chars %in% bases)

  for (p in pairs) {
    first <- substr(p, 1, 1)
    second <- substr(p, 2, 2)
    expected <- freq[[first]] * freq[[second]] * sum(keep)
    out[[p]] <- if (expected > 0) out[[p]] / expected else NA_real_
  }

  out
}

#' Homopolymer run statistics
#'
#' @param x A single nucleotide sequence.
#' @param min_length The shortest run that counts as a homopolymer.
#' @return A list with elements `n_runs`, `longest`, `longest_base`,
#'   `mean_length` and `covered` (the fraction of bases inside runs).
#' @export
run_stats <- function(x, min_length = 3L) {
  if (!is.character(x) || length(x) != 1L) {
    stop(sprintf("`%s` must be a single string.", "x"), call. = FALSE)
  }
  if (!is.numeric(min_length) || length(min_length) != 1L || min_length < 2) {
    stop(sprintf("`%s` must be a single number of at least 2.", "min_length"), call. = FALSE)
  }
  chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  runs <- rle(chars)
  long <- runs$lengths >= min_length

  # n_runs
  n_runs <- sum(long)
  # longest
  longest <- if (n_runs > 0L) max(runs$lengths[long]) else 0L
  # longest_base
  longest_base <- if (n_runs > 0L) {
    runs$values[long][which.max(runs$lengths[long])]
  } else {
    NA_character_
  }
  # mean_length
  mean_length <- if (n_runs > 0L) mean(runs$lengths[long]) else NA_real_
  # covered
  covered <- if (length(chars) > 0L) sum(runs$lengths[long]) / length(chars) else NA_real_

  list(
    n_runs = n_runs,
    longest = longest,
    longest_base = longest_base,
    mean_length = mean_length,
    covered = covered

  )
}

#' GC and AT skew in sliding windows
#'
#' @param x A single DNA sequence.
#' @param width The window width.
#' @param step The step between window starts.
#' @return A data frame with columns `start`, `end`, `gc_skew` and `at_skew`.
#' @export
base_skew <- function(x, width = 100L, step = width) {
  chars <- toupper(strsplit(x, "", fixed = TRUE)[[1]])
  n <- length(chars)
  if (n < width) {
    stop(sprintf("`%s` must not be longer than the sequence.", "width"), call. = FALSE)
  }
  starts <- seq(1L, n - width + 1L, by = step)
  gc_skew <- numeric(length(starts))
  at_skew <- numeric(length(starts))

  for (i in seq_along(starts)) {
    win <- chars[starts[i]:(starts[i] + width - 1L)]
    # GC skew: (G - C) / (G + C)
    n_g <- sum(win == "G")
    n_c <- sum(win == "C")
    gc_skew[i] <- if (n_g + n_c > 0) (n_g - n_c) / (n_g + n_c) else NA_real_
    # AT skew: (A - T) / (A + T)
    n_a <- sum(win == "A")
    n_t <- sum(win == "T")
    at_skew[i] <- if (n_a + n_t > 0) (n_a - n_t) / (n_a + n_t) else NA_real_
  }

  data.frame(
    start = starts,
    end = starts + width - 1L,
    gc_skew = gc_skew,
    at_skew = at_skew

  )
}

#' Base counts at each alignment position
#'
#' @param seqs A character vector of aligned sequences of equal length.
#' @param type The scale of the result: `"count"`, `"frequency"`, `"percent"`
#'   or `"log_odds"` against a uniform background.
#' @param pseudocount Added to every cell before computing log odds.
#' @return A numeric matrix with one row per base and one column per position.
#' @export
positional_counts <- function(seqs, type = c("count", "frequency", "percent", "log_odds"),
                              pseudocount = 0.5) {
  type <- match.arg(type)
  widths <- unique(nchar(seqs))
  if (length(widths) != 1L) {
    stop(sprintf("`%s` must contain sequences of equal length.", "seqs"), call. = FALSE)
  }
  bases <- c("A", "C", "G", "T")
  counts <- matrix(0, nrow = length(bases), ncol = widths, dimnames = list(bases, NULL))
  chars <- do.call(rbind, strsplit(toupper(seqs), "", fixed = TRUE))

  for (pos in seq_len(widths)) {
    column <- chars[, pos]
    for (b in bases) {
      counts[b, pos] <- sum(column == b)
    }
  }
  totals <- colSums(counts)

  switch(type,
    count = {
      counts
    },
    frequency = {
      sweep(counts, 2, pmax(totals, 1), "/")
    },
    percent = {
      100 * sweep(counts, 2, pmax(totals, 1), "/")
    },
    log_odds = {
      freq <- sweep(counts + pseudocount, 2, totals + length(bases) * pseudocount, "/")
      log2(freq / (1 / length(bases)))
    }

  )
}

#' Count k-mers in a sequence
#'
#' @param x A single DNA sequence.
#' @param k The k-mer length.
#' @param step The step between k-mer starts.
#' @param canonical Whether to merge each k-mer with its reverse complement,
#'   keeping the lexically smaller of the two as the name.
#' @return A named integer vector of counts, sorted by k-mer.
#' @export
kmer_counts <- function(x, k = 3L, step = 1L, canonical = FALSE) {
  if (!is.numeric(k) || length(k) != 1L || k < 1) {
    stop(sprintf("`%s` must be a single positive number.", "k"), call. = FALSE)
  }
  if (!is.numeric(step) || length(step) != 1L || step < 1) {
    stop(sprintf("`%s` must be a single positive number.", "step"), call. = FALSE)
  }
  x <- toupper(x)
  n <- nchar(x)
  if (n < k) {
    return(integer(0))
  }
  starts <- seq(1L, n - k + 1L, by = step)
  kmers <- substring(x, starts, starts + k - 1L)
  kmers <- kmers[!grepl("[^ACGT]", kmers)]

  if (canonical) {
    # Reverse complement each k-mer and keep the smaller spelling
    rc <- chartr("ACGT", "TGCA", kmers)
    rc <- vapply(strsplit(rc, "", fixed = TRUE), function(s) {
      paste(rev(s), collapse = "")
    }, character(1))
    kmers <- ifelse(kmers <= rc, kmers, rc)
  }
  counts <- table(kmers)

  setNames(as.integer(counts), names(counts))
}
