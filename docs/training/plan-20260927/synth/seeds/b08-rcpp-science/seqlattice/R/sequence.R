#' Check that input is a vector of DNA sequences
#'
#' Validates type and alphabet before sequences are handed to compiled code,
#' which assumes clean input and does no checking of its own.
#'
#' @param seqs A character vector of DNA sequences.
#' @param arg Name of the argument, used in error messages.
#' @return `seqs`, invisibly and upper-cased, if all checks pass.
#' @keywords internal
check_dna <- function(seqs, arg = "seqs") {
  if (!is.character(seqs)) {
    stop(sprintf("`%s` must be a character vector, not %s.", arg, class(seqs)[1]),
      call. = FALSE)
  }
  if (anyNA(seqs)) {
    stop(sprintf("`%s` must not contain missing values.", arg), call. = FALSE)
  }
  seqs <- toupper(seqs)
  bad <- grepl("[^ACGTN]", seqs)
  if (any(bad)) {
    stop(sprintf("`%s` contains non-ACGTN characters at position(s) %s.", arg,
      paste(which(bad), collapse = ", ")), call. = FALSE)
  }
  return(invisible(seqs))
}

#' Check a positive integer scalar
#'
#' Used for window widths, step sizes and k-mer lengths before `.Call()`.
#'
#' @param x Value to check.
#' @param arg Name of the argument, used in error messages.
#' @param max Largest allowed value.
#' @return `x` coerced to integer.
#' @keywords internal
check_count <- function(x, arg, max = .Machine$integer.max) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be a single number.", arg), call. = FALSE)
  }
  if (x < 1 || x != round(x)) {
    stop(sprintf("`%s` must be a positive whole number.", arg), call. = FALSE)
  }
  if (x > max) {
    stop(sprintf("`%s` must be at most %d.", arg, max), call. = FALSE)
  }
  return(as.integer(x))
}

#' GC content of DNA sequences
#'
#' Computes the fraction of G and C bases in each sequence. Ambiguous `N`
#' bases are excluded from the denominator.
#'
#' @param seqs A character vector of DNA sequences.
#' @param na_if_empty Logical; return `NA` for sequences with no called bases?
#' @return A numeric vector of GC fractions, named like `seqs`.
#' @export
#' @examples
#' gc_content(c(a = "ACGT", b = "GGCCNN"))
gc_content <- function(seqs, na_if_empty = TRUE) {
  seqs <- check_dna(seqs)
  counts <- base_counts_cpp(seqs, TRUE)
  called <- rowSums(counts[, c("A", "C", "G", "T"), drop = FALSE])
  gc <- (counts[, "G"] + counts[, "C"]) / called
  if (na_if_empty) {
    gc[called == 0] <- NA_real_
  } else {
    gc[called == 0] <- 0
  }
  names(gc) <- names(seqs)
  return(gc)
}

#' GC skew of DNA sequences
#'
#' Computes `(G - C) / (G + C)` for each sequence, a common signal for
#' locating replication origins in bacterial genomes.
#'
#' @param seqs A character vector of DNA sequences.
#' @param na_if_empty Logical; return `NA` for sequences with no G or C bases?
#' @return A numeric vector of GC skews, named like `seqs`.
#' @export
gc_skew <- function(seqs, na_if_empty = TRUE) {
  seqs <- check_dna(seqs)
  counts <- base_counts_cpp(seqs, TRUE)
  total <- counts[, "G"] + counts[, "C"]
  skew <- (counts[, "G"] - counts[, "C"]) / total
  if (na_if_empty) {
    skew[total == 0] <- NA_real_
  } else {
    skew[total == 0] <- 0
  }
  names(skew) <- names(seqs)
  return(skew)
}

#' Sliding-window GC profile
#'
#' Splits a single sequence into windows and returns the GC fraction of each,
#' together with window coordinates.
#'
#' @param seq A single DNA sequence as a character string.
#' @param width Integer window width in bases.
#' @param step Integer step between window starts. Defaults to `width`, giving
#'   non-overlapping windows.
#' @return A data frame with columns `start`, `end` and `gc`.
#' @export
gc_profile <- function(seq, width = 1000L, step = width) {
  if (length(seq) != 1L) {
    stop("`seq` must be a single sequence.", call. = FALSE)
  }
  seq <- check_dna(seq, arg = "seq")
  len <- nchar(seq)
  width <- check_count(width, "width", max = len)
  step <- check_count(step, "step", max = width)
  gc <- gc_window_cpp(seq, width, step)
  starts <- seq.int(1L, by = step, length.out = length(gc))
  return(data.frame(
    start = starts,
    end = starts + width - 1L,
    gc = gc
  ))
}

#' Reverse complement of DNA sequences
#'
#' @param seqs A character vector of DNA sequences.
#' @return A character vector the same length as `seqs`, with names kept.
#' @export
#' @examples
#' reverse_complement("AACGTT")
reverse_complement <- function(seqs) {
  seqs <- check_dna(seqs)
  out <- reverse_complement_cpp(seqs)
  names(out) <- names(seqs)
  return(out)
}

#' Tabulate k-mers
#'
#' Counts every overlapping k-mer across a set of sequences, optionally
#' folding each k-mer together with its reverse complement.
#'
#' @param seqs A character vector of DNA sequences.
#' @param k Integer k-mer length.
#' @param canonical Logical; merge each k-mer with its reverse complement?
#' @param skip_n Logical; drop k-mers containing an `N`?
#' @return A data frame with columns `kmer` and `count`, sorted by decreasing
#'   count.
#' @export
kmer_table <- function(seqs, k = 4L, canonical = FALSE, skip_n = TRUE) {
  seqs <- check_dna(seqs)
  k <- check_count(k, "k", max = 16L)
  if (all(nchar(seqs) < k)) {
    return(data.frame(kmer = character(), count = integer()))
  }
  counts <- kmer_count_cpp(seqs, k, skip_n)
  if (canonical) {
    kmers <- names(counts)
    rc <- reverse_complement_cpp(kmers)
    key <- ifelse(kmers < rc, kmers, rc)
    counts <- tapply(counts, key, sum)
  }
  out <- data.frame(
    kmer = names(counts),
    count = as.integer(counts),
    stringsAsFactors = FALSE
  )
  out <- out[order(-out$count, out$kmer), , drop = FALSE]
  rownames(out) <- NULL
  return(out)
}
