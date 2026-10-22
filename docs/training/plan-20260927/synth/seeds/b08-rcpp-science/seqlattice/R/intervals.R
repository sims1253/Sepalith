#' Create a set of genomic intervals
#'
#' Builds a `seq_intervals` object from chromosome names and 1-based,
#' closed start and end coordinates.
#'
#' @param chrom A character vector of chromosome or contig names.
#' @param start Integer vector of 1-based start positions.
#' @param end Integer vector of 1-based end positions (inclusive).
#' @param name Optional character vector of interval labels.
#' @return An object of class `seq_intervals`.
#' @export
seq_intervals <- function(chrom, start, end, name = NULL) {
  n <- length(chrom)
  if (length(start) != n || length(end) != n) {
    stop("`chrom`, `start` and `end` must have the same length.", call. = FALSE)
  }
  if (is.null(name)) {
    name <- sprintf("iv%d", seq_len(n))
  }
  x <- data.frame(
    chrom = as.character(chrom),
    start = as.integer(start),
    end = as.integer(end),
    name = as.character(name),
    stringsAsFactors = FALSE
  )
  validate_intervals(x)
  x <- x[order(x$chrom, x$start, x$end), , drop = FALSE]
  rownames(x) <- NULL
  return(structure(x, class = c("seq_intervals", "data.frame")))
}

#' Validate interval coordinates
#'
#' @param x A data frame with `chrom`, `start` and `end` columns.
#' @return `x`, invisibly, if the coordinates are valid.
#' @keywords internal
validate_intervals <- function(x) {
  if (anyNA(x$start) || anyNA(x$end)) {
    stop("Interval coordinates must not be missing.", call. = FALSE)
  }
  if (any(x$start < 1L)) {
    stop("Interval starts must be 1 or greater.", call. = FALSE)
  }
  bad <- which(x$end < x$start)
  if (length(bad) > 0L) {
    stop(sprintf("Interval end precedes start in row(s) %s.",
      paste(utils::head(bad, 5L), collapse = ", ")), call. = FALSE)
  }
  return(invisible(x))
}

#' Interval widths
#'
#' @param x A `seq_intervals` object.
#' @return An integer vector of widths in bases.
#' @export
interval_width <- function(x) {
  return(x$end - x$start + 1L)
}

#' Find overlapping intervals
#'
#' Reports every pair of query and subject intervals on the same chromosome
#' that share at least `min_overlap` bases. The sweep itself runs in C++.
#'
#' @param query A `seq_intervals` object.
#' @param subject A `seq_intervals` object.
#' @param min_overlap Integer minimum number of shared bases.
#' @return A data frame with columns `query`, `subject` and `overlap`.
#' @export
find_overlaps <- function(query, subject, min_overlap = 1L) {
  stopifnot(inherits(query, "seq_intervals"), inherits(subject, "seq_intervals"))
  min_overlap <- check_count(min_overlap, "min_overlap")
  chroms <- intersect(unique(query$chrom), unique(subject$chrom))
  hits <- vector("list", length(chroms))
  for (i in seq_along(chroms)) {
    qi <- which(query$chrom == chroms[i])
    si <- which(subject$chrom == chroms[i])
    m <- interval_overlap_cpp(query$start[qi], query$end[qi],
      subject$start[si], subject$end[si], min_overlap)
    hits[[i]] <- cbind(qi[m[, 1]], si[m[, 2]])
  }
  hits <- do.call(rbind, hits)
  if (is.null(hits) || nrow(hits) == 0L) {
    return(data.frame(query = integer(), subject = integer(), overlap = integer()))
  }
  overlap <- pmin(query$end[hits[, 1]], subject$end[hits[, 2]]) -
    pmax(query$start[hits[, 1]], subject$start[hits[, 2]]) + 1L
  return(data.frame(query = hits[, 1], subject = hits[, 2], overlap = overlap))
}

#' Merge nearby intervals
#'
#' Collapses intervals on the same chromosome that overlap or are separated
#' by at most `gap` bases.
#'
#' @param x A `seq_intervals` object.
#' @param gap Integer maximum gap between intervals that are merged.
#' @return A `seq_intervals` object with merged intervals and fresh names.
#' @export
merge_intervals <- function(x, gap = 0L) {
  stopifnot(inherits(x, "seq_intervals"))
  if (!is.numeric(gap) || length(gap) != 1L || gap < 0) {
    stop("`gap` must be a single non-negative number.", call. = FALSE)
  }
  pieces <- lapply(split(x, x$chrom), function(part) {
    m <- interval_merge_cpp(part$start, part$end, as.integer(gap))
    data.frame(chrom = part$chrom[1], start = m[, 1], end = m[, 2])
  })
  merged <- do.call(rbind, pieces)
  return(seq_intervals(merged$chrom, merged$start, merged$end))
}

#' Format genomic intervals
#'
#' @param x A `seq_intervals` object.
#' @param ... Unused, for compatibility with [format()].
#' @return A character vector in `chrom:start-end` notation.
#' @export
format.seq_intervals <- function(x, ...) {
  return(sprintf("%s:%s-%s", x$chrom,
    format(x$start, big.mark = ",", trim = TRUE),
    format(x$end, big.mark = ",", trim = TRUE)))
}

#' Print genomic intervals
#'
#' @param x A `seq_intervals` object.
#' @param n Maximum number of intervals to show.
#' @param ... Unused, for compatibility with [print()].
#' @return `x`, invisibly.
#' @export
print.seq_intervals <- function(x, n = 10L, ...) {
  total <- nrow(x)
  cat(sprintf("<seq_intervals: %d interval(s) on %d chromosome(s)>\n",
    total, length(unique(x$chrom))))
  shown <- utils::head(seq_len(total), n)
  if (length(shown) > 0L) {
    labels <- format(x[shown, , drop = FALSE])
    cat(paste0("  ", format(x$name[shown]), "  ", labels), sep = "\n")
  }
  if (total > n) {
    cat(sprintf("  ... and %d more\n", total - n))
  }
  return(invisible(x))
}

#' Summarise genomic intervals
#'
#' @param object A `seq_intervals` object.
#' @param ... Unused, for compatibility with [summary()].
#' @return A data frame with one row per chromosome giving the interval count,
#'   total bases covered and median width.
#' @export
summary.seq_intervals <- function(object, ...) {
  widths <- interval_width(object)
  by_chrom <- split(widths, object$chrom)
  return(data.frame(
    chrom = names(by_chrom),
    n = vapply(by_chrom, length, integer(1)),
    bases = vapply(by_chrom, sum, numeric(1)),
    median_width = vapply(by_chrom, stats::median, numeric(1)),
    row.names = NULL
  ))
}
