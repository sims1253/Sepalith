#' Summarise spans
#'
#' @param object An object of class `spanwise_span`.
#' @param ... Unused, for compatibility with the generic.
#' @return An object of class `summary_spanwise_span`: a list with elements
#'   `n`, `total_width`, `min_width`, `max_width`, `extent` and `n_overlaps`.
#' @export
summary.spanwise_span <- function(object, ...) {
  fields <- unclass(object)
  n <- length(fields$start)
  widths <- fields$end - fields$start

  # Count spans that start before an earlier span has ended
  n_overlaps <- 0L
  if (n > 1L) {
    ord <- order(fields$start)
    starts <- fields$start[ord]
    ends <- fields$end[ord]
    n_overlaps <- sum(starts[-1L] < cummax(ends)[-n])
  }

  out <- list(
    n = n,
    total_width = sum(widths),
    min_width = if (n > 0L) min(widths) else NA_real_,
    max_width = if (n > 0L) max(widths) else NA_real_,
    extent = if (n > 0L) c(min(fields$start), max(fields$end)) else rep(NA_real_, 2L),
    n_overlaps = n_overlaps
  )
  return(structure(out, class = "summary_spanwise_span"))
}

#' Summarise gaps
#'
#' @param object An object of class `spanwise_gap`.
#' @param ... Unused, for compatibility with the generic.
#' @return An object of class `summary_spanwise_gap`: a list with elements
#'   `n`, `total_width`, `min_width`, `max_width`, `extent` and `n_empty`.
#' @export
summary.spanwise_gap <- function(object, ...) {
  fields <- unclass(object)
  n <- length(fields$at)
  widths <- fields$width

  # Count gaps of zero width
  n_empty <- 0L
  if (n > 0L) {
    n_empty <- sum(widths == 0)
  }

  out <- list(
    n = n,
    total_width = sum(widths),
    min_width = if (n > 0L) min(widths) else NA_real_,
    max_width = if (n > 0L) max(widths) else NA_real_,
    extent = if (n > 0L) c(min(fields$at), max(fields$at + widths)) else rep(NA_real_, 2L),
    n_empty = n_empty
  )
  return(structure(out, class = "summary_spanwise_gap"))
}

#' Print a span summary
#'
#' @param x An object of class `summary_spanwise_span`.
#' @param digits Number of significant digits used for widths.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.summary_spanwise_span <- function(x, digits = 4, ...) {
  cat(sprintf("%-12s%d\n", "Spans", x$n))
  if (x$n == 0L) {
    return(invisible(x))
  }
  rows <- c(
    "Total width" = x$total_width,
    "Shortest" = x$min_width,
    "Longest" = x$max_width,
    "Extent from" = x$extent[1L],
    "Extent to" = x$extent[2L]
  )
  labels <- formatC(names(rows), width = -12L)
  cat(paste0(labels, format(rows, digits = digits)), sep = "\n")
  cat(sprintf("%-12s%d\n", "Overlaps", x$n_overlaps))
  return(invisible(x))
}

#' Print a gap summary
#'
#' @param x An object of class `summary_spanwise_gap`.
#' @param digits Number of significant digits used for widths.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.summary_spanwise_gap <- function(x, digits = 4, ...) {
  cat(sprintf("%-12s%d\n", "Gaps", x$n))
  if (x$n == 0L) {
    return(invisible(x))
  }
  rows <- c(
    "Total width" = x$total_width,
    "Shortest" = x$min_width,
    "Longest" = x$max_width,
    "Extent from" = x$extent[1L],
    "Extent to" = x$extent[2L]
  )
  labels <- formatC(names(rows), width = -12L)
  cat(paste0(labels, format(rows, digits = digits)), sep = "\n")
  cat(sprintf("%-12s%d\n", "Empty", x$n_empty))
  return(invisible(x))
}

#' Convert spans to a data frame
#'
#' @param x An object of class `spanwise_span`.
#' @param row.names Optional row names for the result.
#' @param optional Ignored.
#' @param ... Unused.
#' @return A data frame with columns `label`, `start`, `end` and `width`.
#' @export
as.data.frame.spanwise_span <- function(x, row.names = NULL, optional = FALSE, ...) {
  fields <- unclass(x)
  out <- data.frame(
    label = fields$label,
    start = fields$start,
    end = fields$end,
    width = fields$end - fields$start,
    stringsAsFactors = FALSE
  )
  if (!is.null(row.names)) {
    rownames(out) <- row.names
  }
  return(out)
}

#' Convert gaps to a data frame
#'
#' @param x An object of class `spanwise_gap`.
#' @param row.names Optional row names for the result.
#' @param optional Ignored.
#' @param ... Unused.
#' @return A data frame with columns `at`, `width` and `end`.
#' @export
as.data.frame.spanwise_gap <- function(x, row.names = NULL, optional = FALSE, ...) {
  fields <- unclass(x)
  out <- data.frame(
    at = fields$at,
    width = fields$width,
    end = fields$at + fields$width,
    stringsAsFactors = FALSE
  )
  if (!is.null(row.names)) {
    rownames(out) <- row.names
  }
  return(out)
}
