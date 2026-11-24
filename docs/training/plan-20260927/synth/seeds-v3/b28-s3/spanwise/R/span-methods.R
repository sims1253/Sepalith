#' Format spans
#'
#' Each span is rendered as `label [start, end]`.
#'
#' @param x An object of class `spanwise_span`.
#' @param digits Number of significant digits used for positions.
#' @param ... Unused, for compatibility with the generic.
#' @return A character vector with one element per span.
#' @export
format.spanwise_span <- function(x, digits = 4, ...) {
  fields <- unclass(x)
  if (length(fields$start) == 0L) {
    return(character())
  }
  start <- format(fields$start, digits = digits, trim = TRUE)
  end <- format(fields$end, digits = digits, trim = TRUE)
  out <- sprintf("%s [%s, %s]", fields$label, start, end)
  return(out)
}

#' Format gaps
#'
#' Each gap is rendered as `at (+width)`.
#'
#' @param x An object of class `spanwise_gap`.
#' @inheritParams format.spanwise_span
#' @return A character vector with one element per gap.
#' @export
format.spanwise_gap <- function(x, digits = 4, ...) {
  fields <- unclass(x)
  if (length(fields$at) == 0L) {
    return(character())
  }
  at <- format(fields$at, digits = digits, trim = TRUE)
  width <- format(fields$width, digits = digits, trim = TRUE)
  out <- sprintf("%s (+%s)", at, width)
  return(out)
}

#' Print spans
#'
#' Prints a `<spanwise_span[n]>` header and at most `max` formatted spans.
#'
#' @param x An object of class `spanwise_span`.
#' @param ... Passed to [format.spanwise_span()].
#' @param max Maximum number of spans to show.
#' @return `x`, invisibly.
#' @export
print.spanwise_span <- function(x, ..., max = 10L) {
  n <- length(unclass(x)$start)
  cat(sprintf("<spanwise_span[%d]>\n", n))
  if (n == 0L) {
    return(invisible(x))
  }
  shown <- format(x, ...)
  if (n > max) {
    shown <- c(shown[seq_len(max)], sprintf("... and %d more", n - max))
  }
  cat(paste0("  ", shown), sep = "\n")
  return(invisible(x))
}

#' Print gaps
#'
#' Prints a `<spanwise_gap[n]>` header and at most `max` formatted gaps.
#'
#' @param x An object of class `spanwise_gap`.
#' @param ... Passed to [format.spanwise_gap()].
#' @param max Maximum number of gaps to show.
#' @return `x`, invisibly.
#' @export
print.spanwise_gap <- function(x, ..., max = 10L) {
  n <- length(unclass(x)$at)
  cat(sprintf("<spanwise_gap[%d]>\n", n))
  if (n == 0L) {
    return(invisible(x))
  }
  shown <- format(x, ...)
  if (n > max) {
    shown <- c(shown[seq_len(max)], sprintf("... and %d more", n - max))
  }
  cat(paste0("  ", shown), sep = "\n")
  return(invisible(x))
}

#' Number of spans
#'
#' @param x An object of class `spanwise_span`.
#' @return A single integer.
#' @export
length.spanwise_span <- function(x) {
  return(length(unclass(x)$start))
}

#' Number of gaps
#'
#' @param x An object of class `spanwise_gap`.
#' @return A single integer.
#' @export
length.spanwise_gap <- function(x) {
  return(length(unclass(x)$at))
}

#' Subset spans
#'
#' @param x An object of class `spanwise_span`.
#' @param i Positions, a logical vector, or span labels.
#' @return An object of class `spanwise_span`.
#' @export
`[.spanwise_span` <- function(x, i) {
  fields <- unclass(x)
  if (missing(i)) {
    return(x)
  }
  # Character subscripts select spans by label
  if (is.character(i)) {
    i <- match(i, fields$label)
  }
  if (anyNA(i)) {
    stop("Subscript out of bounds.", call. = FALSE)
  }
  out <- new_span(fields$start[i], fields$end[i], fields$label[i])
  return(out)
}

#' Subset gaps
#'
#' @param x An object of class `spanwise_gap`.
#' @param i Positions or a logical vector.
#' @return An object of class `spanwise_gap`.
#' @export
`[.spanwise_gap` <- function(x, i) {
  fields <- unclass(x)
  if (missing(i)) {
    return(x)
  }
  if (is.character(i)) {
    stop("Gaps have no labels to subset by.", call. = FALSE)
  }
  if (anyNA(i)) {
    stop("Subscript out of bounds.", call. = FALSE)
  }
  out <- new_gap(fields$at[i], fields$width[i])
  return(out)
}
