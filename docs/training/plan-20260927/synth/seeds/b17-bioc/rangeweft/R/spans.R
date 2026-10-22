#' Genomic interval set
#'
#' Stores closed 1-based intervals on named sequences together with strand
#' and a data frame of per-interval metadata.
#'
#' @slot seqnames Character vector of sequence names.
#' @slot start Integer vector of 1-based start positions.
#' @slot end Integer vector of 1-based end positions (inclusive).
#' @slot strand Character vector with values `"+"`, `"-"` or `"*"`.
#' @slot mcols A data frame of metadata, one row per interval.
#' @export
methods::setClass(
  "SpanSet",
  methods::representation(
    seqnames = "character",
    start = "integer",
    end = "integer",
    strand = "character",
    mcols = "data.frame"
  )
)

methods::setValidity("SpanSet", function(object) {
  n = length(object@start)
  problems = character()
  lens = c(length(object@seqnames), length(object@end), length(object@strand))
  if (any(lens != n)) {
    problems = c(problems, "seqnames, start, end and strand must have equal length")
  }
  if (nrow(object@mcols) != n) {
    problems = c(problems, "mcols must have one row per interval")
  }
  if (any(object@end < object@start - 1L, na.rm = TRUE)) {
    problems = c(problems, "end must be >= start - 1")
  }
  if (!all(object@strand %in% c("+", "-", "*"))) {
    problems = c(problems, "strand must be one of '+', '-', '*'")
  }
  if (length(problems) == 0L) TRUE else problems
})

#' Create a SpanSet
#'
#' Builds an interval set from positions and widths, recycling scalar
#' `seqnames` and `strand` to the number of intervals.
#'
#' @param seqnames Character vector of sequence names.
#' @param start Integer vector of 1-based start positions.
#' @param end Integer vector of 1-based end positions; computed from `width`
#'   when `NULL`.
#' @param width Integer vector of interval widths, used when `end` is `NULL`.
#' @param strand Character vector with values `"+"`, `"-"` or `"*"`.
#' @param ... Named vectors stored as metadata columns.
#' @return A `SpanSet` object.
#' @export
#' @examples
#' SpanSet("chr1", start = c(100, 250), width = 50, gene = c("a", "b"))
SpanSet = function(seqnames, start, end = NULL, width = NULL, strand = "*", ...) {
  start = as.integer(start)
  n = length(start)
  if (is.null(end)) {
    if (is.null(width)) {
      stop("supply either `end` or `width`", call. = FALSE)
    }
    end = start + as.integer(rep_len(width, n)) - 1L
  } else {
    end = as.integer(end)
  }
  mcols = data.frame(..., stringsAsFactors = FALSE)
  if (ncol(mcols) == 0L) {
    mcols = data.frame(row.names = seq_len(n))
  }
  obj = methods::new(
    "SpanSet",
    seqnames = rep_len(as.character(seqnames), n),
    start = start,
    end = end,
    strand = rep_len(as.character(strand), n),
    mcols = mcols
  )
  return(obj)
}

#' Interval start positions
#'
#' @param x A `SpanSet` object.
#' @return An integer vector.
#' @family span accessors
#' @export
spanStart = function(x) {
  return(x@start)
}

#' Interval end positions
#'
#' @param x A `SpanSet` object.
#' @return An integer vector.
#' @family span accessors
#' @export
spanEnd = function(x) {
  return(x@end)
}

#' Interval widths
#'
#' @param x A `SpanSet` object.
#' @return An integer vector.
#' @family span accessors
#' @export
spanWidth = function(x) {
  return(x@end - x@start + 1L)
}

#' Subset a SpanSet by index
#'
#' Keeps the intervals selected by `i`, along with their metadata rows.
#'
#' @param x A `SpanSet` object.
#' @param i Logical or integer index of intervals to keep.
#' @return A `SpanSet` object.
#' @export
spanSubset = function(x, i) {
  if (is.logical(i)) {
    i = which(rep_len(i, length(x@start)))
  }
  x@seqnames = x@seqnames[i]
  x@start = x@start[i]
  x@end = x@end[i]
  x@strand = x@strand[i]
  x@mcols = x@mcols[i, , drop = FALSE]
  return(x)
}

#' Resize intervals around an anchor
#'
#' Changes interval widths while keeping the start, end or centre fixed.
#' Strand is respected for `"start"` and `"end"` anchors.
#'
#' @param x A `SpanSet` object.
#' @param width New width, recycled to the number of intervals.
#' @param fix Which point of each interval to keep fixed.
#' @return A `SpanSet` object.
#' @export
resizeSpans = function(x, width, fix = c("start", "end", "center")) {
  fix = match.arg(fix)
  width = as.integer(rep_len(width, length(x@start)))
  minus = x@strand == "-"
  newStart = x@start
  newEnd = x@end
  if (fix == "start") {
    newEnd[!minus] = x@start[!minus] + width[!minus] - 1L
    newStart[minus] = x@end[minus] - width[minus] + 1L
  } else if (fix == "end") {
    newStart[!minus] = x@end[!minus] - width[!minus] + 1L
    newEnd[minus] = x@start[minus] + width[minus] - 1L
  } else {
    mid = (x@start + x@end) %/% 2L
    newStart = mid - (width - 1L) %/% 2L
    newEnd = newStart + width - 1L
  }
  x@start = as.integer(newStart)
  x@end = as.integer(newEnd)
  methods::validObject(x)
  return(x)
}

#' Flanking regions of intervals
#'
#' Returns regions of `width` bases immediately upstream or downstream of
#' each interval, following strand when `ignoreStrand` is `FALSE`.
#'
#' @param x A `SpanSet` object.
#' @param width Width of the flanking region.
#' @param side Whether to take the upstream or downstream flank.
#' @param ignoreStrand Treat every interval as if on the `"+"` strand?
#' @return A `SpanSet` object.
#' @export
flankSpans = function(x, width, side = c("upstream", "downstream"), ignoreStrand = FALSE) {
  side = match.arg(side)
  width = as.integer(rep_len(width, length(x@start)))
  minus = if (ignoreStrand) rep(FALSE, length(x@start)) else x@strand == "-"
  takeLeft = if (side == "upstream") !minus else minus
  newStart = ifelse(takeLeft, x@start - width, x@end + 1L)
  newEnd = ifelse(takeLeft, x@start - 1L, x@end + width)
  x@start = as.integer(pmax(newStart, 1L))
  x@end = as.integer(newEnd)
  return(x)
}
