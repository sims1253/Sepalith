#' Grouping key for interval set operations
#'
#' Combines sequence name and (optionally) strand into one key per interval.
#'
#' @param x A `SpanSet` object.
#' @param ignoreStrand Should strand be ignored?
#' @return A character vector with one key per interval.
#' @noRd
.groupKey = function(x, ignoreStrand) {
  if (ignoreStrand) {
    return(paste(x@seqnames, "*", sep = "|"))
  }
  return(paste(x@seqnames, x@strand, sep = "|"))
}

#' Merge sorted intervals on one sequence
#'
#' Walks intervals in start order and merges those that overlap or lie
#' within `minGap` bases of the running block.
#'
#' @param starts,ends Integer vectors of interval coordinates.
#' @param minGap Gaps of at most this many bases are closed.
#' @return A two-column integer matrix of merged `start` and `end`.
#' @noRd
.mergeIntervals = function(starts, ends, minGap = 0L) {
  if (length(starts) == 0L) {
    return(matrix(integer(), ncol = 2L, dimnames = list(NULL, c("start", "end"))))
  }
  ord = order(starts, ends)
  starts = starts[ord]
  ends = ends[ord]
  outStart = integer(length(starts))
  outEnd = integer(length(starts))
  nOut = 1L
  outStart[1L] = starts[1L]
  outEnd[1L] = ends[1L]
  for (i in seq_along(starts)[-1L]) {
    if (starts[i] <= outEnd[nOut] + minGap + 1L) {
      outEnd[nOut] = max(outEnd[nOut], ends[i])
    } else {
      nOut = nOut + 1L
      outStart[nOut] = starts[i]
      outEnd[nOut] = ends[i]
    }
  }
  return(cbind(start = outStart[seq_len(nOut)], end = outEnd[seq_len(nOut)]))
}

#' Rebuild a SpanSet from per-group coordinate matrices
#'
#' @param pieces A named list of two-column matrices, named by group key.
#' @return A `SpanSet` object without metadata columns.
#' @noRd
.spansFromPieces = function(pieces) {
  pieces = pieces[vapply(pieces, nrow, integer(1)) > 0L]
  if (length(pieces) == 0L) {
    return(SpanSet(character(), start = integer(), end = integer()))
  }
  sizes = vapply(pieces, nrow, integer(1))
  keys = strsplit(rep(names(pieces), sizes), "|", fixed = TRUE)
  coords = do.call(rbind, pieces)
  return(SpanSet(
    seqnames = vapply(keys, `[`, character(1), 1L),
    start = coords[, "start"],
    end = coords[, "end"],
    strand = vapply(keys, `[`, character(1), 2L)
  ))
}

#' Merge overlapping intervals
#'
#' Collapses overlapping or adjacent intervals within each sequence and
#' strand into single intervals.
#'
#' @param x A `SpanSet` object.
#' @param minGap Gaps of at most this many bases are closed.
#' @param ignoreStrand Should strand be ignored?
#' @return A `SpanSet` object without metadata columns.
#' @export
reduceSpans = function(x, minGap = 0L, ignoreStrand = FALSE) {
  key = .groupKey(x, ignoreStrand)
  pieces = list()
  for (k in unique(key)) {
    idx = which(key == k)
    pieces[[k]] = .mergeIntervals(x@start[idx], x@end[idx], minGap)
  }
  return(.spansFromPieces(pieces))
}

#' Gaps between intervals
#'
#' Returns the uncovered stretches between reduced intervals within each
#' sequence and strand, optionally bounded by sequence lengths.
#'
#' @param x A `SpanSet` object.
#' @param seqLengths Optional named integer vector of sequence lengths; when
#'   given, the leading and trailing gaps are included.
#' @param ignoreStrand Should strand be ignored?
#' @return A `SpanSet` object without metadata columns.
#' @export
gapSpans = function(x, seqLengths = NULL, ignoreStrand = FALSE) {
  key = .groupKey(x, ignoreStrand)
  pieces = list()
  for (k in unique(key)) {
    idx = which(key == k)
    merged = .mergeIntervals(x@start[idx], x@end[idx])
    gapStart = merged[-nrow(merged), "end"] + 1L
    gapEnd = merged[-1L, "start"] - 1L
    seqName = x@seqnames[idx[1L]]
    if (!is.null(seqLengths) && seqName %in% names(seqLengths)) {
      gapStart = c(1L, gapStart, merged[nrow(merged), "end"] + 1L)
      gapEnd = c(merged[1L, "start"] - 1L, gapEnd, as.integer(seqLengths[[seqName]]))
    }
    keep = gapEnd >= gapStart
    pieces[[k]] = cbind(start = gapStart[keep], end = gapEnd[keep])
  }
  return(.spansFromPieces(pieces))
}

#' Union of two interval sets
#'
#' @param x,y `SpanSet` objects.
#' @param ignoreStrand Should strand be ignored?
#' @return A reduced `SpanSet` covering every base in `x` or `y`.
#' @family span set operations
#' @export
unionSpans = function(x, y, ignoreStrand = FALSE) {
  both = SpanSet(
    seqnames = c(x@seqnames, y@seqnames),
    start = c(x@start, y@start),
    end = c(x@end, y@end),
    strand = c(x@strand, y@strand)
  )
  return(reduceSpans(both, ignoreStrand = ignoreStrand))
}

#' Intersection of two interval sets
#'
#' @param x,y `SpanSet` objects.
#' @param ignoreStrand Should strand be ignored?
#' @return A reduced `SpanSet` covering every base in both `x` and `y`.
#' @family span set operations
#' @export
intersectSpans = function(x, y, ignoreStrand = FALSE) {
  rx = reduceSpans(x, ignoreStrand = ignoreStrand)
  ry = reduceSpans(y, ignoreStrand = ignoreStrand)
  hits = findSpanOverlaps(rx, ry, ignoreStrand = TRUE)
  if (!ignoreStrand) {
    hits = hits[rx@strand[hits$queryHits] == ry@strand[hits$subjectHits], , drop = FALSE]
  }
  if (nrow(hits) == 0L) {
    return(SpanSet(character(), start = integer(), end = integer()))
  }
  return(SpanSet(
    seqnames = rx@seqnames[hits$queryHits],
    start = pmax(rx@start[hits$queryHits], ry@start[hits$subjectHits]),
    end = pmin(rx@end[hits$queryHits], ry@end[hits$subjectHits]),
    strand = rx@strand[hits$queryHits]
  ))
}

#' Difference of two interval sets
#'
#' @param x,y `SpanSet` objects.
#' @param ignoreStrand Should strand be ignored?
#' @return A reduced `SpanSet` covering bases in `x` but not in `y`.
#' @family span set operations
#' @export
setdiffSpans = function(x, y, ignoreStrand = FALSE) {
  rx = reduceSpans(x, ignoreStrand = ignoreStrand)
  ry = reduceSpans(y, ignoreStrand = ignoreStrand)
  hits = findSpanOverlaps(rx, ry, ignoreStrand = ignoreStrand)
  outStart = integer()
  outEnd = integer()
  outIdx = integer()
  for (i in seq_along(rx@start)) {
    cuts = hits$subjectHits[hits$queryHits == i]
    cuts = cuts[order(ry@start[cuts])]
    cursor = rx@start[i]
    for (s in cuts) {
      if (ry@start[s] > cursor) {
        outStart = c(outStart, cursor)
        outEnd = c(outEnd, ry@start[s] - 1L)
        outIdx = c(outIdx, i)
      }
      cursor = max(cursor, ry@end[s] + 1L)
    }
    if (cursor <= rx@end[i]) {
      outStart = c(outStart, cursor)
      outEnd = c(outEnd, rx@end[i])
      outIdx = c(outIdx, i)
    }
  }
  return(SpanSet(
    seqnames = rx@seqnames[outIdx],
    start = outStart,
    end = outEnd,
    strand = rx@strand[outIdx]
  ))
}
