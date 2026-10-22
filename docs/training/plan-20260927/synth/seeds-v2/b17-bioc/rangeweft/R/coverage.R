#' Per-base coverage on one sequence
#'
#' Counts how many intervals cover each base of a sequence using a
#' difference array, then compresses the result with [rle()].
#'
#' @param starts,ends Integer vectors of interval coordinates.
#' @param seqLength Length of the sequence in bases.
#' @param weights Numeric weights, one per interval.
#' @return An object of class `rle`.
#' @noRd
.coverageOnSeq = function(starts, ends, seqLength, weights) {
  delta = numeric(seqLength + 1L)
  # Clip to the sequence; intervals entirely outside it drop out.
  starts = pmax(starts, 1L)
  ends = pmin(ends, seqLength)
  ok = ends >= starts

  for (i in which(ok)) {
    delta[starts[i]] = delta[starts[i]] + weights[i]
    delta[ends[i] + 1L] = delta[ends[i] + 1L] - weights[i]
  }

  depth = cumsum(delta)[seq_len(seqLength)]
  return(rle(depth))
}

#' Per-base coverage of an interval set
#'
#' Computes run-length encoded coverage for every sequence in `seqLengths`.
#'
#' @export
#' @param x A `SpanSet` object.
#' @param seqLengths Named integer vector of sequence lengths.
#' @param weight Either a single number, a numeric vector with one weight per
#'   interval, or the name of a metadata column holding weights.
#' @return A named list of `rle` objects, one per sequence.
spanCoverage = function(x, seqLengths, weight = 1) {
  n = length(x@start)
  if (is.character(weight)) {
    if (!weight %in% names(x@mcols)) {
      stop(sprintf("no metadata column named '%s'", weight), call. = FALSE)
    }
    weight = x@mcols[[weight]]
  }
  weight = rep_len(as.numeric(weight), n)

  out = vector("list", length(seqLengths))
  names(out) = names(seqLengths)
  for (s in names(seqLengths)) {
    idx = which(x@seqnames == s)
    out[[s]] = .coverageOnSeq(x@start[idx], x@end[idx], as.integer(seqLengths[[s]]), weight[idx])
  }
  return(out)
}

#' Tile sequences into fixed-width bins
#'
#' Splits every sequence into consecutive bins of `width` bases; the last bin
#' on each sequence may be shorter.
#'
#' @export
#' @param seqLengths Named integer vector of sequence lengths.
#' @param width Bin width in bases.
#' @return A `SpanSet` object with a `bin` metadata column.
tileSequences = function(seqLengths, width) {
  width = as.integer(width)
  if (width < 1L) {
    stop("`width` must be a positive integer", call. = FALSE)
  }

  seqCol = character()
  startCol = integer()
  endCol = integer()
  for (s in names(seqLengths)) {
    len = as.integer(seqLengths[[s]])
    # seq.int() would fail on an empty sequence; it simply has no bins.
    if (is.na(len) || len < 1L) {
      next
    }
    starts = seq.int(1L, len, by = width)
    seqCol = c(seqCol, rep(s, length(starts)))
    startCol = c(startCol, starts)
    endCol = c(endCol, pmin(starts + width - 1L, len))
  }

  return(SpanSet(
    seqnames = seqCol,
    start = startCol,
    end = endCol,
    bin = paste0(seqCol, ":", startCol, "-", endCol)
  ))
}

#' Count intervals falling into bins
#'
#' Assigns each interval to the bin containing its midpoint and counts the
#' intervals per bin.
#'
#' @export
#' @param x A `SpanSet` object.
#' @param bins A `SpanSet` of non-overlapping bins, e.g. from
#'   [tileSequences()].
#' @return An integer vector with one count per bin.
binSpanCounts = function(x, bins) {
  mid = (x@start + x@end) %/% 2L
  counts = integer(length(bins@start))

  for (s in unique(bins@seqnames)) {
    binIdx = which(bins@seqnames == s)
    binIdx = binIdx[order(bins@start[binIdx])]
    pos = mid[x@seqnames == s]
    # findInterval() gives the last bin starting at or before each
    # midpoint; it still has to end at or after the midpoint.
    slot = findInterval(pos, bins@start[binIdx])
    inside = slot > 0L
    inside[inside] = pos[inside] <= bins@end[binIdx[slot[inside]]]
    tab = tabulate(slot[inside], nbins = length(binIdx))
    counts[binIdx] = counts[binIdx] + tab
  }

  return(counts)
}

#' Summarise coverage within each interval
#'
#' Expands run-length coverage over each region and reduces it with the
#' chosen statistic.
#'
#' @export
#' @param coverage A named list of `rle` objects from [spanCoverage()].
#' @param regions A `SpanSet` of regions to summarise.
#' @param stat Summary statistic to compute per region.
#' @return A numeric vector with one value per region.
summarizeSpanCoverage = function(coverage,
                                 regions,
                                 stat = c("mean", "max", "sum", "fraction_covered")) {
  stat = match.arg(stat)
  out = rep(NA_real_, length(regions@start))
  # Expand each sequence's rle at most once, however many regions use it.
  expanded = list()
  for (i in seq_along(regions@start)) {
    s = regions@seqnames[i]
    if (is.null(coverage[[s]])) {
      next
    }
    if (is.null(expanded[[s]])) {
      expanded[[s]] = inverse.rle(coverage[[s]])
    }
    depth = expanded[[s]]

    lo = max(regions@start[i], 1L)
    hi = min(regions@end[i], length(depth))
    if (hi < lo) {
      next
    }
    vals = depth[lo:hi]
    out[i] = switch(stat,
      mean = mean(vals),
      max = max(vals),
      sum = sum(vals),
      fraction_covered = mean(vals > 0)
    )
  }

  return(out)
}

#' Per-bin mean coverage
#'
#' Convenience wrapper that tiles sequences and summarises coverage over
#' the resulting bins.
#'
#' @export
#' @param x A `SpanSet` object.
#' @param seqLengths Named integer vector of sequence lengths.
#' @param width Bin width in bases.
#' @return A data frame with columns `bin`, `count` and `meanDepth`.
binnedCoverage = function(x, seqLengths, width) {
  bins = tileSequences(seqLengths, width)
  cov = spanCoverage(x, seqLengths)

  return(data.frame(
    bin = bins@mcols$bin,
    count = binSpanCounts(x, bins),
    meanDepth = summarizeSpanCoverage(cov, bins, "mean"),
    stringsAsFactors = FALSE
  ))
}
