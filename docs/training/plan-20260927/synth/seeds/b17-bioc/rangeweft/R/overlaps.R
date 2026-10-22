#' Are two strands compatible?
#'
#' Unstranded (`"*"`) intervals are compatible with either strand.
#'
#' @param a,b Character vectors of strands, recycled to a common length.
#' @return A logical vector.
#' @noRd
.strandCompatible = function(a, b) {
  return(a == "*" | b == "*" | a == b)
}

#' Test a single pair of intervals for overlap
#'
#' Applies the overlap rule named by `type` to one query and one subject
#' interval given as start/end coordinates.
#'
#' @param qs,qe Start and end of the query interval.
#' @param ss,se Start and end of the subject interval.
#' @param type Overlap rule.
#' @param maxgap Largest gap between intervals still counted as overlap.
#' @return A single logical value.
#' @noRd
.pairOverlaps = function(qs, qe, ss, se, type, maxgap) {
  hit = switch(type,
    any = qs <= se + maxgap + 1L && ss <= qe + maxgap + 1L,
    within = qs >= ss && qe <= se,
    start = abs(qs - ss) <= maxgap,
    end = abs(qe - se) <= maxgap,
    equal = abs(qs - ss) <= maxgap && abs(qe - se) <= maxgap
  )
  return(hit)
}

#' Find overlaps between two interval sets on one sequence
#'
#' Sweeps over subject intervals sorted by start to collect hits for every
#' query interval on a single sequence.
#'
#' @param qIdx,sIdx Integer indices of query and subject intervals on the
#'   sequence.
#' @param query,subject `SpanSet` objects.
#' @param type Overlap rule.
#' @param maxgap Largest gap between intervals still counted as overlap.
#' @param ignoreStrand Should strand be ignored?
#' @return A two-column integer matrix of query and subject indices.
#' @noRd
.overlapsOnSeq = function(qIdx, sIdx, query, subject, type, maxgap, ignoreStrand) {
  sIdx = sIdx[order(subject@start[sIdx])]
  sStart = subject@start[sIdx]
  hits = vector("list", length(qIdx))
  for (k in seq_along(qIdx)) {
    q = qIdx[k]
    qs = query@start[q]
    qe = query@end[q]
    # subject intervals starting after the query end cannot overlap
    upper = findInterval(qe + maxgap + 1L, sStart)
    found = integer()
    for (m in seq_len(upper)) {
      s = sIdx[m]
      if (!ignoreStrand && !.strandCompatible(query@strand[q], subject@strand[s])) {
        next
      }
      if (.pairOverlaps(qs, qe, subject@start[s], subject@end[s], type, maxgap)) {
        found = c(found, s)
      }
    }
    if (length(found) > 0L) {
      hits[[k]] = cbind(queryHits = q, subjectHits = sort(found))
    }
  }
  out = do.call(rbind, hits)
  if (is.null(out)) {
    out = matrix(integer(), ncol = 2L, dimnames = list(NULL, c("queryHits", "subjectHits")))
  }
  return(out)
}

#' Find overlapping intervals
#'
#' Reports every pair of query and subject intervals that overlap under the
#' chosen rule, sequence by sequence.
#'
#' @param query,subject `SpanSet` objects.
#' @param type Overlap rule: any overlap, query within subject, matching
#'   starts, matching ends, or identical coordinates.
#' @param maxgap Largest gap between intervals still counted as overlap.
#' @param ignoreStrand Should strand be ignored?
#' @return A data frame with integer columns `queryHits` and `subjectHits`,
#'   sorted by query then subject.
#' @export
findSpanOverlaps = function(query,
                            subject,
                            type = c("any", "within", "start", "end", "equal"),
                            maxgap = -1L,
                            ignoreStrand = FALSE) {
  type = match.arg(type)
  if (type != "any" && maxgap < 0L) {
    maxgap = 0L
  }
  seqs = intersect(unique(query@seqnames), unique(subject@seqnames))
  pieces = vector("list", length(seqs))
  for (i in seq_along(seqs)) {
    qIdx = which(query@seqnames == seqs[i])
    sIdx = which(subject@seqnames == seqs[i])
    pieces[[i]] = .overlapsOnSeq(qIdx, sIdx, query, subject, type, maxgap, ignoreStrand)
  }
  hits = do.call(rbind, pieces)
  if (is.null(hits) || nrow(hits) == 0L) {
    return(data.frame(queryHits = integer(), subjectHits = integer()))
  }
  hits = hits[order(hits[, "queryHits"], hits[, "subjectHits"]), , drop = FALSE]
  return(as.data.frame(hits))
}

#' Count overlaps per query interval
#'
#' @inheritParams findSpanOverlaps
#' @param ... Passed to [findSpanOverlaps()].
#' @return An integer vector with one count per query interval.
#' @export
countSpanOverlaps = function(query, subject, ...) {
  hits = findSpanOverlaps(query, subject, ...)
  counts = tabulate(hits$queryHits, nbins = length(query@start))
  return(counts)
}

#' Does each query interval overlap any subject interval?
#'
#' @inheritParams findSpanOverlaps
#' @param ... Passed to [findSpanOverlaps()].
#' @return A logical vector with one element per query interval.
#' @export
overlapsAnySpan = function(query, subject, ...) {
  hits = findSpanOverlaps(query, subject, ...)
  out = seq_along(query@start) %in% hits$queryHits
  return(out)
}

#' Subset query intervals by overlap
#'
#' Keeps (or, with `invert = TRUE`, drops) query intervals overlapping any
#' subject interval.
#'
#' @inheritParams findSpanOverlaps
#' @param invert Keep the non-overlapping intervals instead?
#' @param ... Passed to [findSpanOverlaps()].
#' @return A `SpanSet` object.
#' @export
subsetByOverlapSpans = function(query, subject, invert = FALSE, ...) {
  keep = overlapsAnySpan(query, subject, ...)
  if (invert) {
    keep = !keep
  }
  return(spanSubset(query, keep))
}

#' Distance to the nearest subject interval
#'
#' For each query interval, finds the closest subject interval on the same
#' sequence and reports the gap in bases (0 when overlapping).
#'
#' @inheritParams findSpanOverlaps
#' @return A data frame with columns `queryHits`, `subjectHits` and
#'   `distance`; unmatched queries get `NA`.
#' @export
nearestSpan = function(query, subject, ignoreStrand = FALSE) {
  n = length(query@start)
  nearestIdx = rep(NA_integer_, n)
  nearestDist = rep(NA_integer_, n)
  for (q in seq_len(n)) {
    cand = which(subject@seqnames == query@seqnames[q])
    if (!ignoreStrand) {
      cand = cand[.strandCompatible(query@strand[q], subject@strand[cand])]
    }
    if (length(cand) == 0L) {
      next
    }
    gapLeft = query@start[q] - subject@end[cand] - 1L
    gapRight = subject@start[cand] - query@end[q] - 1L
    dist = pmax(gapLeft, gapRight, 0L)
    best = which.min(dist)
    nearestIdx[q] = cand[best]
    nearestDist[q] = as.integer(dist[best])
  }
  return(data.frame(
    queryHits = seq_len(n),
    subjectHits = nearestIdx,
    distance = nearestDist
  ))
}
