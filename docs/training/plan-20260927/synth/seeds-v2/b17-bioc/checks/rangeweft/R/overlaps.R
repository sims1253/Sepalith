# skipped: subsetByOverlapSpans() needs spanSubset() from spans.R
# The class lives in spans.R; a minimal stand-in constructor is enough here.
setClass("SpanSet", representation(
  seqnames = "character", start = "integer", end = "integer",
  strand = "character", mcols = "data.frame"
))
mk = function(seqnames, start, end, strand = "*") {
  n = length(start)
  new("SpanSet", seqnames = rep_len(seqnames, n), start = as.integer(start),
      end = as.integer(end), strand = rep_len(strand, n),
      mcols = data.frame(row.names = seq_len(n)))
}
query = mk(c("chr1", "chr1", "chr1", "chr2"), c(100, 300, 900, 50), c(199, 399, 950, 80),
           c("+", "-", "*", "+"))
subject = mk(c("chr1", "chr1", "chr2", "chr3"), c(150, 380, 70, 1), c(250, 500, 120, 1000),
             c("+", "+", "*", "+"))

stopifnot(
  identical(.strandCompatible(c("+", "*", "-"), c("-", "-", "-")), c(FALSE, TRUE, TRUE)),
  .pairOverlaps(1L, 10L, 10L, 20L, "any", -1L),
  !.pairOverlaps(1L, 10L, 11L, 20L, "any", -1L),
  .pairOverlaps(5L, 6L, 1L, 10L, "within", 0L)
)

hits = findSpanOverlaps(query, subject)
stopifnot(
  identical(hits$queryHits, c(1L, 4L)),
  identical(hits$subjectHits, c(1L, 3L)),
  identical(nrow(findSpanOverlaps(query, subject, ignoreStrand = TRUE)), 3L),
  identical(nrow(findSpanOverlaps(mk("chr1", 1, 2), subject)), 0L)
)

stopifnot(
  identical(countSpanOverlaps(query, subject, ignoreStrand = TRUE), c(1L, 1L, 0L, 1L)),
  identical(overlapsAnySpan(query, subject), c(TRUE, FALSE, FALSE, TRUE))
)

near = nearestSpan(mk(c("chr1", "chrX"), c(600, 10), c(650, 20)), subject)
stopifnot(
  identical(near$subjectHits, c(2L, NA)),
  identical(near$distance, c(99L, NA))
)
