# The class and constructor live in spans.R; minimal stand-ins here.
setClass("SpanSet", representation(
  seqnames = "character", start = "integer", end = "integer",
  strand = "character", mcols = "data.frame"
))
SpanSet = function(seqnames, start, end, strand = "*", ...) {
  n = length(start)
  extra = list(...)
  mcols = if (length(extra)) data.frame(extra) else data.frame(row.names = seq_len(n))
  new("SpanSet", seqnames = rep_len(as.character(seqnames), n), start = as.integer(start),
      end = as.integer(end), strand = rep_len(as.character(strand), n), mcols = mcols)
}

cov1 = .coverageOnSeq(c(1L, 3L, 9L), c(4L, 5L, 12L), 10L, c(1, 1, 2))
stopifnot(identical(inverse.rle(cov1), c(1, 1, 2, 2, 1, 0, 0, 0, 2, 2)))

x = SpanSet(c("chr1", "chr1", "chr2"), c(1, 3, 2), c(4, 5, 3), score = c(1, 2, 5))
cv = spanCoverage(x, c(chr1 = 6L, chr2 = 4L))
cw = spanCoverage(x, c(chr1 = 6L), weight = "score")
stopifnot(
  identical(names(cv), c("chr1", "chr2")),
  identical(inverse.rle(cv$chr2), c(0, 1, 1, 0)),
  identical(inverse.rle(cw$chr1), c(1, 1, 3, 3, 2, 0)),
  inherits(try(spanCoverage(x, c(chr1 = 6L), weight = "nope"), silent = TRUE), "try-error")
)

bins = tileSequences(c(chr1 = 10L, chr2 = 4L, chrE = 0L), 4L)
stopifnot(
  identical(bins@start, c(1L, 5L, 9L, 1L)),
  identical(bins@end, c(4L, 8L, 10L, 4L)),
  identical(bins@mcols$bin[3], "chr1:9-10"),
  inherits(try(tileSequences(c(chr1 = 10L), 0), silent = TRUE), "try-error")
)

stopifnot(
  identical(binSpanCounts(x, bins), c(2L, 0L, 0L, 1L)),
  identical(summarizeSpanCoverage(cv, bins, "max"), c(2, 1, NA, 1)),
  identical(summarizeSpanCoverage(cv, bins, "fraction_covered")[2], 0.5)
)

bc = binnedCoverage(x, c(chr1 = 6L), 3L)
stopifnot(
  identical(bc$count, c(1L, 1L)),
  isTRUE(all.equal(bc$meanDepth, c(4 / 3, 1)))
)
