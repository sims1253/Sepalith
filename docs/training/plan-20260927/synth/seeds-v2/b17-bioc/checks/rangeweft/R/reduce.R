# skipped: intersectSpans() and setdiffSpans() need findSpanOverlaps()
# from overlaps.R
# The class and constructor live in spans.R; minimal stand-ins here.
setClass("SpanSet", representation(
  seqnames = "character", start = "integer", end = "integer",
  strand = "character", mcols = "data.frame"
))
SpanSet = function(seqnames, start, end, strand = "*") {
  n = length(start)
  new("SpanSet", seqnames = rep_len(as.character(seqnames), n), start = as.integer(start),
      end = as.integer(end), strand = rep_len(as.character(strand), n),
      mcols = data.frame(row.names = seq_len(n)))
}

m = .mergeIntervals(c(10L, 1L, 30L, 21L), c(20L, 5L, 40L, 25L))
stopifnot(
  identical(unname(m[, "start"]), c(1L, 10L, 30L)),
  identical(unname(m[, "end"]), c(5L, 25L, 40L)),
  identical(nrow(.mergeIntervals(c(1L, 10L), c(5L, 12L), minGap = 4L)), 1L),
  identical(nrow(.mergeIntervals(integer(), integer())), 0L)
)

x = SpanSet(c("chr1", "chr1", "chr1", "chr2"), c(1, 4, 20, 5), c(5, 10, 30, 9),
            c("+", "+", "-", "+"))
stopifnot(
  identical(.groupKey(x, FALSE), c("chr1|+", "chr1|+", "chr1|-", "chr2|+")),
  identical(.groupKey(x, TRUE)[4], "chr2|*")
)

r = reduceSpans(x)
stopifnot(
  identical(r@start, c(1L, 20L, 5L)),
  identical(r@end, c(10L, 30L, 9L)),
  identical(r@strand, c("+", "-", "+")),
  identical(length(reduceSpans(x, ignoreStrand = TRUE)@start), 3L)
)

g = gapSpans(x, seqLengths = c(chr1 = 40L), ignoreStrand = TRUE)
stopifnot(
  identical(g@start, c(11L, 31L)),
  identical(g@end, c(19L, 40L))
)

u = unionSpans(SpanSet("chr1", 1, 10), SpanSet("chr1", 8, 15))
stopifnot(identical(c(u@start, u@end), c(1L, 15L)))
