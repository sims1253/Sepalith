query = SpanSet(
  seqnames = c("chr1", "chr1", "chr1", "chr2"),
  start = c(100, 300, 900, 50),
  end = c(199, 399, 950, 80),
  strand = c("+", "-", "*", "+")
)

subject = SpanSet(
  seqnames = c("chr1", "chr1", "chr2", "chr3"),
  start = c(150, 380, 70, 1),
  end = c(250, 500, 120, 1000),
  strand = c("+", "+", "*", "+")
)

test_that("findSpanOverlaps respects strand by default", {
  hits = findSpanOverlaps(query, subject)
  expect_equal(hits$queryHits, c(1L, 4L))
  expect_equal(hits$subjectHits, c(1L, 3L))
})

test_that("findSpanOverlaps can ignore strand", {
  hits = findSpanOverlaps(query, subject, ignoreStrand = TRUE)
  expect_equal(hits$queryHits, c(1L, 2L, 4L))
  expect_equal(hits$subjectHits, c(1L, 2L, 3L))
})

test_that("findSpanOverlaps honours maxgap", {
  near = SpanSet("chr1", start = 205, end = 210, strand = "+")
  expect_equal(nrow(findSpanOverlaps(near, subject)), 1L)
  expect_equal(nrow(findSpanOverlaps(spanSubset(query, 1L), near)), 0L)
  hits = findSpanOverlaps(spanSubset(query, 1L), near, maxgap = 10L)
  expect_equal(nrow(hits), 1L)
})

test_that("type = 'within' requires containment", {
  inner = SpanSet("chr1", start = 160, end = 170, strand = "+")
  expect_equal(nrow(findSpanOverlaps(inner, subject, type = "within")), 1L)
  expect_equal(nrow(findSpanOverlaps(subject, inner, type = "within")), 0L)
})

test_that("countSpanOverlaps returns one count per query", {
  counts = countSpanOverlaps(query, subject, ignoreStrand = TRUE)
  expect_type(counts, "integer")
  expect_equal(counts, c(1L, 1L, 0L, 1L))
})

test_that("overlapsAnySpan and subsetByOverlapSpans agree", {
  keep = overlapsAnySpan(query, subject)
  expect_equal(keep, c(TRUE, FALSE, FALSE, TRUE))
  kept = subsetByOverlapSpans(query, subject)
  dropped = subsetByOverlapSpans(query, subject, invert = TRUE)
  expect_equal(spanStart(kept), c(100L, 50L))
  expect_equal(spanStart(dropped), c(300L, 900L))
})

test_that("nearestSpan reports gaps and NA for unmatched sequences", {
  lone = SpanSet(c("chr1", "chrX"), start = c(600, 10), end = c(650, 20))
  res = nearestSpan(lone, subject)
  expect_equal(res$subjectHits, c(2L, NA_integer_))
  expect_equal(res$distance, c(99L, NA_integer_))
})

test_that("empty inputs give empty results", {
  none = SpanSet(character(), start = integer(), end = integer())
  hits = findSpanOverlaps(none, subject)
  expect_equal(nrow(hits), 0L)
  expect_named(hits, c("queryHits", "subjectHits"))
  expect_equal(countSpanOverlaps(none, subject), integer())
})
