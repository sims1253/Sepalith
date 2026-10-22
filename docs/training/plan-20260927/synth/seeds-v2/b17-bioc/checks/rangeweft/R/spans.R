x = SpanSet("chr1", start = c(100, 300), width = c(50, 10), strand = c("+", "-"),
            gene = c("a", "b"))
stopifnot(
  identical(spanStart(x), c(100L, 300L)),
  identical(spanEnd(x), c(149L, 309L)),
  identical(spanWidth(x), c(50L, 10L)),
  identical(x@mcols$gene, c("a", "b"))
)

# scalar metadata is recycled; missing end and width is an error
y = SpanSet(c("chr1", "chr2"), start = c(1, 5), end = c(10, 20), source = "ens")
stopifnot(
  identical(y@mcols$source, c("ens", "ens")),
  identical(y@strand, c("*", "*")),
  inherits(try(SpanSet("chr1", start = 1), silent = TRUE), "try-error"),
  inherits(try(SpanSet("chr1", start = 1, end = 2, strand = "x"), silent = TRUE), "try-error")
)

sub = spanSubset(x, c(FALSE, TRUE))
stopifnot(identical(spanStart(sub), 300L), identical(sub@mcols$gene, "b"))

r = resizeSpans(x, 20, fix = "start")
stopifnot(
  identical(spanStart(r), c(100L, 290L)),
  identical(spanEnd(r), c(119L, 309L)),
  identical(spanWidth(resizeSpans(x, 11, fix = "center")), c(11L, 11L))
)

up = flankSpans(x, 5)
down = flankSpans(x, 5, side = "downstream", ignoreStrand = TRUE)
edge = flankSpans(SpanSet("chr1", start = 3, end = 8, strand = "+"), 10)
stopifnot(
  identical(spanStart(up), c(95L, 310L)),
  identical(spanEnd(up), c(99L, 314L)),
  identical(spanStart(down), c(150L, 310L)),
  identical(c(spanStart(edge), spanEnd(edge)), c(1L, 2L))
)
