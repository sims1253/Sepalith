s <- structure(list(start = c(5, 0, 2), end = c(6, 3, 4), label = c("a", "b", "c")),
  class = "spanwise_span")
g <- structure(list(at = c(1, 4, 8), width = c(2, 0, 1)), class = "spanwise_gap")
empty_s <- structure(list(start = double(), end = double(), label = character()),
  class = "spanwise_span")
empty_g <- structure(list(at = double(), width = double()), class = "spanwise_gap")

ss <- summary(s)
stopifnot(inherits(ss, "summary_spanwise_span"), ss$n == 3L, ss$total_width == 6)
stopifnot(ss$min_width == 1, ss$max_width == 3, identical(ss$extent, c(0, 6)))
stopifnot(ss$n_overlaps == 1L, summary(empty_s)$n_overlaps == 0L)
stopifnot(is.na(summary(empty_s)$min_width))

sg <- summary(g)
stopifnot(inherits(sg, "summary_spanwise_gap"), sg$n == 3L, sg$total_width == 3)
stopifnot(sg$min_width == 0, sg$max_width == 2, identical(sg$extent, c(1, 9)))
stopifnot(sg$n_empty == 1L, summary(empty_g)$n_empty == 0L)

out <- capture.output(res <- print(ss))
stopifnot(identical(res, ss), length(out) == 7L, grepl("^Spans +3$", out[1]))
stopifnot(grepl("^Overlaps +1$", out[7]), grepl("^Extent to", out[6]))
stopifnot(identical(capture.output(print(summary(empty_s))), "Spans       0"))

out <- capture.output(res <- print(sg))
stopifnot(identical(res, sg), length(out) == 7L, grepl("^Empty +1$", out[7]))
stopifnot(identical(capture.output(print(summary(empty_g))), "Gaps        0"))

df <- as.data.frame(s)
stopifnot(identical(names(df), c("label", "start", "end", "width")))
stopifnot(identical(df$width, c(1, 3, 2)))
stopifnot(identical(rownames(as.data.frame(s, row.names = c("x", "y", "z"))), c("x", "y", "z")))

dg <- as.data.frame(g)
stopifnot(identical(names(dg), c("at", "width", "end")), identical(dg$end, c(3, 4, 9)))
stopifnot(identical(rownames(as.data.frame(g, row.names = letters[1:3])), letters[1:3]))
