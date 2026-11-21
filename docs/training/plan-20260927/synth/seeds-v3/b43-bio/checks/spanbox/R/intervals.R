x <- make.intervals(c("chr1", "chr1"), c(100, 500), c(199, 549), strand = c("+", "-"))

# make.intervals: every validated argument
stopifnot(inherits(x, "spanbox_intervals"), identical(x$name, c("int1", "int2")))
stopifnot(is.integer(x$start), identical(x$strand, c("+", "-")))
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
stopifnot(fails(make.intervals(1, 1, 2)), fails(make.intervals("c", 0, 2)))
stopifnot(fails(make.intervals("c", 1, NA)), fails(make.intervals("c", 5, 2)))
stopifnot(
  fails(make.intervals("c", 1, 2, strand = "x")),
  fails(make.intervals("c", 1, 2, name = 1))
)

# resize.intervals: every anchor, both strands
r <- resize.intervals(x, 10, "start")
stopifnot(r$start == c(100L, 540L), r$end == c(109L, 549L))
r <- resize.intervals(x, 10, "end")
stopifnot(r$start == c(190L, 500L), r$end == c(199L, 509L))
r <- resize.intervals(x, 10, "center")
stopifnot(r$start == c(145L, 520L), r$end == c(154L, 529L))
stopifnot(fails(resize.intervals(x, 0)))

# flank.intervals: every side, both strands
up <- flank.intervals(x, 20, "upstream")
stopifnot(up$start == c(80L, 550L), up$end == c(99L, 569L))
down <- flank.intervals(x, 20, "downstream")
stopifnot(down$start == c(200L, 480L), down$end == c(219L, 499L))
both <- flank.intervals(x, 20, "both")
stopifnot(nrow(both) == 4L, both$start == c(80L, 200L, 550L, 480L))
stopifnot(nrow(flank.intervals(make.intervals("c", 1, 5), 3)) == 0L)

# clip.intervals: both bounds
y <- make.intervals(c("a", "a", "b"), c(1, 90, 300), c(10, 120, 310))
y$start[1] <- -4L
cl <- clip.intervals(y, c(a = 100L, b = 200L))
stopifnot(nrow(cl) == 2L, cl$start[1] == 1L, cl$end[2] == 100L)
stopifnot(attr(cl, "n.clipped.start") == 1L, attr(cl, "n.clipped.end") == 1L)
stopifnot(nrow(clip.intervals(y, c(a = 100L, b = 200L), drop = FALSE)) == 3L)
stopifnot(fails(clip.intervals(y, c(a = 100L))), fails(clip.intervals(y, c(100, 200))))
stopifnot(fails(clip.intervals(y, c(a = 100L, b = 200L), drop = NA)))

# promoter.intervals: both strands
pr <- promoter.intervals(x, upstream = 50, downstream = 10)
stopifnot(pr$start == c(50L, 540L), pr$end == c(109L, 599L))
stopifnot(promoter.intervals(x, 1000, 10)$start[1] == 1L, fails(promoter.intervals(x, -1)))
stopifnot(fails(promoter.intervals(list(), 1, 1)), fails(promoter.intervals(x, 1, c(1, 2))))
stopifnot(identical(pr$name, c("int1.promoter", "int2.promoter")))

# interval.stats: every element
st <- interval.stats(x)
stopifnot(st$n == 2L, st$total.width == 150L, st$mean.width == 75, st$min.width == 50L)
stopifnot(
  st$max.width == 100L,
  st$n.chrom == 1L,
  identical(unname(st$strand.counts), c(1L, 1L, 0L))
)
stopifnot(st$median.width == 75, interval.stats(x[0, ])$n == 0L)
stopifnot(is.na(interval.stats(x[0, ])$max.width), interval.stats(x[0, ])$total.width == 0L)
