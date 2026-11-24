a <- data.frame(chrom = c("c1", "c1", "c2"), start = c(10L, 50L, 5L), end = c(20L, 60L, 8L),
                strand = c("+", "-", "+"))
b <- data.frame(chrom = c("c1", "c1", "c1"), start = c(10L, 15L, 55L), end = c(20L, 30L, 58L),
                strand = c("+", "+", "+"))

# overlap.pairs: every type arm
op <- overlap.pairs(a, b)
stopifnot(identical(op$a.index, c(1L, 1L, 2L)), identical(op$b.index, c(1L, 2L, 3L)))
stopifnot(
  nrow(overlap.pairs(a, b, "within")) == 1L,
  overlap.pairs(a, b, "contains")$b.index == c(1L, 3L)
)
stopifnot(nrow(overlap.pairs(a, b, "start")) == 1L, nrow(overlap.pairs(a, b, "end")) == 1L)
stopifnot(nrow(overlap.pairs(a, b, "equal")) == 1L)
stopifnot(nrow(overlap.pairs(a, b, ignore.strand = FALSE)) == 2L)
stopifnot(nrow(overlap.pairs(a[3, ], b)) == 0L)

# pairwise.intersect: both bounds
pi2 <- pairwise.intersect(a, b)
stopifnot(pi2$start == c(10L, 15L, 55L), pi2$end == c(20L, 20L, 58L))
stopifnot(nrow(pairwise.intersect(a[3, ], b)) == 0L, pi2$width == c(11L, 6L, 4L))
stopifnot(pi2$a.fraction[2] == 6 / 11, pi2$b.fraction[2] == 6 / 16, pi2$b.fraction[3] == 1)

# reduce.intervals
x <- data.frame(chrom = c("c1", "c1", "c1", "c2"), start = c(1L, 5L, 20L, 1L),
                end = c(10L, 12L, 25L, 3L))
rd <- reduce.intervals(x)
stopifnot(nrow(rd) == 3L, rd$end[1] == 12L, rd$n.merged[1] == 2L)
stopifnot(nrow(reduce.intervals(x, gap = 7L)) == 2L, nrow(reduce.intervals(x[0, ])) == 0L)

# interval.distance: both sides
d <- interval.distance(data.frame(chrom = c("c1", "c1", "c9"), start = c(40L, 12L, 1L),
                                  end = c(45L, 13L, 2L)), b)
stopifnot(d$upstream[1] == 10L, d$downstream[1] == 10L, d$nearest[1] == 10L)
stopifnot(d$upstream[2] == 0L, is.na(d$nearest[3]))
d2 <- interval.distance(data.frame(chrom = "c1", start = 1L, end = 4L), b)
stopifnot(is.na(d2$upstream), d2$downstream == 6L, d2$nearest == 6L)

# subtract.intervals: left pieces and the right remainder
s <- subtract.intervals(data.frame(chrom = "c1", start = 1L, end = 70L), b)
stopifnot(s$start == c(1L, 31L, 59L), s$end == c(9L, 54L, 70L), all(s$source == 1L))
stopifnot(nrow(subtract.intervals(b[1, ], b)) == 0L)

# coverage.depth: opens and closes
cv <- coverage.depth(b, "c1", 32L)
stopifnot(cv$start == c(1L, 10L, 15L, 21L, 31L), cv$depth == c(0L, 1L, 2L, 1L, 0L))
stopifnot(sum(cv$end - cv$start + 1L) == 32L, all(coverage.depth(b, "c2", 5L)$depth == 0L))
stopifnot(identical(coverage.depth(b, "c1", 32L, min.depth = 2L)$depth, c(0L, 2L, 0L)))
stopifnot(inherits(try(coverage.depth(b, 1, 5L), silent = TRUE), "try-error"))
stopifnot(inherits(try(coverage.depth(b, "c1", 0), silent = TRUE), "try-error"))
stopifnot(inherits(try(coverage.depth(b, "c1", 5L, 0), silent = TRUE), "try-error"))
