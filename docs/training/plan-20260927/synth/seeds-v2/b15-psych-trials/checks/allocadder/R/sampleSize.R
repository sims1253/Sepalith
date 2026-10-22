near <- function(a, b, tol = 1e-6) isTRUE(all.equal(a, b, tolerance = tol))

# zCritical
stopifnot(near(zCritical(0.05), qnorm(0.975)), near(zCritical(0.05, 1), qnorm(0.95)))
stopifnot(inherits(try(zCritical(0.05, 3), silent = TRUE), "try-error"))

# inflateForDropout
stopifnot(inflateForDropout(100, 0.2) == 125, inflateForDropout(10, 0) == 10)
stopifnot(inherits(try(inflateForDropout(10, 1), silent = TRUE), "try-error"))

# nTwoMeans: the textbook 63 per arm for a 0.5 SD effect
nm <- nTwoMeans(delta = 0.5, sd = 1)
stopifnot(identical(names(nm), c("n1", "n2", "total")), nm[["n1"]] == 63, nm[["total"]] == 126)
stopifnot(nTwoMeans(0.5, 1, ratio = 2)[["n2"]] == 2 * nTwoMeans(0.5, 1, ratio = 2)[["n1"]])

# nTwoProportions
np <- nTwoProportions(0.3, 0.5)
stopifnot(np[["n1"]] > 80, np[["n1"]] < 100)

# nEventsLogrank: hazard ratio 0.7 needs about 247 events
stopifnot(nEventsLogrank(0.7) == 247)

# powerTwoMeans / powerTwoProportions invert the sample size formulas
stopifnot(powerTwoMeans(63, 0.5, 1) >= 0.8, powerTwoMeans(62, 0.5, 1) < 0.8)
stopifnot(powerTwoProportions(np[["n1"]], 0.3, 0.5) >= 0.8)

# sampleSizeGrid
g <- sampleSizeGrid(c(0.5, 1), sd = 1, dropout = 0.1)
stopifnot(nrow(g) == 4, g$n1[1] == 16, g$randomised[1] == inflateForDropout(32, 0.1))
