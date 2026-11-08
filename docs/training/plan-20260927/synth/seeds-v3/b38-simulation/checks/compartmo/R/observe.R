tr = data.frame(time = 0:3, S = c(100, 90, 75, 70), E = c(0, 6, 10, 8),
  I = c(1, 4, 9, 12), R = c(0, 1, 7, 11))
stopifnot(identical(flowFromTrajectory(tr), c(0, 10, 15, 5)))
stopifnot(identical(flowFromTrajectory(tr, "onset"), c(0, 4, 11, 7)))
stopifnot(identical(flowFromTrajectory(tr, "recovery"), c(0, 1, 6, 4)))
stopifnot(inherits(try(flowFromTrajectory(tr[1, ]), silent = TRUE), "try-error"))
stopifnot(conditionMessage(tryCatch(flowFromTrajectory(tr[, c("time", "S", "R")]),
  error = identity)) == "Trajectory is missing column `E`.")

set.seed(7)
inc = c(0, 10, 50, 100, 80, 20)
stopifnot(identical(observeCases(inc, reportingRate = 0.5), inc / 2))
po = observeCases(inc, "poisson", reportingRate = 0.5)
stopifnot(length(po) == 6L, po[1] == 0, is.integer(po))
nb = observeCases(inc, "negbin", size = 5)
stopifnot(length(nb) == 6L, nb[1] == 0)
bi = observeCases(inc, "binomial", reportingRate = 0.3)
stopifnot(all(bi <= inc), bi[1] == 0)
stopifnot(inherits(try(observeCases(inc, reportingRate = 2), silent = TRUE), "try-error"))

cs = c(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)
stopifnot(identical(delayCases(cs, "none"), cs))
stopifnot(identical(delayCases(cs, "fixed", meanDelay = 2L), c(0, 0, 10, 0, 0, 0, 0, 0, 0, 0)))
stopifnot(isTRUE(all.equal(delayCases(cs, "uniform", 1L), c(10, 10, 10, 0, 0, 0, 0, 0, 0, 0) / 3)))
g = delayCases(cs, "geometric", 1L)
stopifnot(isTRUE(all.equal(sum(g), 10)), g[1] > g[2], g[2] > g[3])

ag = aggregateCounts(1:15)
stopifnot(identical(ag$start, c(0L, 7L)), identical(ag$count, c(28, 77)))
stopifnot(nrow(aggregateCounts(1:15, partial = TRUE)) == 3L)
stopifnot(identical(aggregateCounts(1:3, "day")$count, c(1, 2, 3)))
stopifnot(aggregateCounts(1:28, "fortnight")$count[2] == sum(15:28))

cm = compareObserved(c(1, 3, 5), c(2, 2, 2))
stopifnot(identical(names(cm), c("rmse", "mae", "bias", "deviance")))
stopifnot(isTRUE(all.equal(cm[["rmse"]], sqrt(11 / 3))), cm[["mae"]] == 5 / 3, cm[["bias"]] == 1)
stopifnot(compareObserved(c(0, 2), c(1, 2), "deviance")[["deviance"]] == 2)
stopifnot(inherits(try(compareObserved(1, 1:2), silent = TRUE), "try-error"))

stopifnot(isTRUE(all.equal(observationLogLik(c(1, 2), c(1, 2)), dpois(1, 1,
  TRUE) + dpois(2, 2, TRUE))))
stopifnot(isTRUE(all.equal(observationLogLik(3, 2, "negbin", size = 4), dnbinom(3, 4,
  mu = 2, log = TRUE))))
stopifnot(isTRUE(all.equal(observationLogLik(3, 5, "binomial", trials = 10), dbinom(3,
  10, 0.5, TRUE))))
stopifnot(inherits(try(observationLogLik(3, 5, "binomial"), silent = TRUE), "try-error"))
