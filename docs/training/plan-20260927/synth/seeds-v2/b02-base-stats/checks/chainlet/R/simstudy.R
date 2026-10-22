est <- c(1, 2, 3, NA, Inf)

b <- simBias(est, truth = 1.5)
stopifnot(identical(names(b), c("value", "mcse")), b[["value"]] == 0.5)
stopifnot(all.equal(b[["mcse"]], 1 / sqrt(3)))

se <- simEmpSe(est, truth = 0)
stopifnot(se[["value"]] == 1, all.equal(se[["mcse"]], 1 / 2))

r <- simRmse(c(1, 3), truth = 2)
stopifnot(r[["value"]] == 1, r[["mcse"]] == 0)

cv <- simCoverage(c(0, 0, 2, NA), c(1, 3, 4, 1), truth = 0.5)
stopifnot(all.equal(cv[["value"]], 2 / 3), all.equal(cv[["mcse"]], sqrt(2 / 9 / 3)))

set.seed(41)
gen <- function(n, mu) rnorm(n, mu)
estFn <- function(d) c(mean = mean(d), med = median(d))
res <- runSimulation(gen, estFn, nRep = 20L, params = list(n = 10, mu = 2), seed = 1)
stopifnot(dim(res) == c(20L, 2L), identical(colnames(res), c("mean", "med")), !anyNA(res))
stopifnot(identical(res, runSimulation(gen, estFn, nRep = 20L, list(n = 10, mu = 2), seed = 1)))
flaky <- local({
  k <- 0
  function(d) {
    k <<- k + 1
    if (k == 3) stop("boom")
    c(m = mean(d))
  }
})
rf <- runSimulation(gen, flaky, nRep = 5L, params = list(n = 5, mu = 0))
stopifnot(sum(is.na(rf)) == 1L, is.na(rf[3, 1]))
stopifnot(inherits(try(runSimulation(gen, function(d) stop("x"), 3L, list(n = 2, mu = 0)),
  silent = TRUE), "try-error"))

g <- simulationGrid(n = c(10, 50), mu = c(0, 1))
stopifnot(nrow(g) == 4L, identical(names(g), c("scenario", "n", "mu")), g$scenario == 1:4)
stopifnot(inherits(try(simulationGrid(n = numeric(0)), silent = TRUE), "try-error"))

rg <- runSimulationGrid(g, gen, estFn, nRep = 30L, seed = 100)
stopifnot(length(rg) == 4L, identical(names(rg), paste0("scenario", 1:4)))
stopifnot(identical(rg[[2]], runSimulation(gen, estFn, 30L, list(n = 50, mu = 0), seed = 102)))

ss <- summarizeSimulation(rg, "mean", truth = g$mu)
stopifnot(identical(names(ss), c("scenario", "bias", "empSe", "rmse", "failed")))
stopifnot(all(abs(ss$bias) < 0.5), all(ss$failed == 0), all(ss$rmse >= ss$empSe * 0.9))
