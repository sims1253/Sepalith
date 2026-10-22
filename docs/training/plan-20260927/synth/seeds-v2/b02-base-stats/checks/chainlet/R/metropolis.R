stopifnot(all.equal(logPriorNormal(c(0, 1), sd = 1), sum(dnorm(c(0, 1), log = TRUE))))
stopifnot(logPriorGamma(c(1, -1)) == -Inf, all.equal(logPriorGamma(2, rate = 1), -2))
stopifnot(logPriorBeta(1) == -Inf, logPriorBeta(0.3) == 0)

lp <- makeLogPosterior(function(th) -sum(th^2), function(th) logPriorGamma(th))
stopifnot(lp(-1) == -Inf, all.equal(lp(1), -1 - 1))

set.seed(51)
target <- function(th) sum(dnorm(th, c(1, -1), 1, log = TRUE))

rw <- rwMetropolis(target, c(a = 0, b = 0), nIter = 4000L, proposalSd = 1)
stopifnot(dim(rw) == c(4000L, 2L), identical(colnames(rw), c("a", "b")))
stopifnot(all(abs(colMeans(rw[-(1:500), ]) - c(1, -1)) < 0.25))
stopifnot(inherits(try(rwMetropolis(function(th) -Inf, 0), silent = TRUE), "try-error"))

ind <- indepMetropolis(target, c(0, 0), nIter = 3000L, proposalMean = c(1, -1), proposalSd = 1.5)
stopifnot(all(abs(colMeans(ind) - c(1, -1)) < 0.2), acceptanceRate(ind) > 0.3)

ar <- acceptanceRate(rw)
stopifnot(ar > 0, ar < 1, all.equal(ar, mean(attr(rw, "accepted"))))
stopifnot(inherits(try(acceptanceRate(matrix(1)), silent = TRUE), "try-error"))

sdT <- tuneProposalSd(target, c(0, 0), proposalSd = 0.05, pilotIter = 300L)
stopifnot(sdT > 0.05)

am <- adaptiveMetropolis(target, c(0, 0), nIter = 3000L, adaptStart = 300L)
stopifnot(all(abs(colMeans(am[-(1:1000), ]) - c(1, -1)) < 0.3))
stopifnot(length(attr(am, "accepted")) == 3000L)
