m <- initChainMatrix(3, c("a", "b"))
stopifnot(dim(m) == c(3L, 2L), all(is.na(m)), identical(colnames(m), c("a", "b")))

set.seed(71)
stopifnot(abs(mean(replicate(2000, drawMuNormal(rep(1, 100), 1, 0, 1e6))) - 1) < 0.01)
stopifnot(all(replicate(50, drawSigma2Normal(rnorm(20), 0, 1, 1)) > 0))

X <- cbind(1, runif(50))
y <- drop(X %*% c(2, -1)) + rnorm(50, sd = 0.3)
bd <- replicate(500, drawBetaRegression(X, y, 0.09, 1e-6))
stopifnot(dim(bd) == c(2L, 500L), all(abs(rowMeans(bd) - qr.solve(X, y)) < 0.05))
stopifnot(drawSigma2Regression(X, y, c(2, -1), 1, 1) > 0)

gn <- gibbsNormal(c(rnorm(100, 5, 2), NA), nIter = 1000L)
stopifnot(identical(colnames(gn), c("mu", "sigma2")), abs(mean(gn[, "mu"]) - 5) < 0.6)
stopifnot(abs(mean(gn[, "sigma2"]) - 4) < 1.5)
stopifnot(inherits(try(gibbsNormal(c(1, NA)), silent = TRUE), "try-error"))

gr <- gibbsRegression(X, y, nIter = 800L)
stopifnot(identical(colnames(gr), c("beta1", "beta2", "sigma2")))
stopifnot(all(abs(colMeans(gr[, 1:2]) - c(2, -1)) < 0.3))
stopifnot(inherits(try(gibbsRegression(X, y[-1]), silent = TRUE), "try-error"))

grp <- factor(rep(c("a", "b", "c"), each = 20), levels = c("a", "b", "c", "empty"))
yh <- rnorm(60, c(0, 2, 4)[as.integer(grp)])
yh[5] <- NA
gh <- gibbsHierarchical(yh, grp, nIter = 1000L)
stopifnot(identical(colnames(gh), c("theta[a]", "theta[b]", "theta[c]", "mu", "tau2", "sigma2")))
groupMeans <- tapply(yh, grp, mean, na.rm = TRUE)[1:3]
stopifnot(!anyNA(gh), all(abs(colMeans(gh[, 1:3]) - groupMeans) < 0.15))
stopifnot(inherits(try(gibbsHierarchical(1:3, 1:2), silent = TRUE), "try-error"))
