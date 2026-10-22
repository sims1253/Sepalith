S <- matrix(c(4, 2, 2, 3), 2)

stopifnot(identical(check.square(S), S))
stopifnot(inherits(try(check.square(matrix(1, 2, 3)), silent = TRUE), "try-error"))
stopifnot(inherits(try(check.square(data.frame(a = 1)), silent = TRUE), "try-error"))

stopifnot(identical(check.symmetric(S), S))
stopifnot(inherits(try(check.symmetric(matrix(1:4 + 0, 2)), silent = TRUE), "try-error"))

stopifnot(all.equal(check.pos.def(S), chol(S)))
stopifnot(inherits(try(check.pos.def(matrix(c(1, 2, 2, 1), 2)), silent = TRUE), "try-error"))

stopifnot(all.equal(cholesky.solve(S, c(1, 2)), solve(S, c(1, 2))))

set.seed(2)
X <- cbind(1, x1 = stats::rnorm(30), x2 = stats::runif(30))
y <- drop(X %*% c(1, 2, -1)) + stats::rnorm(30, sd = 0.1)
ref <- stats::lm.fit(X, y)
q <- ols.qr(X, y)
stopifnot(all.equal(unname(q$coef), unname(ref$coefficients)))
stopifnot(all.equal(q$sigma, sqrt(sum(ref$residuals^2) / 27)))
stopifnot(inherits(try(ols.qr(cbind(X, X[, 2]), y), silent = TRUE), "try-error"))
ne <- ols.normal(X, y)
stopifnot(all.equal(unname(ne$coef), unname(q$coef)), all.equal(ne$resid, unname(q$resid)))

Xr <- X[, -1]
r0 <- ols.ridge(Xr, y, lambda = 0)
stopifnot(all.equal(unname(r0$coef), unname(q$coef)))
r10 <- ols.ridge(Xr, y, lambda = 10)
stopifnot(sum(r10$coef[-1]^2) < sum(r0$coef[-1]^2))
stopifnot(inherits(try(ols.ridge(X, y), silent = TRUE), "try-error"))

pi1 <- power.iteration(S)
stopifnot(all.equal(pi1$value, max(eigen(S)$values)))
stopifnot(all.equal(abs(pi1$vector), abs(eigen(S)$vectors[, 1]), tolerance = 1e-6))

m2 <- mahal.sq(X[, -1])
Xs <- X[, -1]
stopifnot(all.equal(m2, unname(stats::mahalanobis(Xs, colMeans(Xs), stats::cov(Xs)))))
stopifnot(all.equal(mahal.sq(diag(2), center = c(0, 0), cov = diag(2)), c(1, 1)))
