#' Matrix argument checks
#'
#' `check.square()` and `check.symmetric()` return `A` invisibly;
#' `check.pos.def()` returns its upper Cholesky factor so callers do not
#' have to factorise twice.
#'
#' @param A A numeric matrix.
#' @param arg Name of the argument, used in error messages.
#' @keywords internal
check.square = function(A, arg = "A") {
  if (!is.matrix(A) || !is.numeric(A)) {
    stop(sprintf("`%s` must be a numeric matrix.", arg), call. = FALSE)
  }
  if (nrow(A) != ncol(A)) {
    stop(sprintf("`%s` must be square, not %d x %d.", arg, nrow(A), ncol(A)), call. = FALSE)
  }
  invisible(A)
}

#' @param tol Tolerance for the elementwise comparison.
#' @rdname check.square
check.symmetric = function(A, arg = "A", tol = 1e-10) {
  check.square(A, arg)
  if (max(abs(A - t(A))) > tol * max(1, max(abs(A)))) {
    stop(sprintf("`%s` must be symmetric.", arg), call. = FALSE)
  }
  invisible(A)
}

#' @rdname check.square
check.pos.def = function(A, arg = "A") {
  check.symmetric(A, arg)

  # chol() fails exactly when A is not positive definite.
  R = tryCatch(chol(A), error = function(e) NULL)
  if (is.null(R)) {
    stop(sprintf("`%s` must be positive definite.", arg), call. = FALSE)
  }
  invisible(R)
}

#' Solve a positive definite system via Cholesky
#'
#' @param A A symmetric positive definite matrix.
#' @param b Right-hand side vector or matrix.
#' @return The solution of `A %*% x = b`.
#' @export
cholesky.solve = function(A, b) {
  R = check.pos.def(A)

  # A = R'R, so solve R'z = b forwards and then R x = z backwards.
  backsolve(R, forwardsolve(t(R), b))
}

#' Least squares via the QR decomposition
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @return A list with `coef`, `fitted`, `resid` and `sigma`.
#' @export
ols.qr = function(X, y) {
  qx = qr(X)
  if (qx$rank < ncol(X)) {
    stop("`X` is rank deficient.", call. = FALSE)
  }

  beta = qr.coef(qx, y)
  fitted = qr.fitted(qx, y)
  resid = y - fitted
  sigma = sqrt(sum(resid^2) / (nrow(X) - ncol(X)))
  list(coef = beta, fitted = fitted, resid = resid, sigma = sigma)
}

#' Least squares via the normal equations
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @return A list with `coef`, `fitted`, `resid` and `sigma`.
#' @export
ols.normal = function(X, y) {
  # Squares the condition number of X; fine for well-scaled designs only.
  XtX = crossprod(X)
  beta = drop(cholesky.solve(XtX, crossprod(X, y)))
  names(beta) = colnames(X)
  fitted = drop(X %*% beta)
  resid = y - fitted
  sigma = sqrt(sum(resid^2) / (nrow(X) - ncol(X)))
  list(coef = beta, fitted = fitted, resid = resid, sigma = sigma)
}

#' Ridge regression for a fixed penalty
#'
#' Centres `y` and standardises the columns of `X` before solving, then maps
#' the coefficients back to the original scale.
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @param lambda Non-negative ridge penalty.
#' @return A list with `coef`, `fitted`, `resid` and `sigma`.
#' @export
ols.ridge = function(X, y, lambda = 1) {
  stopifnot(length(lambda) == 1, lambda >= 0)

  # The intercept is not penalised: it is handled by centring, so `X` must
  # not contain a constant column.
  mx = colMeans(X)
  sx = apply(X, 2, stats::sd)
  if (any(sx == 0)) {
    stop("`X` must not contain constant columns; the intercept is added.", call. = FALSE)
  }
  Z = sweep(sweep(X, 2, mx), 2, sx, "/")
  my = mean(y)

  gamma = drop(cholesky.solve(crossprod(Z) + diag(lambda, ncol(Z)), crossprod(Z, y - my)))

  # Back to the original scale of X.
  beta = gamma / sx
  intercept = my - sum(mx * beta)
  fitted = drop(intercept + X %*% beta)
  resid = y - fitted
  sigma = sqrt(sum(resid^2) / (nrow(X) - ncol(X) - 1))
  list(coef = c("(Intercept)" = intercept, beta), fitted = fitted, resid = resid,
    sigma = sigma)
}

#' Dominant eigenpair by power iteration
#'
#' @param A A square numeric matrix.
#' @param tol Convergence tolerance on the eigenvalue.
#' @param max.iter Maximum number of iterations.
#' @return A list with `value`, `vector` and `iterations`.
#' @export
power.iteration = function(A, tol = 1e-10, max.iter = 1000L) {
  check.square(A)
  v = rep(1 / sqrt(nrow(A)), nrow(A))
  lambda = 0

  for (iter in seq_len(max.iter)) {
    w = drop(A %*% v)
    v.new = w / sqrt(sum(w^2))
    lambda.new = drop(crossprod(v.new, A %*% v.new))
    if (abs(lambda.new - lambda) < tol) {
      return(list(value = lambda.new, vector = v.new, iterations = iter))
    }
    v = v.new
    lambda = lambda.new
  }

  warning("power.iteration did not converge in `max.iter` iterations.", call. = FALSE)
  list(value = lambda, vector = v, iterations = max.iter)
}

#' Squared Mahalanobis distances
#'
#' @param X Numeric matrix with one observation per row.
#' @param center Numeric vector of column centres.
#' @param cov Covariance matrix of the columns.
#' @return A numeric vector of squared distances, one per row of `X`.
#' @export
mahal.sq = function(X, center = colMeans(X), cov = stats::cov(X)) {
  R = check.pos.def(cov, "cov")
  D = sweep(X, 2, center)

  # With cov = R'R, the squared distance is |R'^{-1} d|^2 for each row d.
  Z = forwardsolve(t(R), t(D))
  colSums(Z^2)
}
