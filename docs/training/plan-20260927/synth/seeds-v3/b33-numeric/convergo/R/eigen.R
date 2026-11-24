#' Rayleigh quotient
#'
#' @param A A square numeric matrix.
#' @param x A non-zero vector.
#' @return The single number `t(x) A x / t(x) x`.
#' @noRd
rayleigh.quotient <- function(A, x) {
  sum(x * (A %*% x)) / sum(x^2)
}

#' Gershgorin discs
#'
#' @description
#' Every eigenvalue of `A` lies in the union of the discs centred at the
#' diagonal entries, with radii given by the off-diagonal absolute sums of
#' the rows, and also in the union of the discs built from the columns.
#'
#' @param A A square numeric matrix.
#' @return A list with elements `rows` and `cols`, each a data frame with
#'   columns `center`, `radius`, `lower`, `upper` and `isolated` (whether
#'   the disc meets no other disc of its family), and `bounds`, the tighter
#'   of the two real-axis enclosures.
#' @export
gershgorin <- function(A) {
  if (!is.matrix(A) || nrow(A) != ncol(A)) {
    stop("`A` must be a square matrix.", call. = FALSE)
  }
  n <- nrow(A)
  off <- abs(A)
  diag(off) <- 0
  center <- diag(A)

  # Discs from the rows
  row.radius <- rowSums(off)
  rows <- data.frame(center = center, radius = row.radius, lower = center - row.radius,
                     upper = center + row.radius)
  rows$isolated <- vapply(seq_len(n), function(i) {
    all(abs(center[i] - center[-i]) > row.radius[i] + row.radius[-i])
  }, logical(1))

  # Discs from the columns
  col.radius <- colSums(off)
  cols <- data.frame(center = center, radius = col.radius, lower = center - col.radius,
                     upper = center + col.radius)
  cols$isolated <- vapply(seq_len(n), function(i) {
    all(abs(center[i] - center[-i]) > col.radius[i] + col.radius[-i])
  }, logical(1))

  bounds <- c(lower = max(min(rows$lower), min(cols$lower)),
              upper = min(max(rows$upper), max(cols$upper)))
  list(rows = rows, cols = cols, bounds = bounds)
}

#' Vector iterations for a single eigenpair
#'
#' @description
#' The methods differ only in the vector update: `"power"` multiplies by
#' `A`, `"inverse"` solves with `A`, `"shifted"` solves with `A - shift * I`
#' and `"rayleigh"` solves with `A - mu * I` for the current Rayleigh
#' quotient `mu`. After each update the eigenvalue change, the vector change
#' and the residual are checked in parallel.
#'
#' @param A A square numeric matrix.
#' @param method One of `"power"`, `"inverse"`, `"shifted"` or `"rayleigh"`.
#' @param shift The shift used by `"shifted"`.
#' @param x0 Starting vector.
#' @param tol Tolerance shared by all three checks.
#' @param max.iter Maximum number of iterations.
#' @return A list with elements `value`, `vector`, `iter`, `converged` and
#'   `checks`.
#' @export
power.method <- function(A, method = c("power", "inverse", "shifted", "rayleigh"),
                         shift = 0, x0 = rep(1, nrow(A)), tol = 1e-10, max.iter = 1000L) {
  method <- match.arg(method)
  I <- diag(nrow(A))
  x <- x0 / sqrt(sum(x0^2))
  value <- rayleigh.quotient(A, x)
  converged <- FALSE

  for (iter in seq_len(max.iter)) {
    y <- switch(method,
      power = {
        drop(A %*% x)
      },
      inverse = {
        solve(A, x)
      },
      shifted = {
        solve(A - shift * I, x)
      },
      rayleigh = {
        # Near convergence the shifted matrix is numerically singular
        tryCatch(solve(A - value * I, x), error = function(e) x)
      }
    )
    y <- y / sqrt(sum(y^2))
    # Align the sign with the previous iterate before comparing
    if (sum(y * x) < 0) y <- -y
    value.new <- rayleigh.quotient(A, y)

    checks <- c(
      value = abs(value.new - value) <= tol * max(1, abs(value.new)),
      vector = sqrt(sum((y - x)^2)) <= sqrt(tol),
      residual = sqrt(sum((A %*% y - value.new * y)^2)) <= sqrt(tol) * max(1, abs(value.new))
    )
    x <- y
    value <- value.new
    if (all(checks)) {
      converged <- TRUE
      break
    }
  }
  list(value = value, vector = x, iter = iter, converged = converged, checks = checks)
}

#' Remove a known eigenpair from a matrix
#'
#' @description
#' Each method returns a matrix with the same remaining eigenvalues as `A`
#' and with `value` replaced by zero.
#'
#' @param A A square numeric matrix; `"hotelling"` and `"projection"` need
#'   it symmetric.
#' @param value An eigenvalue of `A`.
#' @param vector The matching eigenvector.
#' @param method One of `"hotelling"`, `"wielandt"` or `"projection"`.
#' @return The deflated matrix.
#' @export
deflate <- function(A, value, vector, method = c("hotelling", "wielandt", "projection")) {
  method <- match.arg(method)
  if (length(vector) != nrow(A)) {
    stop("`vector` must have one element per row of `A`.", call. = FALSE)
  }
  v <- vector / sqrt(sum(vector^2))
  switch(method,
    hotelling = {
      # Subtract the rank-one term of the spectral decomposition
      A - value * tcrossprod(v)
    },
    wielandt = {
      # Use the row where the eigenvector is largest
      i <- which.max(abs(v))
      A - tcrossprod(v, A[i, ]) / v[i]
    },
    projection = {
      # Project onto the orthogonal complement of `v`
      P <- diag(nrow(A)) - tcrossprod(v)
      P %*% A %*% P
    }
  )
}

#' Leading eigenpairs by repeated deflation
#'
#' @param A A symmetric numeric matrix.
#' @param k Number of eigenpairs to compute.
#' @param method Deflation method passed to [deflate()].
#' @param tol Tolerance passed to [power.method()].
#' @param max.iter Maximum iterations passed to [power.method()].
#' @return A list with elements `values` (length `k`), `vectors` (a matrix
#'   with `k` columns), `iter` (iterations used per pair) and `converged`.
#' @export
top.eigen <- function(A, k = 1L, method = "hotelling", tol = 1e-10, max.iter = 5000L) {
  if (!isSymmetric(A)) {
    stop("`A` must be a symmetric matrix.", call. = FALSE)
  }
  n <- nrow(A)
  if (!is.numeric(k) || length(k) != 1L || k < 1 || k > n) {
    stop(sprintf("`k` must be a single whole number between 1 and %d.", n), call. = FALSE)
  }
  values <- numeric(k)
  vectors <- matrix(0, n, k)
  iter <- integer(k)
  converged <- logical(k)
  B <- A

  for (j in seq_len(k)) {
    # Start away from the vectors already found
    x0 <- rep(1, n) + seq_len(n) / n
    fit <- power.method(B, "power", x0 = x0, tol = tol, max.iter = max.iter)
    values[j] <- fit$value
    vectors[, j] <- fit$vector
    iter[j] <- fit$iter
    converged[j] <- fit$converged
    B <- deflate(B, fit$value, fit$vector, method = method)
  }
  list(values = values, vectors = vectors, iter = iter, converged = converged)
}

#' Residual diagnostics for computed eigenpairs
#'
#' @param A A square numeric matrix.
#' @param values Numeric vector of eigenvalues.
#' @param vectors Matrix whose columns are the matching eigenvectors.
#' @return A data frame with columns `value`, `residual` (the norm of
#'   `A v - value v`), `norm` (the norm of `v`) and `orthogonality` (the
#'   largest absolute inner product of `v` with the other unit vectors).
#' @export
eigen.residuals <- function(A, values, vectors) {
  k <- length(values)
  vectors <- as.matrix(vectors)
  if (ncol(vectors) != k) {
    stop("`vectors` must have one column per eigenvalue.", call. = FALSE)
  }
  if (nrow(vectors) != nrow(A)) {
    stop("`vectors` must have one row per row of `A`.", call. = FALSE)
  }
  units <- sweep(vectors, 2L, sqrt(colSums(vectors^2)), "/")
  overlap <- abs(crossprod(units))
  diag(overlap) <- 0

  residual <- numeric(k)
  for (j in seq_len(k)) {
    residual[j] <- sqrt(sum((A %*% vectors[, j] - values[j] * vectors[, j])^2))
  }
  data.frame(
    value = values,
    residual = residual,
    norm = sqrt(colSums(vectors^2)),
    orthogonality = if (k > 1L) apply(overlap, 2L, max) else 0
  )
}
