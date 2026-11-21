#' Diagonal dominance by rows and by columns
#'
#' @param A A square numeric matrix.
#' @return A list with elements `rows` and `cols`. Each is a list with
#'   elements `margin` (the diagonal minus the off-diagonal absolute sum per
#'   row or column), `ratio` (the off-diagonal sum over the diagonal),
#'   `strict`, `weak` and `worst` (the index with the smallest margin).
#' @export
diag.dominance <- function(A) {
  if (!is.matrix(A) || nrow(A) != ncol(A)) {
    stop("`A` must be a square matrix.", call. = FALSE)
  }
  d <- abs(diag(A))
  if (any(d == 0)) {
    stop("`A` must have a non-zero diagonal.", call. = FALSE)
  }
  off <- abs(A)
  diag(off) <- 0

  # Dominance by rows
  row.off <- rowSums(off)
  row.margin <- d - row.off
  rows <- list(margin = row.margin, ratio = row.off / d, strict = all(row.margin > 0),
               weak = all(row.margin >= 0), worst = which.min(row.margin))

  # Dominance by columns
  col.off <- colSums(off)
  col.margin <- d - col.off
  cols <- list(margin = col.margin, ratio = col.off / d, strict = all(col.margin > 0),
               weak = all(col.margin >= 0), worst = which.min(col.margin))
  list(rows = rows, cols = cols)
}

#' Splitting matrices of the stationary iterative methods
#'
#' @description
#' Every method updates `x <- x + solve(M, b - A %*% x)` with a lower
#' triangular `M` built from the splitting `A = D + L + U`.
#'
#' @param A A square numeric matrix.
#' @param method One of `"jacobi"`, `"gauss.seidel"`, `"sor"` or
#'   `"richardson"`.
#' @param omega Relaxation parameter for `"sor"` and `"richardson"`.
#' @return A list with elements `M`, `G` (the iteration matrix
#'   `I - solve(M, A)`) and `radius` (the spectral radius of `G`).
#' @export
iter.split <- function(A, method = c("jacobi", "gauss.seidel", "sor", "richardson"),
                       omega = 1) {
  method <- match.arg(method)
  n <- nrow(A)
  D <- diag(diag(A), n)
  L <- A
  L[upper.tri(L, diag = TRUE)] <- 0

  M <- switch(method,
    jacobi = {
      D
    },
    gauss.seidel = {
      D + L
    },
    sor = {
      D / omega + L
    },
    richardson = {
      diag(n) / omega
    }
  )
  if (any(diag(M) == 0)) {
    stop(sprintf("Method \"%s\" needs a non-zero diagonal.", method), call. = FALSE)
  }
  G <- diag(n) - forwardsolve(M, A)
  list(M = M, G = G, radius = max(Mod(eigen(G, only.values = TRUE)$values)))
}

#' Solve a linear system by a stationary iteration
#'
#' @description
#' The step, residual and divergence checks run in parallel after every
#' sweep. The iteration has `"converged"` when both the step and the residual
#' checks hold, `"diverged"` when the residual is no longer finite, and stops
#' with `"max.iter"` otherwise.
#'
#' @inheritParams iter.split
#' @param b The right-hand side.
#' @param x0 Starting vector.
#' @param tol Relative tolerance for the step and the residual.
#' @param max.iter Maximum number of sweeps.
#' @return A list with elements `x`, `iter`, `reason`, `checks`, `residual`
#'   and `radius`.
#' @export
iter.solve <- function(A, b, method = c("jacobi", "gauss.seidel", "sor", "richardson"),
                       omega = 1, x0 = numeric(length(b)), tol = 1e-10, max.iter = 500L) {
  method <- match.arg(method)
  split <- iter.split(A, method, omega)
  b.norm <- max(sqrt(sum(b^2)), .Machine$double.eps)
  x <- x0
  r <- drop(b - A %*% x)

  for (iter in seq_len(max.iter)) {
    dx <- forwardsolve(split$M, r)
    x <- x + dx
    r <- drop(b - A %*% x)
    residual <- sqrt(sum(r^2))
    # Parallel checks on the step, the residual and divergence
    checks <- c(
      step = sqrt(sum(dx^2)) <= tol * max(1, sqrt(sum(x^2))),
      residual = residual <= tol * b.norm,
      diverged = !is.finite(residual)
    )
    if (checks[["diverged"]] || (checks[["step"]] && checks[["residual"]])) {
      break
    }
  }

  reason <- if (checks[["diverged"]]) {
    "diverged"
  } else if (checks[["step"]] && checks[["residual"]]) {
    "converged"
  } else {
    "max.iter"
  }
  list(x = x, iter = iter, reason = reason, checks = checks, residual = residual,
       radius = split$radius)
}

#' Conjugate gradient solver
#'
#' @description
#' Uses the same parallel step, residual and divergence checks as
#' [iter.solve()].
#'
#' @param A A symmetric positive definite matrix.
#' @inheritParams iter.solve
#' @return A list with elements `x`, `iter`, `reason`, `checks` and
#'   `residual`.
#' @export
cg.solve <- function(A, b, x0 = numeric(length(b)), tol = 1e-10, max.iter = 500L) {
  if (!isSymmetric(A)) {
    stop("`A` must be a symmetric matrix.", call. = FALSE)
  }
  b.norm <- max(sqrt(sum(b^2)), .Machine$double.eps)
  x <- x0
  r <- drop(b - A %*% x)
  p <- r
  rr <- sum(r^2)

  for (iter in seq_len(max.iter)) {
    Ap <- drop(A %*% p)
    alpha <- rr / sum(p * Ap)
    dx <- alpha * p
    x <- x + dx
    r <- r - alpha * Ap
    rr.new <- sum(r^2)
    residual <- sqrt(rr.new)
    # Parallel checks on the step, the residual and divergence
    checks <- c(
      step = sqrt(sum(dx^2)) <= tol * max(1, sqrt(sum(x^2))),
      residual = residual <= tol * b.norm,
      diverged = !is.finite(residual)
    )
    if (checks[["diverged"]] || (checks[["step"]] && checks[["residual"]])) {
      break
    }
    p <- r + (rr.new / rr) * p
    rr <- rr.new
  }

  reason <- if (checks[["diverged"]]) {
    "diverged"
  } else if (checks[["step"]] && checks[["residual"]]) {
    "converged"
  } else {
    "max.iter"
  }
  list(x = x, iter = iter, reason = reason, checks = checks, residual = residual)
}

#' Optimal SOR relaxation parameter
#'
#' Uses the classical formula for consistently ordered matrices, based on
#' the spectral radius of the Jacobi iteration matrix.
#'
#' @param A A square numeric matrix.
#' @return A single number in `[1, 2)`.
#' @export
sor.omega <- function(A) {
  rho <- iter.split(A, "jacobi")$radius
  if (rho >= 1) {
    stop("The Jacobi iteration does not converge for `A`.", call. = FALSE)
  }
  2 / (1 + sqrt(1 - rho^2))
}
