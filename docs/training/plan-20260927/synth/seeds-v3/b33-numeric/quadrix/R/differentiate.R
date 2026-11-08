#' First derivative by finite differences
#'
#' @param f A function of one argument.
#' @param x The point at which to differentiate.
#' @param h Step size; by default scaled to `x` and the machine precision.
#' @param scheme One of `"central"`, `"forward"`, `"backward"` or
#'   `"five_point"`.
#' @return A single number.
#' @rdname fd_derivative
#' @export
fd_derivative <- function(f, x, h = NULL,
                          scheme = c("central", "forward", "backward", "five_point")) {
  scheme <- match.arg(scheme)
  if (is.null(h)) {
    h <- .Machine$double.eps^(1 / 3) * max(1, abs(x))
  }
  switch(scheme,
    forward = {
      offsets <- c(0, 1)
      coefs <- c(-1, 1)
    },
    backward = {
      offsets <- c(-1, 0)
      coefs <- c(-1, 1)
    },
    central = {
      offsets <- c(-1, 1)
      coefs <- c(-1, 1) / 2
    },
    five_point = {
      offsets <- c(-2, -1, 1, 2)
      coefs <- c(1, -8, 8, -1) / 12
    }
  )
  values <- vapply(x + offsets * h, f, numeric(1))
  return(sum(coefs * values) / h)
}

#' @rdname fd_derivative
#' @export
fd_second <- function(f, x, h = NULL,
                      scheme = c("central", "forward", "backward", "five_point")) {
  scheme <- match.arg(scheme)
  if (is.null(h)) {
    h <- .Machine$double.eps^(1 / 4) * max(1, abs(x))
  }
  switch(scheme,
    forward = {
      offsets <- c(0, 1, 2)
      coefs <- c(1, -2, 1)
    },
    backward = {
      offsets <- c(-2, -1, 0)
      coefs <- c(1, -2, 1)
    },
    central = {
      offsets <- c(-1, 0, 1)
      coefs <- c(1, -2, 1)
    },
    five_point = {
      offsets <- c(-2, -1, 0, 1, 2)
      coefs <- c(-1, 16, -30, 16, -1) / 12
    }
  )
  values <- vapply(x + offsets * h, f, numeric(1))
  return(sum(coefs * values) / h^2)
}

#' Derivative of sampled values on an equally spaced grid
#'
#' @description
#' Interior points use the central difference. The end points use one-sided
#' differences of the requested accuracy: a forward difference at the head
#' and the mirrored backward difference at the tail.
#'
#' @param y Numeric vector of function values at equally spaced points.
#' @param h The grid spacing.
#' @param accuracy Order of the one-sided end-point formulas, 1 or 2.
#' @return A numeric vector the length of `y`, named like `y`.
#' @export
fd_grid <- function(y, h = 1, accuracy = 2L) {
  n <- length(y)
  if (!is.numeric(y) || n < 3L) {
    stop(sprintf("`%s` must be a numeric vector of length at least 3.", "y"), call. = FALSE)
  }
  if (!is.numeric(h) || length(h) != 1L || !(h > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "h"), call. = FALSE)
  }
  if (!(accuracy %in% c(1L, 2L))) {
    stop("`accuracy` must be 1 or 2.", call. = FALSE)
  }
  out <- numeric(n)

  # Interior points: central difference
  out[2:(n - 1L)] <- (y[3:n] - y[1:(n - 2L)]) / (2 * h)

  # Head: forward difference
  if (accuracy == 1L) {
    out[1] <- (y[2] - y[1]) / h
  } else {
    out[1] <- (-3 * y[1] + 4 * y[2] - y[3]) / (2 * h)
  }

  # Tail: backward difference
  if (accuracy == 1L) {
    out[n] <- (y[n] - y[n - 1L]) / h
  } else {
    out[n] <- (3 * y[n] - 4 * y[n - 1L] + y[n - 2L]) / (2 * h)
  }
  names(out) <- names(y)
  return(out)
}

#' Gradient of a scalar function
#'
#' @param f A function of a numeric vector returning a single number.
#' @param x The point at which to differentiate.
#' @param h Step size passed to [fd_derivative()]: `NULL`, a single number,
#'   or one number per element of `x`.
#' @param scheme Difference scheme passed to [fd_derivative()].
#' @return A numeric vector the length of `x`, named like `x`.
#' @export
fd_gradient <- function(f, x, h = NULL, scheme = "central") {
  # Validate `x`
  if (!is.numeric(x) || length(x) < 1L) {
    stop(sprintf("`%s` must be a non-empty numeric vector.", "x"), call. = FALSE)
  }

  # Validate `h`
  if (!is.null(h) && (!is.numeric(h) || !(length(h) %in% c(1L, length(x))))) {
    stop(sprintf("`%s` must be NULL, a single number or one per element of `x`.", "h"),
         call. = FALSE)
  }
  if (length(h) == 1L) {
    h <- rep(h, length(x))
  }

  grad <- numeric(length(x))
  for (i in seq_along(x)) {
    # Restrict `f` to the `i`-th coordinate
    f_i <- function(t) {
      y <- x
      y[i] <- t
      f(y)
    }
    grad[i] <- fd_derivative(f_i, x[i], h = h[i], scheme = scheme)
  }
  names(grad) <- names(x)
  return(grad)
}

#' Jacobian of a vector function
#'
#' @param f A function of a numeric vector returning a numeric vector.
#' @param x The point at which to differentiate.
#' @param h Step size; by default scaled to each coordinate.
#' @return A matrix with one row per output of `f` and one column per
#'   element of `x`.
#' @export
fd_jacobian <- function(f, x, h = NULL) {
  f0 <- f(x)
  jac <- matrix(0, length(f0), length(x), dimnames = list(names(f0), names(x)))
  for (j in seq_along(x)) {
    step <- if (is.null(h)) .Machine$double.eps^(1 / 3) * max(1, abs(x[j])) else h
    up <- x
    down <- x
    up[j] <- x[j] + step
    down[j] <- x[j] - step
    jac[, j] <- (f(up) - f(down)) / (2 * step)
  }
  return(jac)
}

#' Hessian of a scalar function
#'
#' @description
#' Diagonal entries use the central second difference; off-diagonal entries
#' use the four-point cross difference and are mirrored into the lower
#' triangle.
#'
#' @param f A function of a numeric vector returning a single number.
#' @param x The point at which to differentiate.
#' @param h Step size used for every coordinate.
#' @return A symmetric `length(x)` by `length(x)` matrix.
#' @export
fd_hessian <- function(f, x, h = 1e-4) {
  n <- length(x)
  hess <- matrix(0, n, n, dimnames = list(names(x), names(x)))
  f0 <- f(x)

  for (i in seq_len(n)) {
    e_i <- replace(numeric(n), i, h)
    hess[i, i] <- (f(x + e_i) - 2 * f0 + f(x - e_i)) / h^2
    for (j in seq_len(i - 1L)) {
      e_j <- replace(numeric(n), j, h)
      cross <- f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)
      hess[i, j] <- cross / (4 * h^2)
      hess[j, i] <- hess[i, j]
    }
  }
  return(hess)
}

#' Richardson extrapolation of the central difference
#'
#' @param f A function of one argument.
#' @param x The point at which to differentiate.
#' @param h Initial step size.
#' @param levels Number of step halvings.
#' @return A list with elements `value`, `error` (the change in the last
#'   extrapolation) and `table`.
#' @export
richardson_derivative <- function(f, x, h = 0.1, levels = 5L) {
  # Validate `h`
  if (!is.numeric(h) || length(h) != 1L || !(h > 0)) {
    stop(sprintf("`%s` must be a single positive number.", "h"), call. = FALSE)
  }

  # Validate `levels`
  if (!is.numeric(levels) || length(levels) != 1L || !(levels >= 1)) {
    stop(sprintf("`%s` must be a single positive number.", "levels"), call. = FALSE)
  }

  table <- matrix(NA_real_, levels, levels)
  for (i in seq_len(levels)) {
    table[i, 1] <- fd_derivative(f, x, h = h, scheme = "central")
    h <- h / 2
  }
  for (j in seq_len(levels)[-1]) {
    factor <- 4^(j - 1)
    for (i in j:levels) {
      table[i, j] <- (factor * table[i, j - 1] - table[i - 1, j - 1]) / (factor - 1)
    }
  }
  value <- table[levels, levels]
  error <- if (levels > 1L) abs(value - table[levels - 1L, levels - 1L]) else NA_real_
  return(list(value = value, error = error, table = table))
}
