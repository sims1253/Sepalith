#' Newton-Cotes stencils used by the composite rules
#'
#' @description
#' Each stencil gives the weights of one panel in units of the step `h`.
#' The open midpoint rule uses a single node per panel.
#'
#' @keywords internal
quad_stencils <- list(
  midpoint = 1,
  trapezoid = c(1, 1) / 2,
  simpson = c(1, 4, 1) / 3,
  boole = c(7, 32, 12, 32, 7) * 2 / 45
)

#' Assemble composite weights from a panel stencil
#'
#' @param stencil Numeric vector of panel weights, as in `quad_stencils`.
#' @param n Number of sub-intervals; must be a multiple of the panel width.
#' @return A numeric vector of length `n + 1` with the composite weights.
#' @keywords internal
composite_weights <- function(stencil, n) {
  width <- length(stencil) - 1L
  if (n %% width != 0L) {
    stop(sprintf("`n` must be a multiple of %d for this rule.", width), call. = FALSE)
  }
  weights <- numeric(n + 1L)
  for (start in seq(1L, n, by = width)) {
    idx <- start:(start + width)
    weights[idx] <- weights[idx] + stencil
  }
  return(weights)
}

#' Integrate a function with a composite Newton-Cotes rule
#'
#' @description
#' Evaluates `f` on an equally spaced grid over `[a, b]` and combines the
#' values with the weights of the chosen rule.
#'
#' @param f A vectorised function of one argument.
#' @param a Lower limit of integration.
#' @param b Upper limit of integration.
#' @param n Number of sub-intervals.
#' @param method One of `"midpoint"`, `"trapezoid"`, `"simpson"` or `"boole"`.
#' @return A list with elements `value`, `nodes`, `weights` and `method`.
#' @export
quad_rule <- function(f, a, b, n = 12L,
                      method = c("midpoint", "trapezoid", "simpson", "boole")) {
  method <- match.arg(method)

  # Validate `a`
  if (!is.numeric(a) || length(a) != 1L || !is.finite(a)) {
    stop(sprintf("`%s` must be a single finite number.", "a"), call. = FALSE)
  }

  # Validate `b`
  if (!is.numeric(b) || length(b) != 1L || !is.finite(b)) {
    stop(sprintf("`%s` must be a single finite number.", "b"), call. = FALSE)
  }

  # Validate `n`
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop(sprintf("`%s` must be a single positive integer.", "n"), call. = FALSE)
  }

  n <- as.integer(n)
  h <- (b - a) / n
  switch(method,
    midpoint = {
      nodes <- a + h * (seq_len(n) - 0.5)
      weights <- h * rep(quad_stencils$midpoint, n)
    },
    trapezoid = {
      nodes <- a + h * (0:n)
      weights <- h * composite_weights(quad_stencils$trapezoid, n)
    },
    simpson = {
      nodes <- a + h * (0:n)
      weights <- h * composite_weights(quad_stencils$simpson, n)
    },
    boole = {
      nodes <- a + h * (0:n)
      weights <- h * composite_weights(quad_stencils$boole, n)
    }
  )

  values <- f(nodes)
  if (length(values) != length(nodes)) {
    stop("`f` must return one value per node.", call. = FALSE)
  }
  return(list(value = sum(weights * values), nodes = nodes, weights = weights,
              method = method))
}

#' Map an integral onto a finite interval
#'
#' @description
#' Substitutes the integration variable so that an integral over an
#' infinite range becomes an integral over a finite one. The transformed
#' integrand is singular at the new end points, so use an open rule such as
#' `quad_rule(method = "midpoint")` on the result.
#'
#' @param f A vectorised function of one argument.
#' @param a Lower limit of integration, possibly `-Inf`.
#' @param b Upper limit of integration, possibly `Inf`.
#' @return A list with elements `g` (the transformed integrand), `lower`,
#'   `upper` and `type`.
#' @export
quad_transform <- function(f, a, b) {
  type <- if (is.finite(a) && is.finite(b)) {
    "finite"
  } else if (is.finite(b)) {
    "lower_inf"
  } else if (is.finite(a)) {
    "upper_inf"
  } else {
    "both_inf"
  }

  switch(type,
    finite = {
      g <- f
      lower <- a
      upper <- b
    },
    lower_inf = {
      g <- function(t) f(b - (1 - t) / t) / t^2
      lower <- 0
      upper <- 1
    },
    upper_inf = {
      g <- function(t) f(a + t / (1 - t)) / (1 - t)^2
      lower <- 0
      upper <- 1
    },
    both_inf = {
      g <- function(t) f(t / (1 - t^2)) * (1 + t^2) / (1 - t^2)^2
      lower <- -1
      upper <- 1
    }
  )
  return(list(g = g, lower = lower, upper = upper, type = type))
}

#' Romberg integration
#'
#' Repeatedly halves the trapezoid step and applies Richardson
#' extrapolation to the sequence of estimates.
#'
#' @param f A vectorised function of one argument.
#' @param a Lower limit of integration.
#' @param b Upper limit of integration.
#' @param levels Maximum number of halvings.
#' @param tol Absolute tolerance between successive diagonal entries.
#' @return A list with elements `value`, `table`, `levels` and `converged`.
#' @export
romberg <- function(f, a, b, levels = 8L, tol = 1e-10) {
  table <- matrix(NA_real_, levels, levels)
  h <- b - a
  table[1, 1] <- h * (f(a) + f(b)) / 2
  converged <- FALSE
  used <- 1L

  for (i in seq_len(levels)[-1]) {
    h <- h / 2
    # Only the new midpoints need evaluating
    new_nodes <- a + h * seq(1, 2^(i - 1), by = 2)
    table[i, 1] <- table[i - 1, 1] / 2 + h * sum(f(new_nodes))
    for (j in 2:i) {
      factor <- 4^(j - 1)
      table[i, j] <- (factor * table[i, j - 1] - table[i - 1, j - 1]) / (factor - 1)
    }
    used <- i
    if (abs(table[i, i] - table[i - 1, i - 1]) < tol) {
      converged <- TRUE
      break
    }
  }

  table <- table[seq_len(used), seq_len(used), drop = FALSE]
  return(list(value = table[used, used], table = table, levels = used,
              converged = converged))
}

#' Gaussian quadrature rules
#'
#' @description
#' Uses the Golub-Welsch method: the nodes are the eigenvalues of the
#' symmetric Jacobi matrix built from the three-term recurrence of the
#' orthogonal polynomials, and the weights are `mu0` times the squared first
#' components of the eigenvectors.
#'
#' @param n Number of nodes.
#' @param family One of `"legendre"` (weight 1 on `[-1, 1]`), `"chebyshev1"`
#'   (weight `1 / sqrt(1 - x^2)`), `"chebyshev2"` (weight `sqrt(1 - x^2)`),
#'   `"hermite"` (weight `exp(-x^2)` on the real line) or `"laguerre"`
#'   (weight `exp(-x)` on `[0, Inf)`).
#' @return A list with elements `nodes`, `weights` and `family`.
#' @export
gauss_rule <- function(n, family = c("legendre", "chebyshev1", "chebyshev2",
                                     "hermite", "laguerre")) {
  family <- match.arg(family)
  if (!is.numeric(n) || length(n) != 1L || n < 1) {
    stop(sprintf("`%s` must be a single positive integer.", "n"), call. = FALSE)
  }
  n <- as.integer(n)
  k <- seq_len(n - 1L)

  # Recurrence coefficients `alpha`, `beta` and the total weight `mu0`
  switch(family,
    legendre = {
      alpha <- rep(0, n)
      beta <- k^2 / (4 * k^2 - 1)
      mu0 <- 2
    },
    chebyshev1 = {
      alpha <- rep(0, n)
      beta <- ifelse(k == 1L, 1 / 2, 1 / 4)
      mu0 <- pi
    },
    chebyshev2 = {
      alpha <- rep(0, n)
      beta <- rep(1 / 4, n - 1L)
      mu0 <- pi / 2
    },
    hermite = {
      alpha <- rep(0, n)
      beta <- k / 2
      mu0 <- sqrt(pi)
    },
    laguerre = {
      alpha <- 2 * seq_len(n) - 1
      beta <- k^2
      mu0 <- 1
    }
  )

  jacobi <- diag(alpha, n)
  jacobi[cbind(k, k + 1L)] <- sqrt(beta)
  jacobi[cbind(k + 1L, k)] <- sqrt(beta)
  decomposed <- eigen(jacobi, symmetric = TRUE)
  ord <- order(decomposed$values)
  return(list(nodes = decomposed$values[ord], weights = mu0 * decomposed$vectors[1, ord]^2,
              family = family))
}

#' Compare quadrature rules against a known value
#'
#' @param f A vectorised function of one argument.
#' @param a Lower limit of integration.
#' @param b Upper limit of integration.
#' @param exact The exact value of the integral.
#' @param n Integer vector of sub-interval counts; each must suit every rule.
#' @param methods Character vector of rules understood by [quad_rule()].
#' @return A data frame with columns `method`, `n`, `value` and `error`.
#' @export
quad_compare <- function(f, a, b, exact, n = c(4L, 8L, 16L),
                         methods = c("midpoint", "trapezoid", "simpson", "boole")) {
  # Validate `exact`
  if (!is.numeric(exact) || length(exact) != 1L || !is.finite(exact)) {
    stop(sprintf("`%s` must be a single finite number.", "exact"), call. = FALSE)
  }

  # Validate `n`
  if (!is.numeric(n) || length(n) < 1L || any(n < 1)) {
    stop(sprintf("`%s` must be a vector of positive integers.", "n"), call. = FALSE)
  }

  # Validate `methods`
  unknown <- setdiff(methods, names(quad_stencils))
  if (length(unknown) > 0L) {
    stop(sprintf("Unknown `methods`: %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }

  rows <- vector("list", length(methods) * length(n))
  k <- 0L
  for (method in methods) {
    for (m in n) {
      k <- k + 1L
      value <- quad_rule(f, a, b, n = m, method = method)$value
      rows[[k]] <- data.frame(method = method, n = m, value = value,
                              error = abs(value - exact))
    }
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  return(out)
}
