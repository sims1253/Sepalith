#' Composite trapezoid rule
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param n Number of subintervals.
#' @return The approximate integral.
#' @family Newton-Cotes rules
#' @export
trapezoid.rule = function(f, a, b, n = 100L) {
  x = seq(a, b, length.out = n + 1L)
  h = (b - a) / n
  fx = f(x)
  h * (sum(fx) - (fx[1] + fx[n + 1L]) / 2)
}

#' Composite midpoint rule
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param n Number of subintervals.
#' @return The approximate integral.
#' @family Newton-Cotes rules
#' @export
midpoint.rule = function(f, a, b, n = 100L) {
  h = (b - a) / n
  x = a + h * (seq_len(n) - 0.5)
  h * sum(f(x))
}

#' Composite Simpson rule
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param n Number of subintervals, rounded up to an even number.
#' @return The approximate integral.
#' @family Newton-Cotes rules
#' @export
simpson.rule = function(f, a, b, n = 100L) {
  if (n %% 2L == 1L) {
    n = n + 1L
  }
  x = seq(a, b, length.out = n + 1L)
  h = (b - a) / n
  w = rep(c(2, 4), length.out = n + 1L)
  w[c(1L, n + 1L)] = 1
  h / 3 * sum(w * f(x))
}

#' Gauss-Legendre nodes and weights
#'
#' Computes the nodes and weights on `[-1, 1]` from the eigen-decomposition
#' of the symmetric Jacobi matrix.
#'
#' @param n Number of nodes.
#' @return A list with numeric vectors `nodes` and `weights`.
#' @family Gaussian quadrature
#' @export
gauss.legendre.nodes = function(n) {
  if (n < 1L) {
    stop("`n` must be at least 1.", call. = FALSE)
  }
  if (n == 1L) {
    return(list(nodes = 0, weights = 2))
  }
  k = seq_len(n - 1L)
  off = k / sqrt(4 * k^2 - 1)
  J = diag(0, n)
  J[cbind(k, k + 1L)] = off
  J[cbind(k + 1L, k)] = off
  eig = eigen(J, symmetric = TRUE)
  ord = order(eig$values)
  list(nodes = eig$values[ord], weights = 2 * eig$vectors[1, ord]^2)
}

#' Gauss-Legendre quadrature on a finite interval
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param n Number of nodes.
#' @return The approximate integral.
#' @family Gaussian quadrature
#' @export
gauss.legendre.rule = function(f, a, b, n = 20L) {
  gl = gauss.legendre.nodes(n)
  half = (b - a) / 2
  mid = (a + b) / 2
  half * sum(gl$weights * f(half * gl$nodes + mid))
}

#' Adaptive Simpson quadrature
#'
#' Recursively bisects the interval until the Richardson error estimate is
#' below `tol` or the recursion depth is exhausted.
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param tol Absolute error tolerance.
#' @param max.depth Maximum recursion depth.
#' @return The approximate integral, with attribute `depth.hit` set when the
#'   recursion limit was reached.
#' @export
adaptive.simpson = function(f, a, b, tol = 1e-8, max.depth = 30L) {
  depth.hit = FALSE
  simpson3 = function(fa, fm, fb, a, b) (b - a) / 6 * (fa + 4 * fm + fb)
  recurse = function(a, b, fa, fm, fb, whole, tol, depth) {
    m = (a + b) / 2
    lm = (a + m) / 2
    rm = (m + b) / 2
    flm = f(lm)
    frm = f(rm)
    left = simpson3(fa, flm, fm, a, m)
    right = simpson3(fm, frm, fb, m, b)
    delta = left + right - whole
    if (depth <= 0L) {
      depth.hit <<- TRUE
      return(left + right + delta / 15)
    }
    if (abs(delta) <= 15 * tol) {
      return(left + right + delta / 15)
    }
    recurse(a, m, fa, flm, fm, left, tol / 2, depth - 1L) +
      recurse(m, b, fm, frm, fb, right, tol / 2, depth - 1L)
  }
  fa = f(a)
  fb = f(b)
  fm = f((a + b) / 2)
  out = recurse(a, b, fa, fm, fb, simpson3(fa, fm, fb, a, b), tol, max.depth)
  if (depth.hit) {
    attr(out, "depth.hit") = TRUE
  }
  out
}

#' Integrate with a named rule
#'
#' Dispatches to one of the fixed-grid quadrature rules and optionally
#' compares the result with [stats::integrate()].
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param rule Quadrature rule to use.
#' @param n Number of subintervals or nodes.
#' @param check Should the result be compared with [stats::integrate()]?
#' @return The approximate integral, with attribute `abs.diff` when `check`
#'   is `TRUE`.
#' @export
quad.integrate = function(f, a, b, rule = c("simpson", "trapezoid", "midpoint", "gauss"),
                          n = 100L, check = FALSE) {
  rule = match.arg(rule)
  f = match.fun(f)
  value = switch(rule,
    simpson = simpson.rule(f, a, b, n),
    trapezoid = trapezoid.rule(f, a, b, n),
    midpoint = midpoint.rule(f, a, b, n),
    gauss = gauss.legendre.rule(f, a, b, n)
  )
  if (check) {
    ref = stats::integrate(f, a, b)$value
    attr(value, "abs.diff") = abs(value - ref)
  }
  value
}

#' Romberg integration
#'
#' Builds the Romberg tableau from successively refined trapezoid estimates
#' and stops when two diagonal entries agree to within `tol`.
#'
#' @param f Vectorised function to integrate.
#' @param a,b Lower and upper limits of integration.
#' @param tol Absolute error tolerance.
#' @param max.iter Maximum number of refinements.
#' @return The approximate integral, with attribute `converged`.
#' @export
romberg = function(f, a, b, tol = 1e-10, max.iter = 20L) {
  R = matrix(NA_real_, max.iter, max.iter)
  h = b - a
  R[1, 1] = h / 2 * (f(a) + f(b))
  converged = FALSE
  last = 1L
  for (i in 2:max.iter) {
    h = h / 2
    # only the new midpoints need evaluating
    new.x = a + h * seq(1, 2^(i - 1) - 1, by = 2)
    R[i, 1] = R[i - 1, 1] / 2 + h * sum(f(new.x))
    for (j in 2:i) {
      R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4^(j - 1) - 1)
    }
    last = i
    if (abs(R[i, i] - R[i - 1, i - 1]) < tol) {
      converged = TRUE
      break
    }
  }
  out = R[last, last]
  attr(out, "converged") = converged
  out
}
