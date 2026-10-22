#' Central-difference gradient
#'
#' @param fn Objective function of a numeric vector.
#' @param x Point at which to evaluate the derivative.
#' @param h Step size.
#' @return A numeric vector of partial derivatives.
#' @family numerical derivatives
#' @export
num.gradient = function(fn, x, h = 1e-5) {
  p = length(x)
  grad = numeric(p)
  for (j in seq_len(p)) {
    e = replace(numeric(p), j, h)
    grad[j] = (fn(x + e) - fn(x - e)) / (2 * h)
  }
  grad
}

#' Central-difference Hessian
#'
#' @param fn Objective function of a numeric vector.
#' @param x Point at which to evaluate the derivative.
#' @param h Step size.
#' @return A symmetric numeric matrix of second derivatives.
#' @family numerical derivatives
#' @export
num.hessian = function(fn, x, h = 1e-4) {
  p = length(x)
  hess = matrix(0, p, p)
  for (i in seq_len(p)) {
    for (j in i:p) {
      ei = replace(numeric(p), i, h)
      ej = replace(numeric(p), j, h)
      hess[i, j] = (fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej) +
        fn(x - ei - ej)) / (4 * h^2)
      hess[j, i] = hess[i, j]
    }
  }
  hess
}

#' Run optim() with uniform error handling
#'
#' Calls [stats::optim()] and converts errors and non-convergence into a
#' consistent result list.
#'
#' @param par Numeric vector of starting values.
#' @param fn Objective function to minimise.
#' @param method Optimisation method passed to [stats::optim()].
#' @param lower,upper Bounds used by `"L-BFGS-B"`.
#' @param ... Passed to `fn`.
#' @return A list with `par`, `value`, `converged` and `message`.
#' @family optimisation drivers
#' @export
optim.safe = function(par, fn, method = c("BFGS", "Nelder-Mead", "L-BFGS-B"),
                      lower = -Inf, upper = Inf, ...) {
  method = match.arg(method)
  fit = tryCatch(
    if (method == "L-BFGS-B") {
      stats::optim(par, fn, method = method, lower = lower, upper = upper, ...)
    } else {
      stats::optim(par, fn, method = method, ...)
    },
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(par = rep(NA_real_, length(par)), value = NA_real_,
      converged = FALSE, message = conditionMessage(fit)))
  }
  list(par = fit$par, value = fit$value, converged = fit$convergence == 0L,
    message = if (is.null(fit$message)) "" else fit$message)
}

#' Multi-start optimisation
#'
#' Runs [optim.safe()] from every row of `starts` and keeps the best
#' converged solution.
#'
#' @param starts Numeric matrix of starting values, one row per start.
#' @param fn Objective function to minimise.
#' @param ... Passed to [optim.safe()].
#' @return The best result list, with an extra `n.converged` element.
#' @family optimisation drivers
#' @export
multi.start = function(starts, fn, ...) {
  starts = as.matrix(starts)
  runs = lapply(seq_len(nrow(starts)), function(i) optim.safe(starts[i, ], fn, ...))
  ok = vapply(runs, function(r) isTRUE(r$converged), logical(1))
  if (!any(ok)) {
    stop("No start converged.", call. = FALSE)
  }
  values = vapply(runs[ok], function(r) r$value, numeric(1))
  best = runs[ok][[which.min(values)]]
  best$n.converged = sum(ok)
  best
}

#' Maximum likelihood fit of a normal sample
#'
#' Optimises over the mean and the log standard deviation.
#'
#' @param x Numeric vector of observations.
#' @return A list with `estimate`, `se` and `loglik`.
#' @family likelihood fits
#' @export
mle.normal = function(x) {
  x = x[is.finite(x)]
  nll = function(theta) -sum(stats::dnorm(x, theta[1], exp(theta[2]), log = TRUE))
  start = c(mean(x), log(stats::sd(x)))
  fit = optim.safe(start, nll, method = "BFGS")
  est = c(mean = fit$par[1], sd = exp(fit$par[2]))
  vc = solve(num.hessian(nll, fit$par))
  se = c(mean = sqrt(vc[1, 1]), sd = est[["sd"]] * sqrt(vc[2, 2]))
  list(estimate = est, se = se, loglik = -fit$value)
}

#' Maximum likelihood fit of a gamma sample
#'
#' Optimises over the log shape and the log rate.
#'
#' @param x Numeric vector of observations.
#' @return A list with `estimate`, `se` and `loglik`.
#' @family likelihood fits
#' @export
mle.gamma = function(x) {
  x = x[is.finite(x) & x > 0]
  nll = function(theta) -sum(stats::dgamma(x, exp(theta[1]), exp(theta[2]), log = TRUE))
  m = mean(x)
  v = stats::var(x)
  start = c(log(m^2 / v), log(m / v))
  fit = optim.safe(start, nll, method = "BFGS")
  est = c(shape = exp(fit$par[1]), rate = exp(fit$par[2]))
  vc = solve(num.hessian(nll, fit$par))
  se = est * sqrt(diag(vc))
  list(estimate = est, se = se, loglik = -fit$value)
}

#' Maximum likelihood fit of a Weibull sample
#'
#' Optimises over the log shape and the log scale.
#'
#' @param x Numeric vector of observations.
#' @return A list with `estimate`, `se` and `loglik`.
#' @family likelihood fits
#' @export
mle.weibull = function(x) {
  x = x[is.finite(x) & x > 0]
  nll = function(theta) -sum(stats::dweibull(x, exp(theta[1]), exp(theta[2]), log = TRUE))
  start = c(0, log(mean(x)))
  fit = optim.safe(start, nll, method = "BFGS")
  est = c(shape = exp(fit$par[1]), scale = exp(fit$par[2]))
  vc = solve(num.hessian(nll, fit$par))
  se = est * sqrt(diag(vc))
  list(estimate = est, se = se, loglik = -fit$value)
}

#' Newton-Raphson minimisation with step halving
#'
#' Uses numerical derivatives when `gr` or `hess` are not supplied and halves
#' the step until the objective decreases.
#'
#' @param fn Objective function of a numeric vector.
#' @param par Numeric vector of starting values.
#' @param gr Optional gradient function.
#' @param hess Optional Hessian function.
#' @param tol Convergence tolerance on the gradient norm.
#' @param max.iter Maximum number of iterations.
#' @return A list with `par`, `value`, `converged` and `iterations`.
#' @family optimisation drivers
#' @export
newton.raphson = function(fn, par, gr = NULL, hess = NULL, tol = 1e-8, max.iter = 100L) {
  if (is.null(gr)) {
    gr = function(x) num.gradient(fn, x)
  }
  if (is.null(hess)) {
    hess = function(x) num.hessian(fn, x)
  }
  value = fn(par)
  converged = FALSE
  for (iter in seq_len(max.iter)) {
    g = gr(par)
    if (sqrt(sum(g^2)) < tol) {
      converged = TRUE
      break
    }
    H = hess(par)
    step = tryCatch(solve(H, g), error = function(e) g)
    step.size = 1
    repeat {
      cand = par - step.size * step
      cand.value = fn(cand)
      if (is.finite(cand.value) && cand.value < value) {
        break
      }
      step.size = step.size / 2
      if (step.size < 1e-10) {
        return(list(par = par, value = value, converged = FALSE, iterations = iter))
      }
    }
    par = cand
    value = cand.value
  }
  list(par = par, value = value, converged = converged, iterations = iter)
}
