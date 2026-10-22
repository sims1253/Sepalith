#' EWMA statistic
#'
#' Computes the exponentially weighted moving average of a series, carrying
#' the previous value forward over missing observations.
#'
#' @param x Numeric vector of individual observations.
#' @param lambda Smoothing weight in (0, 1].
#' @param start Starting value, usually the in-control mean.
#' @return A numeric vector the same length as `x`.
#' @export
ewma.stat = function(x, lambda = 0.2, start = mean(x, na.rm = TRUE)) {
  if (lambda <= 0 || lambda > 1) {
    stop("`lambda` must be in (0, 1].", call. = FALSE)
  }

  n = length(x)
  z = numeric(n)
  prev = start
  for (i in seq_len(n)) {
    if (is.na(x[i])) {
      z[i] = prev
    } else {
      z[i] = lambda * x[i] + (1 - lambda) * prev
    }
    prev = z[i]
  }
  z
}

#' EWMA control limits
#'
#' Computes time-varying limits for the EWMA statistic, or the steady-state
#' limits when `asymptotic` is `TRUE`.
#'
#' @param n Number of time points.
#' @param target In-control process mean.
#' @param sigma Process standard deviation.
#' @param lambda Smoothing weight in (0, 1].
#' @param L Width of the limits in standard deviations of the statistic.
#' @param asymptotic Whether to use the steady-state variance.
#' @return A data frame with columns `lcl` and `ucl`.
#' @export
ewma.limits = function(n, target, sigma, lambda = 0.2, L = 3, asymptotic = FALSE) {
  # Variance of the statistic in units of sigma^2; it approaches the
  # steady state geometrically from the start of the series.
  steady = lambda / (2 - lambda)
  if (asymptotic) {
    v = rep(steady, n)
  } else {
    v = steady * (1 - (1 - lambda)^(2 * seq_len(n)))
  }

  half.width = L * sigma * sqrt(v)
  data.frame(
    lcl = target - half.width,
    ucl = target + half.width
  )
}

#' EWMA control chart
#'
#' Combines the EWMA statistic with its limits and records the points that
#' fall outside.
#'
#' @param x Numeric vector of individual observations.
#' @param target In-control process mean.
#' @param sigma Process standard deviation; estimated from moving ranges when
#'   `NULL`.
#' @param lambda Smoothing weight in (0, 1].
#' @param L Width of the limits in standard deviations of the statistic.
#' @return An object of class `qc_ewma`.
#' @export
ewma.chart = function(x, target, sigma = NULL, lambda = 0.2, L = 3) {
  if (!is.numeric(x) || length(x) < 2L) {
    stop("`x` must be a numeric vector with at least two values.", call. = FALSE)
  }
  if (lambda <= 0 || lambda > 1) {
    stop("`lambda` must be in (0, 1].", call. = FALSE)
  }
  if (L <= 0) {
    stop("`L` must be positive.", call. = FALSE)
  }
  if (is.null(sigma)) {
    sigma = mean(abs(diff(x)), na.rm = TRUE) / 1.128
  } else if (sigma <= 0) {
    stop("`sigma` must be positive.", call. = FALSE)
  }

  z = ewma.stat(x, lambda = lambda, start = target)
  lim = ewma.limits(length(x), target, sigma, lambda = lambda, L = L)
  outside = z < lim$lcl | z > lim$ucl
  signals = which(outside)

  # run length is the index of the first alarm, or NA when none occurred
  run.length = if (length(signals) > 0L) signals[1L] else NA_integer_
  structure(
    list(
      x = x,
      stat = z,
      lcl = lim$lcl,
      ucl = lim$ucl,
      target = target,
      sigma = sigma,
      lambda = lambda,
      L = L,
      signals = signals,
      run.length = run.length
    ),
    class = "qc_ewma"
  )
}

#' Simulated average run length of an EWMA chart
#'
#' Estimates the ARL by simulating standard normal series with a mean shift
#' and recording the first alarm of each run.
#'
#' @param lambda Smoothing weight in (0, 1].
#' @param L Width of the limits in standard deviations of the statistic.
#' @param delta Mean shift in standard deviations.
#' @param n.sim Number of simulated runs.
#' @param max.run Maximum length of a single run before it is censored.
#' @return A list with the ARL estimate, its standard error and the number of
#'   censored runs.
#' @export
ewma.arl.sim = function(lambda = 0.2, L = 3, delta = 0, n.sim = 1000, max.run = 5000) {
  if (lambda <= 0 || lambda > 1) {
    stop("`lambda` must be in (0, 1].", call. = FALSE)
  }
  if (n.sim < 2L || max.run < 1L) {
    stop("`n.sim` must be at least 2 and `max.run` at least 1.", call. = FALSE)
  }

  # Limits for a standardized process, precomputed for every time step.
  half.width = L * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2 * seq_len(max.run))))
  run.lengths = integer(n.sim)
  censored = 0L

  for (s in seq_len(n.sim)) {
    z = 0
    alarm = NA_integer_
    for (t in seq_len(max.run)) {
      z = lambda * stats::rnorm(1L, mean = delta) + (1 - lambda) * z
      if (abs(z) > half.width[t]) {
        alarm = t
        break
      }
    }
    # Censored runs count as max.run, which biases the ARL downwards.
    if (is.na(alarm)) {
      censored = censored + 1L
      alarm = max.run
    }
    run.lengths[s] = alarm
  }

  if (censored > 0L) {
    warning(sprintf("%d of %d runs were censored at %d.", censored, n.sim, max.run))
  }
  list(
    arl = mean(run.lengths),
    se = stats::sd(run.lengths) / sqrt(n.sim),
    censored = censored
  )
}

#' Compare EWMA charts across smoothing weights
#'
#' Runs [ewma.chart()] once per value of `lambdas` and reports when each one
#' first signals.
#'
#' @param x Numeric vector of individual observations.
#' @param target In-control process mean.
#' @param sigma Process standard deviation; estimated from moving ranges when
#'   `NULL`.
#' @param lambdas Numeric vector of smoothing weights to compare.
#' @param L Width of the limits in standard deviations of the statistic.
#' @return A data frame with one row per smoothing weight.
#' @export
ewma.compare = function(x, target, sigma = NULL, lambdas = c(0.05, 0.1, 0.2, 0.4), L = 3) {
  first.signal = rep(NA_integer_, length(lambdas))
  n.signals = integer(length(lambdas))

  for (i in seq_along(lambdas)) {
    chart = ewma.chart(x, target, sigma = sigma, lambda = lambdas[i], L = L)
    n.signals[i] = length(chart$signals)
    if (n.signals[i] > 0L) {
      first.signal[i] = chart$signals[1L]
    }
  }

  data.frame(
    lambda = lambdas,
    first.signal = first.signal,
    n.signals = n.signals
  )
}

#' Print and summarise an EWMA chart
#'
#' `print()` shows the design parameters and the number of points beyond the
#' limits.
#'
#' @param x An object of class `qc_ewma`.
#' @param ... Unused.
#' @return `print()` returns `x`, invisibly.
#' @export
print.qc_ewma = function(x, ...) {
  cat("<EWMA chart>\n")
  cat(sprintf(
    "  target = %g, sigma = %g, lambda = %g, L = %g\n",
    x$target, x$sigma, x$lambda, x$L
  ))
  cat(sprintf("  observations: %d\n", length(x$x)))
  cat(sprintf("  out of control: %d\n", length(x$signals)))
  invisible(x)
}

#' @rdname print.qc_ewma
#' @description `summary()` collects the design, the signalling points and
#'   the largest standardized excursion of the statistic into a list of class
#'   `summary.qc_ewma`.
#' @param object An object of class `qc_ewma`.
#' @export
summary.qc_ewma = function(object, ...) {
  # Excursions are relative to the limits: 1 means on the limit.
  half.width = (object$ucl - object$lcl) / 2
  excursion = (object$stat - object$target) / half.width

  out = list(
    design = c(
      target = object$target,
      sigma = object$sigma,
      lambda = object$lambda,
      L = object$L
    ),
    signals = object$signals,
    max.excursion = max(abs(excursion)),
    in.control = length(object$signals) == 0L
  )
  class(out) = "summary.qc_ewma"
  out
}
