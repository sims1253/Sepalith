#' Simulate a homogeneous Poisson process
#'
#' Draws event times on `[0, horizon]` by accumulating exponential
#' inter-event gaps until the horizon is passed.
#'
#' @param rate Constant event rate per unit time.
#' @param horizon Length of the observation window.
#' @return A numeric vector of event times in increasing order.
#' @export
pp.homogeneous <- function(rate, horizon) {
  stopifnot(rate > 0, horizon > 0)

  # Grow one gap at a time; event counts here are small enough for c().
  times <- numeric(0)
  t <- stats::rexp(1, rate)
  while (t <= horizon) {
    times <- c(times, t)
    t <- t + stats::rexp(1, rate)
  }
  return(times)
}

#' Simulate an inhomogeneous Poisson process
#'
#' Uses thinning: candidate events are drawn from a homogeneous process at
#' the maximum rate and each is kept with probability `rate.fun(t) /
#' rate.max`.
#'
#' @param rate.fun Function of time returning the instantaneous rate.
#' @param horizon Length of the observation window.
#' @param rate.max Upper bound of `rate.fun` on the window.
#' @return A numeric vector of event times in increasing order.
#' @export
pp.inhomogeneous <- function(rate.fun, horizon, rate.max) {
  stopifnot(is.function(rate.fun), horizon > 0, rate.max > 0)
  candidates <- pp.homogeneous(rate.max, horizon)
  if (length(candidates) == 0) {
    return(candidates)
  }

  rates <- rate.fun(candidates)
  if (any(rates > rate.max + 1e-12)) {
    stop("`rate.fun` exceeds `rate.max` on the window.", call. = FALSE)
  }

  keep <- stats::runif(length(candidates)) < rates / rate.max
  return(candidates[keep])
}

#' Simulate a compound Poisson process
#'
#' Attaches an independent random mark to every event of a homogeneous
#' process and returns the running total, as used for aggregate claim or
#' demand models.
#'
#' @param rate Constant event rate per unit time.
#' @param horizon Length of the observation window.
#' @param mark.fun Function of `n` returning `n` random marks.
#' @return A data frame with columns `time`, `mark` and `total`.
#' @export
pp.compound <- function(rate, horizon, mark.fun = function(n) stats::rexp(n)) {
  times <- pp.homogeneous(rate, horizon)
  marks <- mark.fun(length(times))
  if (length(marks) != length(times)) {
    stop("`mark.fun` must return one mark per event.", call. = FALSE)
  }
  return(data.frame(time = times, mark = marks, total = cumsum(marks)))
}

#' Count events in consecutive windows
#'
#' Bins event times into equal-width windows covering `[0, horizon]`.
#'
#' @param times Numeric vector of event times.
#' @param horizon Length of the observation window.
#' @param width Width of each counting window.
#' @return A data frame with columns `start`, `end` and `count`.
#' @export
pp.window.counts <- function(times, horizon, width = 1) {
  breaks <- seq(0, horizon, by = width)
  # Close a shorter last window, but ignore floating point shortfalls such
  # as seq(0, 0.9, by = 0.3) ending just below 0.9.
  gap <- horizon - utils::tail(breaks, 1)
  if (gap > 1e-8 * width) {
    breaks <- c(breaks, horizon)
  } else {
    breaks[length(breaks)] <- horizon
  }

  bins <- cut(times, breaks = breaks, include.lowest = TRUE, right = FALSE)
  counts <- as.integer(table(bins))
  return(data.frame(
    start = utils::head(breaks, -1),
    end = breaks[-1],
    count = counts
  ))
}

#' Test window counts for Poisson dispersion
#'
#' Compares the variance of window counts with their mean using the
#' classical index-of-dispersion statistic, which is approximately
#' chi-squared under a homogeneous Poisson process.
#'
#' @param counts Integer vector of event counts per window.
#' @param alternative Direction of the test: `"two.sided"`, `"over"` or
#'   `"under"`.
#' @return A list with elements `index`, `statistic`, `df` and `p.value`.
#' @export
pp.dispersion.test <- function(counts, alternative = c("two.sided", "over", "under")) {
  alternative <- match.arg(alternative)
  k <- length(counts)
  if (k < 2) {
    stop("Need at least two windows.", call. = FALSE)
  }
  m <- mean(counts)
  if (m == 0) {
    stop("No events observed.", call. = FALSE)
  }

  statistic <- sum((counts - m)^2) / m
  df <- k - 1
  # Overdispersion shows up in the upper tail, underdispersion in the lower.
  upper <- stats::pchisq(statistic, df, lower.tail = FALSE)
  lower <- stats::pchisq(statistic, df, lower.tail = TRUE)
  p.value <- switch(alternative,
    two.sided = min(1, 2 * min(upper, lower)),
    over = upper,
    under = lower
  )
  return(list(index = stats::var(counts) / m, statistic = statistic, df = df,
    p.value = p.value))
}

#' Estimate a piecewise-constant rate
#'
#' Maximum likelihood estimate of the event rate in each window, with exact
#' Poisson confidence limits.
#'
#' @param times Numeric vector of event times.
#' @param horizon Length of the observation window.
#' @param width Width of each counting window.
#' @param level Confidence level of the interval.
#' @return The data frame from [pp.window.counts()] with added columns
#'   `rate`, `lower` and `upper`.
#' @export
pp.rate.estimate <- function(times, horizon, width = 1, level = 0.95) {
  windows <- pp.window.counts(times, horizon, width)
  exposure <- windows$end - windows$start
  alpha <- 1 - level

  # Exact (Garwood) limits from the gamma-Poisson relationship.
  lower <- ifelse(windows$count == 0, 0,
    stats::qgamma(alpha / 2, windows$count))
  upper <- stats::qgamma(1 - alpha / 2, windows$count + 1)

  windows$rate <- windows$count / exposure
  windows$lower <- lower / exposure
  windows$upper <- upper / exposure
  return(windows)
}
