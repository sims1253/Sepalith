#' Critical value of the standard normal for a test
#'
#' Returns the quantile used for one- or two-sided tests at level `alpha`.
#'
#' @param alpha Type I error rate.
#' @param sides Number of sides of the test, 1 or 2.
#' @return A single positive number.
#' @keywords internal
zCritical <- function(alpha, sides = 2) {
  stopifnot(alpha > 0, alpha < 1, sides %in% c(1, 2))
  stats::qnorm(1 - alpha / sides)
}

#' Inflate a sample size for expected dropout
#'
#' Divides by the expected completion rate and rounds up.
#'
#' @param n Sample size required among completers.
#' @param dropout Expected proportion of participants lost to follow-up.
#' @return The number of participants to randomise.
#' @export
inflateForDropout <- function(n, dropout) {
  if (any(dropout < 0 | dropout >= 1)) {
    stop("`dropout` must be in [0, 1).", call. = FALSE)
  }
  ceiling(n / (1 - dropout))
}

#' Sample size for comparing two means
#'
#' Normal-approximation sample size for a parallel-group trial with a
#' continuous endpoint and allocation ratio `ratio` (arm 2 : arm 1).
#'
#' @param delta Clinically relevant difference to detect.
#' @param sd Common standard deviation of the endpoint.
#' @param alpha Type I error rate.
#' @param power Desired power.
#' @param ratio Allocation ratio of arm 2 to arm 1.
#' @param sides Number of sides of the test, 1 or 2.
#' @return A named integer vector with `n1`, `n2` and `total`.
#' @family sample size
#' @export
nTwoMeans <- function(delta, sd, alpha = 0.05, power = 0.8, ratio = 1, sides = 2) {
  stopifnot(delta != 0, sd > 0, ratio > 0)
  zA <- zCritical(alpha, sides)
  zB <- stats::qnorm(power)
  n1 <- (1 + 1 / ratio) * (sd * (zA + zB) / delta)^2
  n1 <- ceiling(n1)
  n2 <- ceiling(ratio * n1)
  c(n1 = n1, n2 = n2, total = n1 + n2)
}

#' Sample size for comparing two proportions
#'
#' Normal-approximation sample size for a parallel-group trial with a binary
#' endpoint, using the pooled variance under the null.
#'
#' @param p1 Expected response proportion in arm 1.
#' @param p2 Expected response proportion in arm 2.
#' @param alpha Type I error rate.
#' @param power Desired power.
#' @param ratio Allocation ratio of arm 2 to arm 1.
#' @param sides Number of sides of the test, 1 or 2.
#' @return A named integer vector with `n1`, `n2` and `total`.
#' @family sample size
#' @export
nTwoProportions <- function(p1, p2, alpha = 0.05, power = 0.8, ratio = 1, sides = 2) {
  stopifnot(p1 != p2, p1 > 0, p1 < 1, p2 > 0, p2 < 1, ratio > 0)
  zA <- zCritical(alpha, sides)
  zB <- stats::qnorm(power)
  pBar <- (p1 + ratio * p2) / (1 + ratio)
  num <- zA * sqrt(pBar * (1 - pBar) * (1 + 1 / ratio)) +
    zB * sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
  n1 <- ceiling((num / (p1 - p2))^2)
  n2 <- ceiling(ratio * n1)
  c(n1 = n1, n2 = n2, total = n1 + n2)
}

#' Number of events for a log-rank comparison
#'
#' Schoenfeld's approximation for the number of events needed to detect a
#' hazard ratio with the requested power.
#'
#' @param hr Target hazard ratio of arm 2 versus arm 1.
#' @param alpha Type I error rate.
#' @param power Desired power.
#' @param ratio Allocation ratio of arm 2 to arm 1.
#' @param sides Number of sides of the test, 1 or 2.
#' @return The required number of events, rounded up.
#' @family sample size
#' @export
nEventsLogrank <- function(hr, alpha = 0.05, power = 0.8, ratio = 1, sides = 2) {
  stopifnot(hr > 0, hr != 1, ratio > 0)
  zA <- zCritical(alpha, sides)
  zB <- stats::qnorm(power)
  share <- ratio / (1 + ratio)
  ceiling((zA + zB)^2 / (share * (1 - share) * log(hr)^2))
}

#' Power for comparing two means
#'
#' Inverts [nTwoMeans()]: the power achieved with `n1` participants in arm 1.
#'
#' @param n1 Number of participants in arm 1.
#' @param delta Clinically relevant difference to detect.
#' @param sd Common standard deviation of the endpoint.
#' @param alpha Type I error rate.
#' @param ratio Allocation ratio of arm 2 to arm 1.
#' @param sides Number of sides of the test, 1 or 2.
#' @return The approximate power.
#' @family power
#' @export
powerTwoMeans <- function(n1, delta, sd, alpha = 0.05, ratio = 1, sides = 2) {
  zA <- zCritical(alpha, sides)
  se <- sd * sqrt(1 / n1 + 1 / (ratio * n1))
  stats::pnorm(abs(delta) / se - zA)
}

#' Power for comparing two proportions
#'
#' Inverts [nTwoProportions()]: the power achieved with `n1` participants in
#' arm 1.
#'
#' @param n1 Number of participants in arm 1.
#' @param p1 Expected response proportion in arm 1.
#' @param p2 Expected response proportion in arm 2.
#' @param alpha Type I error rate.
#' @param ratio Allocation ratio of arm 2 to arm 1.
#' @param sides Number of sides of the test, 1 or 2.
#' @return The approximate power.
#' @family power
#' @export
powerTwoProportions <- function(n1, p1, p2, alpha = 0.05, ratio = 1, sides = 2) {
  zA <- zCritical(alpha, sides)
  n2 <- ratio * n1
  pBar <- (n1 * p1 + n2 * p2) / (n1 + n2)
  se0 <- sqrt(pBar * (1 - pBar) * (1 / n1 + 1 / n2))
  se1 <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
  stats::pnorm((abs(p1 - p2) - zA * se0) / se1)
}

#' Sample size grid over effect sizes and power levels
#'
#' Evaluates [nTwoMeans()] for every combination of `deltas` and `powers`,
#' after inflating for dropout. Handy for the sample size section of a
#' protocol.
#'
#' @param deltas Numeric vector of differences to detect.
#' @param sd Common standard deviation of the endpoint.
#' @param powers Numeric vector of desired power levels.
#' @param dropout Expected proportion of participants lost to follow-up.
#' @param ... Passed to [nTwoMeans()].
#' @return A data frame with one row per combination of `deltas` and `powers`.
#' @export
sampleSizeGrid <- function(deltas, sd, powers = c(0.8, 0.9), dropout = 0, ...) {
  grid <- expand.grid(delta = deltas, power = powers, KEEP.OUT.ATTRS = FALSE)
  grid$n1 <- NA_integer_
  grid$n2 <- NA_integer_
  grid$randomised <- NA_integer_
  for (i in seq_len(nrow(grid))) {
    n <- nTwoMeans(grid$delta[i], sd = sd, power = grid$power[i], ...)
    grid$n1[i] <- n[["n1"]]
    grid$n2[i] <- n[["n2"]]
    grid$randomised[i] <- inflateForDropout(n[["total"]], dropout)
  }
  grid$effectSize <- grid$delta / sd
  grid[order(grid$power, -abs(grid$delta)), , drop = FALSE]
}
