#' Daily flows from a trajectory
#'
#' Recovers the number of individuals moving along one flow per day from
#' the differences of consecutive compartment sizes. Assumes no waning and
#' no births or deaths.
#'
#' @param traj A data frame with columns `time`, `S`, `E`, `I` and `R`.
#' @param flow One of `"infection"` (S to E), `"onset"` (E to I) or
#'   `"recovery"` (I to R).
#' @return A numeric vector with one value per row of `traj`; the first
#'   value is `0`.
#' @export
flowFromTrajectory = function(traj, flow = c("infection", "onset", "recovery")) {
  flow = match.arg(flow)
  absent = setdiff(c("S", "E", "R"), names(traj))
  if (length(absent) > 0L) {
    stop(sprintf("Trajectory is missing column `%s`.", absent[[1L]]), call. = FALSE)
  }
  n = nrow(traj)
  if (n < 2L) {
    stop("`traj` must have at least two rows.", call. = FALSE)
  }

  daily = switch(flow,
    infection = {
      -diff(traj$S)
    },
    onset = {
      infection = -diff(traj$S)
      infection - diff(traj$E)
    },
    recovery = {
      diff(traj$R)
    }
  )
  c(0, pmax(daily, 0))
}

#' Simulate reported cases
#'
#' @param incidence Numeric vector of true daily infections or onsets.
#' @param model Observation model: `"exact"` (expected reports, no noise),
#'   `"poisson"`, `"negbin"` or `"binomial"` thinning.
#' @param reportingRate Probability that a case is reported.
#' @param size Dispersion of the negative binomial model.
#' @return An integer vector of reported cases (numeric for `"exact"`), the
#'   same length as `incidence`.
#' @export
observeCases = function(incidence, model = c("exact", "poisson", "negbin", "binomial"),
                        reportingRate = 1, size = 10) {
  model = match.arg(model)
  if (!is.numeric(reportingRate) || reportingRate < 0 || reportingRate > 1) {
    stop("`reportingRate` must lie between 0 and 1.", call. = FALSE)
  }
  n = length(incidence)
  expected = reportingRate * incidence

  switch(model,
    exact = {
      expected
    },
    poisson = {
      stats::rpois(n, lambda = expected)
    },
    negbin = {
      stats::rnbinom(n, size = size, mu = expected)
    },
    binomial = {
      stats::rbinom(n, size = round(incidence), prob = reportingRate)
    }
  )
}

#' Delay reported cases
#'
#' Convolves a case series with a reporting delay distribution truncated at
#' `maxDelay = 4 * meanDelay` days.
#'
#' @param cases Numeric vector of daily cases.
#' @param delay Delay distribution: `"none"`, `"fixed"` (exactly
#'   `meanDelay` days), `"geometric"` or `"uniform"` (on `0` to
#'   `2 * meanDelay` days).
#' @param meanDelay Mean delay in whole days.
#' @return A numeric vector of delayed cases, the same length as `cases`.
#' @export
delayCases = function(cases, delay = c("none", "fixed", "geometric", "uniform"),
                      meanDelay = 2L) {
  delay = match.arg(delay)
  maxDelay = 4L * meanDelay

  # Delay probabilities for 0, 1, 2, ... days
  weights = switch(delay,
    none = {
      1
    },
    fixed = {
      c(rep(0, meanDelay), 1)
    },
    geometric = {
      prob = 1 / (1 + meanDelay)
      stats::dgeom(0:maxDelay, prob = prob)
    },
    uniform = {
      rep(1, 2L * meanDelay + 1L)
    }
  )
  weights = weights / sum(weights)

  out = numeric(length(cases))
  for (lag in seq_along(weights) - 1L) {
    shifted = c(rep(0, lag), utils::head(cases, length(cases) - lag))
    out = out + weights[[lag + 1L]] * shifted
  }
  out
}

#' Aggregate daily counts to longer periods
#'
#' @param counts Numeric vector of daily counts.
#' @param period One of `"day"`, `"week"` or `"fortnight"`.
#' @param partial Keep a final incomplete period?
#' @return A data frame with columns `start` (first day of the period,
#'   counting from `0`) and `count`.
#' @export
aggregateCounts = function(counts, period = c("week", "day", "fortnight"), partial = FALSE) {
  period = match.arg(period)
  lengths = c(day = 1L, week = 7L, fortnight = 14L)
  width = lengths[[period]]
  index = (seq_along(counts) - 1L) %/% width
  totals = as.numeric(tapply(counts, index, sum))
  out = data.frame(start = sort(unique(index)) * width, count = totals)
  if (!partial && length(counts) %% width != 0L) {
    out = out[-nrow(out), , drop = FALSE]
  }
  out
}

#' Compare observed and expected counts
#'
#' @param observed Numeric vector of observed counts.
#' @param expected Numeric vector of expected counts, the same length.
#' @param metrics Any of `"rmse"`, `"mae"`, `"bias"` and `"deviance"`
#'   (Poisson deviance).
#' @return A named numeric vector with one element per requested metric.
#' @export
compareObserved = function(observed, expected,
                           metrics = c("rmse", "mae", "bias", "deviance")) {
  metrics = match.arg(metrics, several.ok = TRUE)
  if (length(observed) != length(expected)) {
    stop("`observed` and `expected` must have the same length.", call. = FALSE)
  }
  residual = observed - expected
  out = numeric(0)

  # Metric `rmse`
  if ("rmse" %in% metrics) {
    out[["rmse"]] = sqrt(mean(residual^2))
  }
  # Metric `mae`
  if ("mae" %in% metrics) {
    out[["mae"]] = mean(abs(residual))
  }
  # Metric `bias`
  if ("bias" %in% metrics) {
    out[["bias"]] = mean(residual)
  }
  # Metric `deviance`
  if ("deviance" %in% metrics) {
    ratio = ifelse(observed > 0, observed * log(observed / expected), 0)
    out[["deviance"]] = 2 * sum(ratio - residual)
  }
  out
}

#' Log-likelihood of observed counts
#'
#' @param observed Integer vector of observed counts.
#' @param expected Numeric vector of expected counts, the same length.
#' @param model One of `"poisson"`, `"negbin"` or `"binomial"`.
#' @param size Dispersion of the negative binomial model.
#' @param trials Number of trials per day for the binomial model.
#' @return A single number, the summed log-likelihood.
#' @export
observationLogLik = function(observed, expected, model = c("poisson", "negbin", "binomial"),
                             size = 10, trials = NULL) {
  model = match.arg(model)
  if (length(observed) != length(expected)) {
    stop("`observed` and `expected` must have the same length.", call. = FALSE)
  }

  logLik = switch(model,
    poisson = {
      stats::dpois(observed, lambda = expected, log = TRUE)
    },
    negbin = {
      stats::dnbinom(observed, size = size, mu = expected, log = TRUE)
    },
    binomial = {
      if (is.null(trials)) {
        stop("`trials` is required for the binomial model.", call. = FALSE)
      }
      stats::dbinom(observed, size = trials, prob = expected / trials, log = TRUE)
    }
  )
  sum(logLik)
}
