#' Descriptive profile of a demand series
#'
#' @param x Numeric demand per period, zeros included.
#' @return A list with elements `n`, `mean`, `sd`, `cv`, `adi`, `cv2`,
#'   `zero.share` and `trend`.
#' @export
demand.profile <- function(x) {
  if (!is.numeric(x) || length(x) < 2) {
    stop("`x` must be a numeric vector with at least two periods.", call. = FALSE)
  }
  x <- x[!is.na(x)]
  n <- length(x)
  nonzero <- x[x > 0]

  # Level and spread over all periods
  avg <- mean(x)
  spread <- stats::sd(x)
  cv <- if (avg > 0) spread / avg else NA_real_
  # Average interval between non-zero demands
  adi <- if (length(nonzero) > 0) n / length(nonzero) else Inf
  # Squared variation of the non-zero sizes only
  cv2 <- if (length(nonzero) > 1) (stats::sd(nonzero) / mean(nonzero))^2 else 0
  # Share of periods without demand
  zero.share <- mean(x == 0)
  # Least-squares slope per period
  trend <- unname(stats::coef(stats::lm(x ~ seq_len(n)))[2])

  return(list(n = n, mean = avg, sd = spread, cv = cv, adi = adi, cv2 = cv2,
    zero.share = zero.share, trend = trend))
}

#' Classify items by their demand profile
#'
#' @param profiles A data frame with one row per item and columns `item`,
#'   `mean`, `cv`, `adi`, `cv2` and `zero.share`.
#' @param method Classification, one of `"syntetos"`, `"xyz"`, `"fsn"` or
#'   `"volume"`.
#' @param cutoffs Boundaries used by the methods: `adi` and `cv2` for
#'   `"syntetos"`, `x` and `y` for `"xyz"`, `fast` and `slow` for `"fsn"`,
#'   `a` and `b` for `"volume"`.
#' @return `profiles` with an added `class` column.
#' @export
classify.demand <- function(profiles, method = c("syntetos", "xyz", "fsn", "volume"),
                            cutoffs = c(adi = 1.32, cv2 = 0.49, x = 0.5, y = 1,
                                        fast = 0.2, slow = 0.6, a = 0.8, b = 0.95)) {
  method <- match.arg(method)

  profiles$class <- switch(method,
    syntetos = {
      # Interval against size variability of non-zero demands
      frequent <- profiles$adi < cutoffs[["adi"]]
      steady <- profiles$cv2 < cutoffs[["cv2"]]
      ifelse(frequent, ifelse(steady, "smooth", "erratic"),
        ifelse(steady, "intermittent", "lumpy"))
    },
    xyz = {
      # Coefficient of variation over all periods
      cv <- profiles$cv
      ifelse(is.na(cv), "Z", ifelse(cv <= cutoffs[["x"]], "X",
        ifelse(cv <= cutoffs[["y"]], "Y", "Z")))
    },
    fsn = {
      # Share of periods without demand
      idle <- profiles$zero.share
      ifelse(idle <= cutoffs[["fast"]], "fast",
        ifelse(idle <= cutoffs[["slow"]], "slow", "non-moving"))
    },
    volume = {
      # Cumulative share of total demand, largest items first
      share <- profiles$mean / sum(profiles$mean)
      cumulative <- cumsum(sort(share, decreasing = TRUE))[rank(-share, ties.method = "first")]
      ifelse(cumulative <= cutoffs[["a"]], "A",
        ifelse(cumulative <= cutoffs[["b"]], "B", "C"))
    }
  )
  return(profiles)
}

#' Forecast demand for the next periods
#'
#' @param x Numeric demand per period, oldest first.
#' @param method Forecasting method, one of `"naive"`, `"mean"`,
#'   `"moving"`, `"ses"` or `"croston"`.
#' @param h Number of periods to forecast.
#' @param window Number of recent periods used by `"moving"`.
#' @param alpha Smoothing constant used by `"ses"` and `"croston"`.
#' @return A numeric vector of length `h` with a `method` attribute.
#' @export
forecast.demand <- function(x, method = c("naive", "mean", "moving", "ses", "croston"),
                            h = 1, window = 3, alpha = 0.2) {
  method <- match.arg(method)
  n <- length(x)

  level <- switch(method,
    naive = {
      # Last observation carried forward
      x[n]
    },
    mean = {
      # Average of the whole history
      mean(x)
    },
    moving = {
      # Average of the most recent periods
      recent <- x[max(1, n - window + 1):n]
      mean(recent)
    },
    ses = {
      # Simple exponential smoothing started at the first value
      smoothed <- x[1]
      for (t in seq_len(n)[-1]) smoothed <- alpha * x[t] + (1 - alpha) * smoothed
      smoothed
    },
    croston = {
      # Smooth non-zero sizes and the intervals between them separately
      hits <- which(x > 0)
      if (length(hits) == 0) return(structure(rep(0, h), method = method))
      size <- x[hits[1]]
      interval <- hits[1]
      for (k in seq_along(hits)[-1]) {
        size <- alpha * x[hits[k]] + (1 - alpha) * size
        interval <- alpha * (hits[k] - hits[k - 1]) + (1 - alpha) * interval
      }
      size / interval
    }
  )
  return(structure(rep(level, h), method = method))
}

#' Accuracy measures of a forecast
#'
#' @param actual Observed demand.
#' @param forecast Forecast for the same periods.
#' @param naive.mae Mean absolute error of the naive forecast, used to scale
#'   `mase`; computed from `actual` when `NULL`.
#' @return A list with elements `me`, `mae`, `rmse`, `mape`, `smape` and
#'   `mase`.
#' @export
forecast.errors <- function(actual, forecast, naive.mae = NULL) {
  if (length(actual) != length(forecast)) {
    stop("`actual` and `forecast` must have the same length.", call. = FALSE)
  }
  error <- actual - forecast
  if (is.null(naive.mae)) naive.mae <- mean(abs(diff(actual)))

  # Mean error, positive when the forecast is too low
  me <- mean(error)
  # Mean absolute error
  mae <- mean(abs(error))
  # Root mean squared error
  rmse <- sqrt(mean(error^2))
  # Mean absolute percentage error over non-zero actuals
  pos <- actual != 0
  mape <- if (any(pos)) 100 * mean(abs(error[pos]) / abs(actual[pos])) else NA_real_
  # Symmetric percentage error over periods where either side is non-zero
  denom <- abs(actual) + abs(forecast)
  smape <- if (any(denom > 0)) 200 * mean(abs(error[denom > 0]) / denom[denom > 0]) else 0
  # Mean absolute scaled error
  mase <- if (naive.mae > 0) mae / naive.mae else NA_real_

  return(list(me = me, mae = mae, rmse = rmse, mape = mape, smape = smape, mase = mase))
}

#' Demand over the lead time at a service level
#'
#' @param x Numeric demand per period.
#' @param lead.time Lead time in whole periods.
#' @param method Distribution assumed for lead-time demand, one of
#'   `"normal"`, `"empirical"`, `"poisson"` or `"gamma"`.
#' @param service Probability of covering lead-time demand.
#' @return A list with elements `mean`, `sd`, `quantile` and `safety`.
#' @export
lead.time.demand <- function(x, lead.time, method = c("normal", "empirical", "poisson", "gamma"),
                             service = 0.95) {
  method <- match.arg(method)
  mu <- mean(x) * lead.time
  sigma <- stats::sd(x) * sqrt(lead.time)

  q <- switch(method,
    normal = {
      # Sum of independent periods is roughly normal
      z <- stats::qnorm(service)
      mu + z * sigma
    },
    empirical = {
      # Rolling sums of the observed periods
      sums <- stats::filter(x, rep(1, lead.time), sides = 1)
      stats::quantile(sums[!is.na(sums)], service, names = FALSE)
    },
    poisson = {
      # Count demand with variance equal to the mean
      sigma <- sqrt(mu)
      stats::qpois(service, mu)
    },
    gamma = {
      # Skewed demand matched on mean and variance
      shape <- mu^2 / sigma^2
      stats::qgamma(service, shape = shape, rate = shape / mu)
    }
  )
  return(list(mean = mu, sd = sigma, quantile = q, safety = q - mu))
}
