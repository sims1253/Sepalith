#' Centred moving average for a seasonal period
#'
#' Uses a `2 x m` moving average for even periods so the result is centred on
#' an observation, and a plain `m`-term average for odd periods.
#'
#' @param x A numeric vector.
#' @param period Seasonal period in observations.
#' @return A numeric vector the same length as `x`, `NA` at both ends.
#' @keywords internal
centredMovingAverage <- function(x, period) {
  if (period %% 2 == 1) {
    weights <- rep(1 / period, period)
  } else {
    weights <- c(0.5, rep(1, period - 1), 0.5) / period
  }
  as.numeric(stats::filter(x, weights, method = "convolution", sides = 2))
}

#' Estimate seasonal indices from a detrended series
#'
#' Averages the detrended values for each position in the cycle and
#' normalises the indices so they sum to zero (additive) or average one
#' (multiplicative).
#'
#' @param detrended A numeric vector of detrended values.
#' @param period Seasonal period in observations.
#' @param type Decomposition type, `"additive"` or `"multiplicative"`.
#' @return A numeric vector of length `period`.
#' @keywords internal
seasonalIndices <- function(detrended, period, type) {
  position <- (seq_along(detrended) - 1L) %% period + 1L
  idx <- vapply(seq_len(period), function(p) {
    mean(detrended[position == p], na.rm = TRUE)
  }, numeric(1))
  switch(type,
    additive = idx - mean(idx),
    multiplicative = idx / mean(idx)
  )
}

#' Classical seasonal decomposition
#'
#' Splits a series into trend, seasonal and remainder components using a
#' centred moving average, written out by hand so each step can be inspected.
#'
#' @param x A numeric vector or univariate `ts` object.
#' @param period Seasonal period in observations. Taken from `frequency(x)`
#'   when `x` is a `ts`.
#' @param type Decomposition type, `"additive"` or `"multiplicative"`.
#' @return An object of class `geodrift_decomp`.
#' @export
decomposeSeries <- function(x, period = stats::frequency(x),
                            type = c("additive", "multiplicative")) {
  type <- match.arg(type)
  if (!is.numeric(period) || length(period) != 1L || period < 2) {
    stop("`period` must be a single number of at least 2.", call. = FALSE)
  }
  if (length(x) < 2 * period) {
    stop("`x` must cover at least two full seasonal cycles.", call. = FALSE)
  }
  if (type == "multiplicative" && any(x <= 0, na.rm = TRUE)) {
    stop("Multiplicative decomposition needs strictly positive values.", call. = FALSE)
  }
  values <- as.numeric(x)
  trend <- centredMovingAverage(values, period)
  detrended <- if (type == "additive") values - trend else values / trend
  idx <- seasonalIndices(detrended, period, type)
  seasonal <- rep_len(idx, length(values))
  remainder <- if (type == "additive") {
    values - trend - seasonal
  } else {
    values / (trend * seasonal)
  }
  structure(
    list(
      observed = values,
      trend = trend,
      seasonal = seasonal,
      remainder = remainder,
      indices = idx,
      period = period,
      type = type,
      start = if (is.ts(x)) stats::start(x) else 1
    ),
    class = "geodrift_decomp"
  )
}

#' Print a seasonal decomposition
#'
#' @param x A `geodrift_decomp` object.
#' @param digits Number of significant digits for the seasonal indices.
#' @param ... Unused, for compatibility with [print()].
#' @return `x`, invisibly.
#' @export
print.geodrift_decomp <- function(x, digits = 3, ...) {
  cat(sprintf("<geodrift_decomp: %s, period %d, %d observations>\n",
    x$type, as.integer(x$period), length(x$observed)))
  cat("Seasonal indices:\n")
  print(signif(x$indices, digits))
  invisible(x)
}

#' Summarise a seasonal decomposition
#'
#' Reports the share of variance attributable to trend and seasonality,
#' measured on the detrended and deseasonalised series respectively.
#'
#' @param object A `geodrift_decomp` object.
#' @param ... Unused, for compatibility with [summary()].
#' @return A named numeric vector with `trendStrength` and `seasonalStrength`.
#' @export
summary.geodrift_decomp <- function(object, ...) {
  ok <- !is.na(object$trend)
  obs <- object$observed[ok]
  if (object$type == "multiplicative") {
    rem <- log(object$remainder[ok])
    detrended <- log(obs / object$trend[ok])
    deseasoned <- log(obs / object$seasonal[ok])
  } else {
    rem <- object$remainder[ok]
    detrended <- obs - object$trend[ok]
    deseasoned <- obs - object$seasonal[ok]
  }
  c(
    trendStrength = max(0, 1 - stats::var(rem) / stats::var(deseasoned)),
    seasonalStrength = max(0, 1 - stats::var(rem) / stats::var(detrended))
  )
}

#' Convert a decomposition to a multivariate ts
#'
#' @param x A `geodrift_decomp` object.
#' @param ... Unused.
#' @return A `ts` matrix with columns `observed`, `trend`, `seasonal` and
#'   `remainder`.
#' @export
as.ts.geodrift_decomp <- function(x, ...) {
  parts <- cbind(
    observed = x$observed,
    trend = x$trend,
    seasonal = x$seasonal,
    remainder = x$remainder
  )
  stats::ts(parts, start = x$start, frequency = x$period)
}

#' Convert a decomposition to a zoo series
#'
#' @param x A `geodrift_decomp` object.
#' @param index Optional vector of time stamps, one per observation, such as
#'   dates. Defaults to the observation number.
#' @return A `zoo` object with one column per component.
#' @export
decompToZoo <- function(x, index = seq_along(x$observed)) {
  if (!requireNamespace("zoo", quietly = TRUE)) {
    stop("Package 'zoo' is required for decompToZoo().", call. = FALSE)
  }
  if (length(index) != length(x$observed)) {
    stop("`index` must have one entry per observation.", call. = FALSE)
  }
  parts <- unclass(stats::as.ts(x))
  attr(parts, "tsp") <- NULL
  zoo::zoo(parts, order.by = index)
}
