#' Simple exponential smoothing
#'
#' Filters `y` with a single smoothing weight. The level is initialised at
#' the first observation unless `level0` is given.
#'
#' @param y Numeric vector of observations in time order.
#' @param alpha Smoothing weight for the level, between 0 and 1.
#' @param level0 Optional starting level.
#' @return A list of class `drift_smooth` with `fitted`, `level`, `sse` and
#'   the smoothing parameters.
#' @family smoothing filters
#' @export
ses_filter <- function(y, alpha, level0 = NULL) {
  check_weight(alpha, "alpha")
  n <- length(y)
  level <- numeric(n)
  fitted <- numeric(n)
  prev <- if (is.null(level0)) y[1] else level0
  for (t in seq_len(n)) {
    fitted[t] <- prev
    level[t] <- alpha * y[t] + (1 - alpha) * prev
    prev <- level[t]
  }
  structure(
    list(fitted = fitted, level = level, trend = NULL, sse = sum((y - fitted)^2),
      params = c(alpha = alpha), method = "ses"),
    class = "drift_smooth"
  )
}

#' Holt linear trend smoothing
#'
#' Filters `y` with separate smoothing weights for the level and the trend.
#' The trend is initialised from the first two observations.
#'
#' @param y Numeric vector of observations in time order.
#' @param alpha Smoothing weight for the level, between 0 and 1.
#' @param beta Smoothing weight for the trend, between 0 and 1.
#' @return A list of class `drift_smooth` with `fitted`, `level`, `trend`,
#'   `sse` and the smoothing parameters.
#' @family smoothing filters
#' @export
holt_filter <- function(y, alpha, beta) {
  check_weight(alpha, "alpha")
  check_weight(beta, "beta")
  n <- length(y)
  if (n < 3) {
    stop("Holt smoothing needs at least three observations.", call. = FALSE)
  }
  level <- numeric(n)
  trend <- numeric(n)
  fitted <- numeric(n)
  prev_level <- y[1]
  prev_trend <- y[2] - y[1]
  for (t in seq_len(n)) {
    fitted[t] <- prev_level + prev_trend
    level[t] <- alpha * y[t] + (1 - alpha) * (prev_level + prev_trend)
    trend[t] <- beta * (level[t] - prev_level) + (1 - beta) * prev_trend
    prev_level <- level[t]
    prev_trend <- trend[t]
  }
  structure(
    list(fitted = fitted, level = level, trend = trend, sse = sum((y - fitted)^2),
      params = c(alpha = alpha, beta = beta), method = "holt"),
    class = "drift_smooth"
  )
}

#' Fit smoothing weights by minimising squared one-step errors
#'
#' Searches a regular grid of weights, which is slow but robust for the
#' short monthly series this package targets.
#'
#' @param y Numeric vector of observations in time order.
#' @param method Either `"ses"` or `"holt"`.
#' @param grid Candidate weights to try for each parameter.
#' @return The best fitting `drift_smooth` object.
#' @export
fit_smoother <- function(y, method = c("ses", "holt"), grid = seq(0.05, 0.95, by = 0.05)) {
  method <- match.arg(method)
  best <- NULL
  if (method == "ses") {
    for (a in grid) {
      fit <- ses_filter(y, a)
      if (is.null(best) || fit$sse < best$sse) best <- fit
    }
  } else {
    for (a in grid) {
      for (b in grid) {
        fit <- holt_filter(y, a, b)
        if (is.null(best) || fit$sse < best$sse) best <- fit
      }
    }
  }
  best
}

#' Forecast from a fitted smoother
#'
#' @param object A `drift_smooth` object.
#' @param h Number of periods ahead to forecast.
#' @param ... Ignored.
#' @return A numeric vector of length `h`.
#' @export
predict.drift_smooth <- function(object, h = 1, ...) {
  n <- length(object$level)
  last_level <- object$level[n]
  if (identical(object$method, "holt")) {
    last_level + seq_len(h) * object$trend[n]
  } else {
    rep(last_level, h)
  }
}

#' Print a fitted smoother
#'
#' @param x A `drift_smooth` object.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.drift_smooth <- function(x, ...) {
  label <- switch(x$method, ses = "Simple exponential smoothing", holt = "Holt linear trend")
  cat(label, "\n")
  cat("  Parameters:", paste(names(x$params), format(x$params, digits = 3), sep = " = ",
    collapse = ", "), "\n")
  cat("  SSE:", format(x$sse, digits = 5), "\n")
  invisible(x)
}

#' Rolling-origin forecast accuracy
#'
#' Refits the smoother on expanding windows and records the `h`-step-ahead
#' error at each origin.
#'
#' @param y Numeric vector of observations in time order.
#' @param method Either `"ses"` or `"holt"`.
#' @param h Number of periods ahead to forecast.
#' @param min_train Size of the first training window.
#' @return A named numeric vector with `mae`, `rmse` and `mape`.
#' @export
rolling_accuracy <- function(y, method = c("ses", "holt"), h = 1, min_train = 12) {
  method <- match.arg(method)
  if (min_train > length(y) - h) {
    stop("Series too short for the requested `min_train` and `h`.", call. = FALSE)
  }
  origins <- seq(min_train, length(y) - h)
  errors <- numeric(length(origins))
  for (i in seq_along(origins)) {
    train <- y[seq_len(origins[i])]
    fit <- fit_smoother(train, method)
    errors[i] <- y[origins[i] + h] - predict(fit, h)[h]
  }
  actual <- y[origins + h]
  c(
    mae = mean(abs(errors)),
    rmse = sqrt(mean(errors^2)),
    mape = mean(abs(errors / actual)) * 100
  )
}

#' Validate a smoothing weight
#'
#' @param w Candidate weight.
#' @param name Argument name used in the error message.
#' @return `w`, invisibly.
#' @noRd
check_weight <- function(w, name) {
  if (!is.numeric(w) || length(w) != 1 || is.na(w) || w < 0 || w > 1) {
    stop(sprintf("`%s` must be a single number between 0 and 1.", name), call. = FALSE)
  }
  invisible(w)
}
