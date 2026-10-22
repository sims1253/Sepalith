#' Predict without surprises
#'
#' Wraps [stats::predict()] so that errors and warnings carry a consistent
#' prefix and predictions are always returned as a plain numeric vector.
#'
#' @param model A fitted model with a `predict()` method accepting `newdata`.
#' @param newdata A data frame to predict for.
#' @param type Prediction scale passed on to `predict()`.
#' @return A numeric vector with one value per row of `newdata`.
#' @keywords internal
predict_safely <- function(model, newdata, type = c("response", "link")) {
  type <- match.arg(type)
  pred <- tryCatch(
    stats::predict(model, newdata = newdata, type = type),
    error = function(e) {
      stop(sprintf("Prediction failed: %s", conditionMessage(e)), call. = FALSE)
    },
    warning = function(w) {
      warning(sprintf("Prediction warning: %s", conditionMessage(w)), call. = FALSE)
      suppressWarnings(stats::predict(model, newdata = newdata, type = type))
    }
  )
  as.numeric(pred)
}

#' Choose a finite-difference step
#'
#' Scales a small step to the magnitude of `x` so the difference quotient is
#' neither swamped by rounding nor too coarse.
#'
#' @param x A numeric vector of predictor values.
#' @param eps Optional fixed step; overrides the automatic choice.
#' @return A single positive number.
#' @keywords internal
fd_step <- function(x, eps = NULL) {
  if (!is.null(eps)) {
    if (!is.numeric(eps) || length(eps) != 1L || eps <= 0) {
      stop("`eps` must be a single positive number.", call. = FALSE)
    }
    return(eps)
  }
  scale <- max(abs(x), na.rm = TRUE)
  if (!is.finite(scale) || scale == 0) {
    scale <- 1
  }
  1e-5 * scale
}

#' Unit-level slopes by central differences
#'
#' Shifts a numeric predictor half a step down and up for every row and
#' divides the change in prediction by the step.
#'
#' @param model A fitted model with a `predict()` method accepting `newdata`.
#' @param data A data frame of observations to evaluate the slope at.
#' @param variable Name of a numeric predictor column in `data`.
#' @param eps Optional fixed step; overrides the automatic choice.
#' @param type Prediction scale passed on to `predict()`.
#' @return A numeric vector of slopes, one per row of `data`.
#' @family marginal effects
#' @export
slopes_fd <- function(model, data, variable, eps = NULL,
                      type = c("response", "link")) {
  type <- match.arg(type)
  if (!variable %in% names(data)) {
    stop(sprintf("Column `%s` not found in `data`.", variable), call. = FALSE)
  }
  x <- data[[variable]]
  if (!is.numeric(x)) {
    stop("`variable` must be numeric; use contrast_levels() for factors.", call. = FALSE)
  }
  h <- fd_step(x, eps)
  lo <- data
  hi <- data
  lo[[variable]] <- x - h / 2
  hi[[variable]] <- x + h / 2
  pred_lo <- predict_safely(model, lo, type = type)
  pred_hi <- predict_safely(model, hi, type = type)
  (pred_hi - pred_lo) / h
}

#' Average marginal slope
#'
#' Averages the unit-level finite-difference slopes over the observed data.
#'
#' @inheritParams slopes_fd
#' @return A one-row data frame with the average slope, its spread and the
#'   number of rows used.
#' @family marginal effects
#' @export
average_slope <- function(model, data, variable, eps = NULL,
                          type = c("response", "link")) {
  type <- match.arg(type)
  dydx <- slopes_fd(model, data, variable, eps = eps, type = type)
  ok <- !is.na(dydx)
  data.frame(
    term = variable,
    estimate = mean(dydx[ok]),
    std.dev = stats::sd(dydx[ok]),
    n = sum(ok),
    type = type,
    stringsAsFactors = FALSE
  )
}

#' Average slopes at fixed predictor values
#'
#' Sets `variable` to each value in `at` for every row, then averages the
#' slopes over the remaining covariates.
#'
#' @inheritParams slopes_fd
#' @param at A numeric vector of values for `variable`.
#' @return A data frame with one row per value of `at`.
#' @family marginal effects
#' @export
slopes_at <- function(model, data, variable, at, eps = NULL,
                      type = c("response", "link")) {
  type <- match.arg(type)
  if (!is.numeric(at) || length(at) == 0L) {
    stop("`at` must be a non-empty numeric vector.", call. = FALSE)
  }
  estimate <- numeric(length(at))
  for (i in seq_along(at)) {
    grid <- data
    grid[[variable]] <- at[i]
    dydx <- slopes_fd(model, grid, variable, eps = eps, type = type)
    estimate[i] <- mean(dydx, na.rm = TRUE)
  }
  data.frame(
    term = variable,
    at = at,
    estimate = estimate,
    stringsAsFactors = FALSE
  )
}

#' Average slopes within groups
#'
#' Computes unit-level slopes once and averages them within the levels of a
#' grouping column.
#'
#' @inheritParams slopes_fd
#' @param by Name of a grouping column in `data`.
#' @return A data frame with one row per group.
#' @family marginal effects
#' @export
slopes_by <- function(model, data, variable, by, eps = NULL,
                      type = c("response", "link")) {
  type <- match.arg(type)
  if (!by %in% names(data)) {
    stop(sprintf("Column `%s` not found in `data`.", by), call. = FALSE)
  }
  dydx <- slopes_fd(model, data, variable, eps = eps, type = type)
  groups <- split(seq_len(nrow(data)), data[[by]], drop = TRUE)
  out <- data.frame(
    group = names(groups),
    estimate = NA_real_,
    std.dev = NA_real_,
    n = NA_integer_,
    stringsAsFactors = FALSE
  )
  for (i in seq_along(groups)) {
    idx <- groups[[i]]
    ok <- idx[!is.na(dydx[idx])]
    if (length(ok) == 0L) {
      out$n[i] <- 0L
      next
    }
    out$estimate[i] <- mean(dydx[ok])
    out$std.dev[i] <- if (length(ok) > 1L) stats::sd(dydx[ok]) else NA_real_
    out$n[i] <- length(ok)
  }
  names(out)[1L] <- by
  out
}

#' Average contrasts between factor levels
#'
#' Sets every row to the reference level and to each other level in turn and
#' averages the difference in predictions.
#'
#' @inheritParams slopes_fd
#' @param variable Name of a factor or character column in `data`.
#' @param ref Reference level; defaults to the first level.
#' @return A data frame with one row per non-reference level.
#' @family marginal effects
#' @export
contrast_levels <- function(model, data, variable, ref = NULL,
                            type = c("response", "link")) {
  type <- match.arg(type)
  x <- data[[variable]]
  if (is.null(x)) {
    stop(sprintf("Column `%s` not found in `data`.", variable), call. = FALSE)
  }
  lev <- if (is.factor(x)) levels(x) else sort(unique(as.character(x)))
  if (is.null(ref)) {
    ref <- lev[1L]
  }
  if (!ref %in% lev) {
    stop(sprintf("`ref` must be one of: %s.", paste(lev, collapse = ", ")), call. = FALSE)
  }
  counterfactual <- function(value) {
    cf <- data
    values <- rep(value, nrow(data))
    cf[[variable]] <- if (is.factor(x)) factor(values, levels = lev) else values
    cf
  }
  others <- setdiff(lev, ref)
  pred_ref <- counterfactual(ref) |> predict_safely(model = model, type = type)
  estimate <- numeric(length(others))
  for (i in seq_along(others)) {
    pred <- counterfactual(others[i]) |> predict_safely(model = model, type = type)
    estimate[i] <- mean(pred - pred_ref, na.rm = TRUE)
  }
  data.frame(
    term = variable,
    contrast = paste(others, "-", ref),
    estimate = estimate,
    stringsAsFactors = FALSE
  )
}
