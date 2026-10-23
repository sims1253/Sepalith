#' Validate a weight vector
#'
#' @param w A numeric vector of non-negative weights.
#' @param n The required length.
#' @keywords internal
#' @noRd
check_weights <- function(w, n) {
  if (!is.numeric(w) || length(w) != n) {
    stop(sprintf("`w` must be a numeric vector of length %d.", n), call. = FALSE)
  }
  if (anyNA(w) || any(w < 0)) {
    stop("`w` must be non-negative and free of missing values.", call. = FALSE)
  }
  if (sum(w) <= 0) {
    stop("`w` must have a positive sum.", call. = FALSE)
  }
  invisible(w)
}

#' Weighted mean with missing-value handling
#'
#' @param x A numeric vector.
#' @param w A numeric vector of non-negative weights, the same length as `x`.
#' @param na_rm Logical; drop missing values of `x` together with their
#'   weights?
#' @export
weighted_mean <- function(x, w, na_rm = TRUE) {
  check_weights(w, length(x))
  if (!na_rm && anyNA(x)) {
    return(NA_real_)
  }
  keep <- !is.na(x)
  sum(w[keep] * x[keep]) / sum(w[keep])
}

#' Weighted variance
#'
#' @description
#' Frequency weights count repeated observations, so the denominator is
#' `sum(w) - 1`. Reliability weights are relative, so the denominator uses
#' `sum(w) - sum(w^2) / sum(w)`.
#'
#' @param x A numeric vector.
#' @param w A numeric vector of non-negative weights, the same length as `x`.
#' @param type Either `"frequency"` or `"reliability"`.
#' @export
weighted_var <- function(x, w, type = c("frequency", "reliability")) {
  type <- match.arg(type)
  check_weights(w, length(x))
  keep <- !is.na(x)
  x <- x[keep]
  w <- w[keep]
  center <- sum(w * x) / sum(w)
  ss <- sum(w * (x - center)^2)

  if (type == "frequency") {
    denom <- sum(w) - 1
  } else {
    denom <- sum(w) - sum(w^2) / sum(w)
  }
  if (denom <= 0) {
    return(NA_real_)
  }
  ss / denom
}

#' Weighted quantiles
#'
#' @description
#' Uses linear interpolation of the cumulative weight at the midpoint of
#' each observation.
#'
#' @param x A numeric vector.
#' @param w A numeric vector of non-negative weights, the same length as `x`.
#' @param probs Numeric vector of probabilities in `[0, 1]`.
#' @return A numeric vector named like `"50%"`.
#' @export
weighted_quantile <- function(x, w, probs = c(0.25, 0.5, 0.75)) {
  check_weights(w, length(x))
  if (any(probs < 0 | probs > 1)) {
    stop("`probs` must lie in [0, 1].", call. = FALSE)
  }

  # Interpolate the cumulative weight at the midpoint of each observation
  keep <- !is.na(x) & w > 0
  ord <- order(x[keep])
  x <- x[keep][ord]
  w <- w[keep][ord]
  cum <- (cumsum(w) - w / 2) / sum(w)
  out <- stats::approx(cum, x, xout = probs, rule = 2, ties = "ordered")$y
  names(out) <- paste0(format(100 * probs, trim = TRUE), "%")
  out
}

#' Weighted summaries of several columns
#'
#' @param data A data frame.
#' @param value_cols Character vector of numeric column names to summarise.
#' @param weight_col Name of the weight column.
#' @return A data frame with one row per entry of `value_cols` and columns
#'   `column`, `n_eff`, `mean`, `sd`, `median` and `iqr`.
#' @details `n_eff` is the Kish effective sample size
#'   `sum(w)^2 / sum(w^2)` over the non-missing rows of each column.
#' @export
weighted_summary <- function(data, value_cols, weight_col) {
  w_all <- data[[weight_col]]
  check_weights(w_all, nrow(data))
  n <- length(value_cols)
  n_eff <- mean <- sd <- median <- iqr <- rep(NA_real_, n)

  for (i in seq_len(n)) {
    x <- data[[value_cols[i]]]
    keep <- !is.na(x)
    if (!any(keep)) {
      next
    }
    w <- w_all[keep]
    x <- x[keep]
    # Effective sample size
    n_eff[i] <- sum(w)^2 / sum(w^2)
    # Location
    mean[i] <- weighted_mean(x, w)
    median[i] <- weighted_quantile(x, w, probs = 0.5)
    # Spread
    sd[i] <- sqrt(weighted_var(x, w, type = "reliability"))
    quartiles <- weighted_quantile(x, w, probs = c(0.25, 0.75))
    iqr[i] <- quartiles[[2]] - quartiles[[1]]
  }

  data.frame(
    column = value_cols,
    n_eff = n_eff,
    mean = mean,
    sd = sd,
    median = median,
    iqr = iqr,
    stringsAsFactors = FALSE
  )
}

#' Central moments of several columns
#'
#' @description
#' Skewness and kurtosis come with their large-sample standard errors under
#' normality and the matching z scores.
#'
#' @param data A data frame.
#' @param cols Character vector of numeric column names.
#' @param excess Logical; subtract 3 from the kurtosis?
#' @return A matrix with one row per column and the columns `n`, `mean`,
#'   `variance`, `skewness`, `skewness_se`, `skewness_z`, `kurtosis`,
#'   `kurtosis_se` and `kurtosis_z`.
#' @export
column_moments <- function(data, cols = names(data), excess = TRUE) {
  stats_names <- c("n", "mean", "variance", "skewness", "skewness_se", "skewness_z",
                   "kurtosis", "kurtosis_se", "kurtosis_z")
  out <- matrix(NA_real_, nrow = length(cols), ncol = length(stats_names),
                dimnames = list(cols, stats_names))

  for (col in cols) {
    x <- data[[col]]
    x <- x[!is.na(x)]
    n <- length(x)
    out[col, "n"] <- n
    # The kurtosis standard error needs at least four values
    if (n < 4L) {
      next
    }
    dev <- x - mean(x)
    m2 <- mean(dev^2)
    out[col, "mean"] <- mean(x)
    out[col, "variance"] <- m2 * n / (n - 1)

    # Skewness and its standard error
    skew_se <- sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    out[col, "skewness"] <- mean(dev^3) / m2^1.5
    out[col, "skewness_se"] <- skew_se
    out[col, "skewness_z"] <- out[col, "skewness"] / skew_se
    # Kurtosis and its standard error
    kurt_se <- 2 * skew_se * sqrt((n^2 - 1) / ((n - 3) * (n + 5)))
    out[col, "kurtosis"] <- mean(dev^4) / m2^2 - if (excess) 3 else 0
    out[col, "kurtosis_se"] <- kurt_se
    out[col, "kurtosis_z"] <- (mean(dev^4) / m2^2 - 3) / kurt_se
  }
  out
}
