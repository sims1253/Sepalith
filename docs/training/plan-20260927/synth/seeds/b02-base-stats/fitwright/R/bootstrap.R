#' Construct a bootstrap result object
#'
#' Low-level constructor for objects of class `fw_boot`.
#'
#' @param t0 Statistic evaluated on the original data.
#' @param t Matrix of bootstrap replicates, one row per resample.
#' @param call The matched call that produced the result.
#' @return An object of class `fw_boot`.
#' @keywords internal
new_fw_boot <- function(t0, t, call) {
  stopifnot(is.numeric(t0), is.matrix(t), ncol(t) == length(t0))
  colnames(t) <- names(t0)
  structure(
    list(t0 = t0, t = t, R = nrow(t), call = call),
    class = "fw_boot"
  )
}

#' Draw bootstrap resample indices
#'
#' Generates a matrix of row indices for ordinary or stratified resampling.
#'
#' @param n Number of observations.
#' @param R Number of bootstrap resamples.
#' @param strata Optional vector of stratum labels of length `n`.
#' @return An integer matrix with `R` rows and `n` columns.
#' @keywords internal
boot_indices <- function(n, R, strata = NULL) {
  if (is.null(strata)) {
    return(matrix(sample.int(n, n * R, replace = TRUE), nrow = R))
  }
  stopifnot(length(strata) == n)
  idx <- matrix(0L, nrow = R, ncol = n)
  for (grp in split(seq_len(n), strata)) {
    draws <- grp[sample.int(length(grp), length(grp) * R, replace = TRUE)]
    idx[, grp] <- matrix(draws, nrow = R)
  }
  idx
}

#' Bootstrap a statistic
#'
#' Resamples rows of `data` and evaluates `statistic` on every resample.
#'
#' @param data A data frame or numeric vector to resample.
#' @param statistic A function taking the resampled data and returning a
#'   named numeric vector.
#' @param R Number of bootstrap resamples.
#' @param strata Optional vector of stratum labels, one per observation.
#' @return An object of class `fw_boot`.
#' @export
boot_stat <- function(data, statistic, R = 999L, strata = NULL) {
  statistic <- match.fun(statistic)
  is_df <- is.data.frame(data)
  n <- if (is_df) nrow(data) else length(data)
  t0 <- statistic(data)
  if (is.null(names(t0))) {
    names(t0) <- paste0("t", seq_along(t0))
  }
  idx <- boot_indices(n, R, strata)
  t <- vapply(seq_len(R), function(r) {
    rows <- idx[r, ]
    unname(statistic(if (is_df) data[rows, , drop = FALSE] else data[rows]))
  }, numeric(length(t0)))
  t <- if (length(t0) == 1L) matrix(t, ncol = 1L) else t(t)
  new_fw_boot(t0, t, match.call())
}

#' Percentile bootstrap interval
#'
#' @param boot An object of class `fw_boot`.
#' @param level Confidence level of the interval.
#' @return A matrix with columns `lower` and `upper`, one row per statistic.
#' @family bootstrap intervals
#' @export
boot_ci_percentile <- function(boot, level = 0.95) {
  alpha <- (1 - level) / 2
  q <- apply(boot$t, 2, stats::quantile, probs = c(alpha, 1 - alpha), na.rm = TRUE)
  out <- t(q)
  colnames(out) <- c("lower", "upper")
  out
}

#' Basic bootstrap interval
#'
#' @param boot An object of class `fw_boot`.
#' @param level Confidence level of the interval.
#' @return A matrix with columns `lower` and `upper`, one row per statistic.
#' @family bootstrap intervals
#' @export
boot_ci_basic <- function(boot, level = 0.95) {
  alpha <- (1 - level) / 2
  q <- apply(boot$t, 2, stats::quantile, probs = c(1 - alpha, alpha), na.rm = TRUE)
  out <- 2 * boot$t0 - t(q)
  colnames(out) <- c("lower", "upper")
  out
}

#' Normal-approximation bootstrap interval
#'
#' @param boot An object of class `fw_boot`.
#' @param level Confidence level of the interval.
#' @return A matrix with columns `lower` and `upper`, one row per statistic.
#' @family bootstrap intervals
#' @export
boot_ci_normal <- function(boot, level = 0.95) {
  z <- stats::qnorm(1 - (1 - level) / 2)
  bias <- colMeans(boot$t, na.rm = TRUE) - boot$t0
  se <- apply(boot$t, 2, stats::sd, na.rm = TRUE)
  centre <- boot$t0 - bias
  out <- cbind(lower = centre - z * se, upper = centre + z * se)
  rownames(out) <- names(boot$t0)
  out
}

#' Bootstrap confidence intervals
#'
#' Dispatches to one of the interval helpers by name.
#'
#' @param boot An object of class `fw_boot`.
#' @param type Interval type.
#' @param level Confidence level of the interval.
#' @return A matrix with columns `lower` and `upper`, one row per statistic.
#' @family bootstrap intervals
#' @export
boot_ci <- function(boot, type = c("percentile", "basic", "normal"), level = 0.95) {
  if (!inherits(boot, "fw_boot")) {
    stop("`boot` must be an `fw_boot` object.", call. = FALSE)
  }
  type <- match.arg(type)
  switch(type,
    percentile = boot_ci_percentile(boot, level),
    basic = boot_ci_basic(boot, level),
    normal = boot_ci_normal(boot, level)
  )
}

#' Case-resampling bootstrap of regression coefficients
#'
#' Refits a linear model on resampled rows and collects the coefficients.
#' Resamples where the fit is rank deficient are recorded as `NA`.
#'
#' @param formula A model formula.
#' @param data A data frame containing the model variables.
#' @param R Number of bootstrap resamples.
#' @return An object of class `fw_boot`.
#' @export
boot_coefs <- function(formula, data, R = 999L) {
  fit0 <- stats::lm(formula, data = data)
  t0 <- stats::coef(fit0)
  idx <- boot_indices(nrow(data), R)
  t <- matrix(NA_real_, nrow = R, ncol = length(t0))
  for (r in seq_len(R)) {
    fit_r <- tryCatch(
      stats::lm(formula, data = data[idx[r, ], , drop = FALSE]),
      error = function(e) NULL
    )
    if (!is.null(fit_r)) {
      t[r, ] <- stats::coef(fit_r)
    }
  }
  new_fw_boot(t0, t, match.call())
}
