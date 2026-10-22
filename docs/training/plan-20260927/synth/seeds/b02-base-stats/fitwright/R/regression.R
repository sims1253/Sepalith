#' Fit a linear model and capture failures
#'
#' Wraps [stats::lm()] so that rank deficiency and fitting errors are
#' reported as conditions attached to the result instead of aborting a loop.
#'
#' @param formula A model formula.
#' @param data A data frame containing the model variables.
#' @param weights Optional numeric vector of case weights.
#' @return A list with elements `fit` (an `lm` object or `NULL`), `ok` and
#'   `message`.
#' @export
fit_lm_safely <- function(formula, data, weights = NULL) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  if (!is.null(weights) && length(weights) != nrow(data)) {
    stop("`weights` must have one entry per row of `data`.", call. = FALSE)
  }
  data$.fw_weights <- if (is.null(weights)) rep(1, nrow(data)) else weights
  fit <- tryCatch(
    stats::lm(formula, data = data, weights = .fw_weights),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(fit = NULL, ok = FALSE, message = conditionMessage(fit)))
  }
  aliased <- names(which(is.na(stats::coef(fit))))
  if (length(aliased) > 0L) {
    msg <- paste0("aliased terms: ", paste(aliased, collapse = ", "))
    return(list(fit = fit, ok = FALSE, message = msg))
  }
  list(fit = fit, ok = TRUE, message = NA_character_)
}

#' Coefficient table for a linear model
#'
#' Builds a data frame of estimates, standard errors, t statistics, p-values
#' and confidence limits for an `lm` fit.
#'
#' @param fit A fitted model object.
#' @param level Confidence level for the interval columns.
#' @return A data frame with one row per coefficient.
#' @family coefficient tables
#' @export
coef_table_lm <- function(fit, level = 0.95) {
  stopifnot(inherits(fit, "lm"), level > 0, level < 1)
  est <- stats::coef(fit)
  se <- sqrt(diag(stats::vcov(fit)))
  df_res <- stats::df.residual(fit)
  stat <- est / se
  crit <- stats::qt(1 - (1 - level) / 2, df = df_res)
  data.frame(
    term = names(est),
    estimate = unname(est),
    std_error = unname(se),
    statistic = unname(stat),
    p_value = 2 * stats::pt(abs(stat), df = df_res, lower.tail = FALSE),
    lower = unname(est - crit * se),
    upper = unname(est + crit * se),
    row.names = NULL
  )
}

#' Coefficient table for a generalised linear model
#'
#' Builds a data frame of estimates, standard errors, z statistics, p-values
#' and Wald confidence limits for a `glm` fit.
#'
#' @param fit A fitted model object.
#' @param level Confidence level for the interval columns.
#' @param exponentiate Should estimates and limits be exponentiated?
#' @return A data frame with one row per coefficient.
#' @family coefficient tables
#' @export
coef_table_glm <- function(fit, level = 0.95, exponentiate = FALSE) {
  stopifnot(inherits(fit, "glm"), level > 0, level < 1)
  est <- stats::coef(fit)
  se <- sqrt(diag(stats::vcov(fit)))
  stat <- est / se
  crit <- stats::qnorm(1 - (1 - level) / 2)
  out <- data.frame(
    term = names(est),
    estimate = unname(est),
    std_error = unname(se),
    statistic = unname(stat),
    p_value = 2 * stats::pnorm(abs(stat), lower.tail = FALSE),
    lower = unname(est - crit * se),
    upper = unname(est + crit * se),
    row.names = NULL
  )
  if (exponentiate) {
    out[c("estimate", "lower", "upper")] <- lapply(out[c("estimate", "lower", "upper")], exp)
  }
  out
}

#' Extract residuals of a chosen type
#'
#' Returns raw, standardised or studentised residuals for a linear model,
#' computed from the hat values rather than delegated to helpers.
#'
#' @param fit A fitted model object.
#' @param type Type of residual to return.
#' @return A numeric vector of residuals.
#' @export
model_residuals <- function(fit, type = c("raw", "standardized", "studentized")) {
  type <- match.arg(type)
  e <- stats::residuals(fit)
  h <- stats::hatvalues(fit)
  n <- length(e)
  p <- length(stats::coef(fit))
  s <- sqrt(sum(e^2) / (n - p))
  switch(type,
    raw = e,
    standardized = e / (s * sqrt(1 - h)),
    studentized = {
      s_i <- sqrt((sum(e^2) - e^2 / (1 - h)) / (n - p - 1))
      e / (s_i * sqrt(1 - h))
    }
  )
}

#' Flag high-leverage observations
#'
#' @param fit A fitted model object.
#' @param threshold Cut-off above which a point is flagged. Defaults to
#'   twice the mean hat value.
#' @return A logical vector, one entry per observation.
#' @family influence flags
#' @export
leverage_flags <- function(fit, threshold = NULL) {
  h <- stats::hatvalues(fit)
  if (is.null(threshold)) {
    threshold <- 2 * mean(h)
  }
  h > threshold
}

#' Flag influential observations by Cook's distance
#'
#' @param fit A fitted model object.
#' @param threshold Cut-off above which a point is flagged. Defaults to
#'   `4 / n`.
#' @return A logical vector, one entry per observation.
#' @family influence flags
#' @export
cooks_flags <- function(fit, threshold = NULL) {
  d <- stats::cooks.distance(fit)
  if (is.null(threshold)) {
    threshold <- 4 / length(d)
  }
  d > threshold
}

#' Flag observations with large scaled DFFITS
#'
#' @param fit A fitted model object.
#' @param threshold Cut-off above which a point is flagged. Defaults to
#'   `2 * sqrt(p / n)`.
#' @return A logical vector, one entry per observation.
#' @family influence flags
#' @export
dffits_flags <- function(fit, threshold = NULL) {
  d <- abs(stats::dffits(fit))
  if (is.null(threshold)) {
    threshold <- 2 * sqrt(length(stats::coef(fit)) / length(d))
  }
  d > threshold
}

#' Compare two nested models
#'
#' Computes the change in residual deviance between a smaller and a larger
#' model and the matching F or chi-squared test.
#'
#' @param small,large Fitted models, with `small` nested in `large`.
#' @param test Reference distribution for the test statistic.
#' @return A one-row data frame with the statistic, degrees of freedom and
#'   p-value.
#' @export
compare_nested <- function(small, large, test = c("F", "Chisq")) {
  test <- match.arg(test)
  dev_small <- stats::deviance(small)
  dev_large <- stats::deviance(large)
  df_diff <- stats::df.residual(small) - stats::df.residual(large)
  if (df_diff <= 0) {
    stop("`small` must have more residual degrees of freedom than `large`.", call. = FALSE)
  }
  df_res <- stats::df.residual(large)
  if (test == "F") {
    stat <- ((dev_small - dev_large) / df_diff) / (dev_large / df_res)
    p <- stats::pf(stat, df_diff, df_res, lower.tail = FALSE)
  } else {
    stat <- dev_small - dev_large
    p <- stats::pchisq(stat, df_diff, lower.tail = FALSE)
  }
  data.frame(test = test, statistic = stat, df = df_diff, p_value = p)
}

#' K-fold cross-validated prediction error for a linear model
#'
#' Splits the rows into `k` folds, refits on each training set and returns
#' the out-of-fold squared and absolute errors.
#'
#' @param formula A model formula.
#' @param data A data frame containing the model variables.
#' @param k Number of folds.
#' @param seed Optional integer seed used to assign folds.
#' @return A list with the per-fold errors in `folds`, the pooled `rmse`
#'   and `mae`, and the out-of-fold `predictions`.
#' @export
cv_lm <- function(formula, data, k = 10L, seed = NULL) {
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  n <- nrow(data)
  if (k < 2L || k > n) {
    stop("`k` must lie between 2 and the number of rows.", call. = FALSE)
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }
  fold_id <- sample(rep_len(seq_len(k), n))
  response <- all.vars(formula)[1L]
  pred <- rep(NA_real_, n)
  fold_rmse <- numeric(k)
  fold_mae <- numeric(k)
  for (f in seq_len(k)) {
    test <- fold_id == f
    fit <- stats::lm(formula, data = data[!test, , drop = FALSE])
    pred[test] <- stats::predict(fit, newdata = data[test, , drop = FALSE])
    err <- data[[response]][test] - pred[test]
    fold_rmse[f] <- sqrt(mean(err^2, na.rm = TRUE))
    fold_mae[f] <- mean(abs(err), na.rm = TRUE)
  }
  err_all <- data[[response]] - pred
  list(
    folds = data.frame(fold = seq_len(k), rmse = fold_rmse, mae = fold_mae),
    rmse = sqrt(mean(err_all^2, na.rm = TRUE)),
    mae = mean(abs(err_all), na.rm = TRUE),
    predictions = pred
  )
}
