#' Check paired truth and estimate vectors
#'
#' Verifies that `truth` and `estimate` are numeric, of equal length, and
#' drops pairs where either value is missing when `na_rm` is `TRUE`.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A list with elements `truth` and `estimate`.
#' @keywords internal
align_pairs <- function(truth, estimate, na_rm = TRUE) {
  if (!is.numeric(truth) || !is.numeric(estimate)) {
    stop("`truth` and `estimate` must both be numeric.", call. = FALSE)
  }
  if (length(truth) != length(estimate)) {
    stop("`truth` and `estimate` must have the same length.", call. = FALSE)
  }
  keep <- if (na_rm) stats::complete.cases(truth, estimate) else rep(TRUE, length(truth))
  list(truth = truth[keep], estimate = estimate[keep])
}

#' Root mean squared error
#'
#' Computes the square root of the mean squared difference between observed
#' and predicted values.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
#' @examples
#' rmse_vec(c(1, 2, 3), c(1.1, 1.9, 3.2))
rmse_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  sqrt(mean((p$truth - p$estimate)^2))
}

#' Mean absolute error
#'
#' Computes the mean absolute difference between observed and predicted
#' values.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
mae_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  mean(abs(p$truth - p$estimate))
}

#' Mean absolute percentage error
#'
#' Computes the mean absolute error relative to the observed values, as a
#' percentage. Pairs with a zero observed value are skipped.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
mape_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  nonzero <- p$truth != 0
  if (!any(nonzero)) {
    return(NA_real_)
  }
  100 * mean(abs((p$truth[nonzero] - p$estimate[nonzero]) / p$truth[nonzero]))
}

#' Coefficient of determination
#'
#' Computes one minus the ratio of residual to total sum of squares.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family regression metrics
#' @export
rsq_trad_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(truth, estimate, na_rm)
  ss_res <- sum((p$truth - p$estimate)^2)
  ss_tot <- sum((p$truth - mean(p$truth))^2)
  if (ss_tot == 0) {
    return(NA_real_)
  }
  1 - ss_res / ss_tot
}

#' Binary log loss
#'
#' Computes the mean negative log-likelihood of 0/1 outcomes under the given
#' event probabilities, clipping probabilities away from 0 and 1.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted event probabilities.
#' @param eps Clipping tolerance applied to `estimate`.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family classification metrics
#' @export
log_loss_vec <- function(truth, estimate, eps = 1e-15, na_rm = TRUE) {
  p <- align_pairs(as.numeric(truth), estimate, na_rm)
  if (!all(p$truth %in% c(0, 1))) {
    stop("`truth` must be coded as 0/1.", call. = FALSE)
  }
  prob <- pmin(pmax(p$estimate, eps), 1 - eps)
  -mean(p$truth * log(prob) + (1 - p$truth) * log(1 - prob))
}

#' Brier score
#'
#' Computes the mean squared difference between 0/1 outcomes and the given
#' event probabilities.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted event probabilities.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value.
#' @family classification metrics
#' @export
brier_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(as.numeric(truth), estimate, na_rm)
  if (!all(p$truth %in% c(0, 1))) {
    stop("`truth` must be coded as 0/1.", call. = FALSE)
  }
  mean((p$truth - p$estimate)^2)
}

#' Area under the ROC curve by rank
#'
#' Computes the AUC using the Mann-Whitney rank-sum formulation, which handles
#' tied scores by averaging their ranks.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted event probabilities.
#' @param na_rm Logical; drop incomplete pairs before computing.
#' @return A single numeric value between 0 and 1.
#' @family classification metrics
#' @export
auc_vec <- function(truth, estimate, na_rm = TRUE) {
  p <- align_pairs(as.numeric(truth), estimate, na_rm)
  n_pos <- sum(p$truth == 1)
  n_neg <- sum(p$truth == 0)
  if (n_pos == 0 || n_neg == 0) {
    warning("AUC is undefined with only one class present.", call. = FALSE)
    return(NA_real_)
  }
  ranks <- rank(p$estimate, ties.method = "average")
  rank_sum <- sum(ranks[p$truth == 1])
  (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

#' Compute several metrics at once
#'
#' Evaluates a named set of metric functions on the same truth/estimate pair
#' and returns a tidy one-row-per-metric data frame.
#'
#' @param truth A numeric vector of observed values.
#' @param estimate A numeric vector of predicted values.
#' @param metrics A named list of metric functions taking `truth` and `estimate`.
#' @return A data frame with columns `metric` and `value`.
#' @export
metric_table <- function(truth, estimate,
                         metrics = list(rmse = rmse_vec, mae = mae_vec, rsq = rsq_trad_vec)) {
  if (is.null(names(metrics)) || any(names(metrics) == "")) {
    stop("`metrics` must be a fully named list.", call. = FALSE)
  }
  values <- vapply(metrics, function(f) f(truth, estimate), numeric(1))
  data.frame(metric = names(metrics), value = unname(values), stringsAsFactors = FALSE)
}
