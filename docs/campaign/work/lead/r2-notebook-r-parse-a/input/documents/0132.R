summary.BayesPET_oc <- function(object, thresholds = c(0.25,0.5,1,1.5,2), ...) {
  df <- object$replicate
  if (!is.data.frame(df) || !"difference" %in% names(df)) {
    stop("Invalid 'BayesPET_oc_predtime' object: missing 'replicate$difference'.", call. = FALSE)
  }

  d <- df$difference
  d <- d[is.finite(d)]

  pr_lt <- vapply(thresholds, function(t) mean(d < t), numeric(1))

  out <- list(
    n_valid = object$n_valid,
    n_attempt = object$n_attempt,
    success_rate = if (!is.null(object$n_attempt) && object$n_attempt > 0)
      object$n_valid / object$n_attempt else NA_real_,
    thresholds = thresholds,
    pr_lt = pr_lt,
    mae = mean(d),
    median_ae = stats::median(d),
    rmse = sqrt(mean(d^2)),
    call = object$call
  )

  class(out) <- "summary.BayesPET_oc"
  out
}