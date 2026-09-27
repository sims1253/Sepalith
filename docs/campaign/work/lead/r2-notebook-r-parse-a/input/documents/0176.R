summary.FusionForest <- function(object, ...) {
  out <- list(
    meta = object$meta,
    sigma = object$sigma,
    treatment_effects = object$train_predictions_treat,
    acceptance_ratios = c(
      control   = mean(object$acceptance_ratio_control),
      treat     = mean(object$acceptance_ratio_treat),
      deconf    = mean(object$acceptance_ratio_deconf),
      deviation = if (!is.null(object$acceptance_ratio_deviation))
        mean(object$acceptance_ratio_deviation) else NA_real_
    )
  )
  class(out) <- "summary.FusionForest"
  out
}