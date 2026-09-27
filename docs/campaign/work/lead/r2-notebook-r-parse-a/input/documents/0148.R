clean_outlier <- function(
  signal,
  loess_span = 0.25,
  threshold = 2,
  replace = c("gaussian", "uniform", "loess"),
  seed = 123
) {
  if (!is.numeric(signal)) {
    stop("\`signal\` must be numeric")
  }

  # Ensure that signal have complete cases
  signal <- signal[!is.na(signal)]
  seq_vec <- seq_along(along.with = signal)

  # Fit a local regression model to capture dynamic trends
  fit <- stats::loess(signal ~ seq_vec, span = loess_span)
  predicted <- stats::predict(fit)

  # Compute residuals and determine adaptive threshold based on MAD
  residual_values <- signal - predicted
  mad_value <- stats::mad(residual_values)
  cutoff_mad <- mad_value * threshold

  # Flag ectopic (noisy) beats based on the adaptive threshold
  ectopic_values <- abs(residual_values) > cutoff_mad

  # Set seed for reproducibility
  set.seed(seed)

  # Validate the replacement method
  replace <- match.arg(replace)

  # Replace ectopic values based on the chosen method
  if (replace == "gaussian") {
    signal[ectopic_values] <- stats::rnorm(
      n = sum(ectopic_values),
      mean = predicted[ectopic_values],
      sd = mad_value
    )
  } else if (replace == "uniform") {
    signal[ectopic_values] <- stats::runif(
      n = sum(ectopic_values),
      min = predicted[ectopic_values] - mad_value,
      max = predicted[ectopic_values] + mad_value
    )
  } else if (replace == "loess") {
    signal[ectopic_values] <- predicted[ectopic_values]
  }

  # Return the cleaned signal along with time
  return(signal)
}