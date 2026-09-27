compute_dynamic_bewrs <- function(
  fit,
  id_cols = c('provider', 'pathway'),
  time_col = 'time',
  window = 3,
  weights = c(current = 1.0, persistence = 0.8, deterioration = 0.6)
) {
  data <- fit$data
  check_required(data, c(id_cols, time_col))
  data$posterior_up <- fit$posterior_up
  data <- data[do.call(order, data[c(id_cols, time_col)]), , drop = FALSE]
  id <- interaction(data[id_cols], drop = TRUE)
  persistence <- deterioration <- rep(NA_real_, nrow(data))
  for (lev in levels(id)) {
    idx <- which(id == lev)
    r <- data$posterior_up[idx]
    for (k in seq_along(idx)) {
      lo <- max(1, k - window + 1)
      persistence[idx[k]] <- mean(r[lo:k] > 0.5, na.rm = TRUE)
      deterioration[idx[k]] <- if (k == 1) 0 else r[k] - r[k - 1]
    }
  }
  eta <- weights['current'] *
    logit(clip01(data$posterior_up)) +
    weights['persistence'] * persistence +
    weights['deterioration'] * deterioration
  data$persistence <- persistence
  data$deterioration <- deterioration
  data$dynamic_bewrs <- inv_logit(eta)
  data
}