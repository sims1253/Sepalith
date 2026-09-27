strubesMethod <- function(p_values) {
  # Acquiring the unadjusted z-value from Stouffer's method
  stouffer_z <- c()
  for (i in seq_along(p_values[, 1])) {
    stouffer_z <- c(stouffer_z, stouffersMethod(p_values[i, ]))
  }

  # Correlation matrix
  cor_mtx <- stats::cor(p_values, use = "complete.obs")
  cor_mtx[is.na(cor_mtx)] <- 0
  cor_mtx <- abs(cor_mtx)

  # Adjusted p-value
  k <- length(p_values[1, ])
  adjusted_z <- stouffer_z * sqrt(k) / sqrt(sum(cor_mtx))
  p_strube <- 2 * stats::pnorm(-1 * abs(adjusted_z))
  names(p_strube) <- rownames(p_values)
  p_strube
}