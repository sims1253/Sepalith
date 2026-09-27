brownsMethod <- function(p_values, data_matrix = NULL, cov_matrix = NULL) {
  if (missing(data_matrix) && missing(cov_matrix)) {
    stop("Either data_matrix or cov_matrix must be supplied")
  }
  if (!(missing(data_matrix) || missing(cov_matrix))) {
    message(
      "Both data_matrix and cov_matrix were supplied. Ignoring data_matrix"
    )
  }
  if (missing(cov_matrix)) {
    cov_matrix <- calculateCovariances(data_matrix)
  }

  N <- ncol(cov_matrix)
  expected <- 2 * N
  cov_sum <- 2 * sum(cov_matrix[lower.tri(cov_matrix, diag = FALSE)])
  var <- (4 * N) + cov_sum
  sf <- var / (2 * expected)

  df <- (2 * expected^2) / var
  if (df > 2 * N) {
    df <- 2 * N
    sf <- 1
  }

  # Acquiring the unadjusted chi-squared values from Fisher's method
  fisher_chisq <- c()
  for (i in seq_along(p_values[, 1])) {
    fisher_chisq <- c(fisher_chisq, fishersMethod(p_values[i, ]))
  }

  # Adjusted p-value
  p_brown <- stats::pchisq(df = df, q = fisher_chisq / sf, lower.tail = FALSE)
  names(p_brown) <- rownames(p_values)
  p_brown
}