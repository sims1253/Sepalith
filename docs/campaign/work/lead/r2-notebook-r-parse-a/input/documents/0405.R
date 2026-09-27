compute_M_optimized <- function(times, status, f, u_bis, clusters) {
  N <- length(times)
  unique_times <- sort(unique(times))

  # Censoring survival estimation
  if (f == "KaplanMeier_Censoring_vectorized") {
    G_vals <- KaplanMeier_Censoring_vectorized(times, status, unique_times)
  }
  if (f == "Nelson_Censoring_vectorized") {
    G_vals <- Nelson_Censoring_vectorized(
      times,
      unique_times,
      status,
      clusters,
      u_bis
    )
  }
  G_func <- create_G_function(unique_times, G_vals)

  # Matrix form for efficient vectorized comparison
  T_mat_i <- matrix(rep(times, each = N), nrow = N)
  T_mat_j <- matrix(rep(times, times = N), nrow = N)
  fail_mat_j <- matrix(rep(status, times = N), nrow = N)

  cond1 <- (T_mat_j <= T_mat_i) & (fail_mat_j > 1)
  cond2 <- (T_mat_j >= T_mat_i)
  cond3 <- (T_mat_j < T_mat_i) & (fail_mat_j <= 1)

  G_T_i <- G_func(times)
  G_T_j <- G_func(times)
  G_T_i_mat <- matrix(rep(G_T_i, each = N), nrow = N)
  G_T_j_mat <- matrix(rep(G_T_j, times = N), nrow = N)

  M <- Matrix(0, nrow = N, ncol = N)
  M[cond1] <- G_T_i_mat[cond1] / G_T_j_mat[cond1]
  M[cond2] <- 1
  M[cond3] <- 0

  return(M)
}