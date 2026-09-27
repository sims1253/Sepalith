t_statistic_Q <- function(f_data, lag) {
  N = NCOL(f_data)
  J = NROW(f_data)
  gamma_hat <- autocov_approx_h(f_data, lag)
  Q_T_h <- N * sum(gamma_hat^2) / (J^2)
  Q_T_h
}