autocorrelation_coeff_h <- function(f_data, lag) {
  N <- NCOL(f_data)
  num <- sqrt(t_statistic_Q(f_data, lag))
  denom <- sqrt(N) * diagonal_autocov_approx_0(f_data)
  coefficient <- num / denom
  coefficient
}