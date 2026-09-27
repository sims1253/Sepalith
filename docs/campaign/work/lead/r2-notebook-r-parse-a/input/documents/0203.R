autocov_approx_h <- function(f_data, lag) {
  N = NCOL(f_data)
  c_f_data <- center(f_data)
  gamma_hat <- c_f_data[,(1+lag):N]%*%t(c_f_data[,1:(N-lag)])/ N
  gamma_hat
}