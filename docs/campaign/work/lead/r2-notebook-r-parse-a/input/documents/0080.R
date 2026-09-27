compute_p_adj <- function(r, u, d, lambda, v_u, v_d) {
  u_tilde <- u * exp(lambda * v_u)
  d_tilde <- d * exp(-lambda * v_d)
  p_adj <- (r - d_tilde) / (u_tilde - d_tilde)
  return(p_adj)
}