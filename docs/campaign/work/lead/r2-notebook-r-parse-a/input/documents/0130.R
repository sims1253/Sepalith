admm_spca_rk1vec <- function(X) {
  y <- as.vector(base::eigen(X)$vectors[, 1])
  return(y)
}