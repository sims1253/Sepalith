DualObj <- function(X, x, Pi, ll_cur, gamma_n = 0.05) {
  n <- nrow(X)
  A <- t(X) %*% Pi %*% X
  obj <- t(ll_cur) %*%
    A %*%
    ll_cur /
    (4 * n) +
    sum(x * ll_cur) +
    gamma_n * sum(abs(ll_cur))

  return(obj)
}