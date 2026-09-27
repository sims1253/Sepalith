build_integration_matrix <- function(n, dt) {
  A <- matrix(0, n, n)
  for (i in 2:n) {
    A[i, 1:(i - 1)] <- dt[1:(i - 1)]
  }
  A
}