build_difference_matrix <- function(n) {
  D <- matrix(0, n - 1, n)
  for (i in 1:(n - 1)) {
    D[i, i] <- -1
    D[i, i + 1] <- 1
  }
  D
}