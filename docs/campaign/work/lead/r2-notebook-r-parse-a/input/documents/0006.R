sigma_mv <- function(num_sigma, numcol) {
  i <- 1
  sigma_vec <- list()
  # Initialze a diagonal matrix as the
  # co-variance matrix.
  while (i <= num_sigma) {
    sigma_vec[[i]] <- diag(numcol)
    i <- i + 1
  }
  return(sigma_vec)
}