rztpois_sampler <- function(n, lambda) {
  N <- n * 5 * as.integer((lambda) / (1 - exp(-lambda))) + 100
  u <- stats::runif(N, 0, 1)
  k <- stats::rpois(N, lambda) + 1
  out <- k[u < 1 / k]
  if (length(out) < n) {
    additional_samples <- rztpois_sampler(n * 2, lambda)
    out <- c(out, additional_samples)
  }
  return(out[1:n])
}