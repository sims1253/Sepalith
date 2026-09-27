rztpois_er <- function(n, lambda) {
  if (length(n) > 1) {
    n <- length(n)
  }
  out <- rep(NA, n)
  index <- 1:n
  len_lam <- length(lambda)
  index <- rep(1:len_lam, length.out = n)
  for (i in 1:len_lam) {
    out[index == i] <- rztpois_sampler(n = sum(index == i), lambda = lambda[i])
  }
  return(out)
}