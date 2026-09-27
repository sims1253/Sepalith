dSlash <- function(y, mu, sigma2, nu) {
  resp <- z <- vector(mode = "numeric", length = length(y))
  z <- (y - mu)/sqrt(sigma2)
  for (i in 1:length(y)) {
    f1 <- function(u) {
      nu * u^(nu - 0.5) * dnorm(z[i] * sqrt(u))/sqrt(sigma2)
    }
    resp[i] <- integrate(f1, 0, 1)$value
  }
  return(resp)
}