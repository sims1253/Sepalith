AcumSlash <- function(y, mu, sigma2, nu) {
  Acum <- z <- vector(mode = "numeric", length = length(y))
  z <- (y - mu)/sqrt(sigma2)
  for (i in 1:length(y)) {
    f1 <- function(u) {
      nu * u^(nu - 1) * pnorm(z[i] * sqrt(u))
    }
    Acum[i] <- integrate(f1, 0, 1)$value
  }
  return(Acum)
}