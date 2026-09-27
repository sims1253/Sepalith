AcumNormalC <- function(y, mu, sigma2, nu) {
  Acum <- vector(mode = "numeric", length = length(y))
  eta <- nu[1]
  gama <- nu[2]
  Acum <- eta * pnorm(y, mu, sqrt(sigma2/gama)) + (1 - eta) * pnorm(y, 
                                                                    mu, sqrt(sigma2))
  return(Acum)
}