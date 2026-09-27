A1 <- function(kappa) {
  result <- besselI(kappa, nu = 1, expon.scaled = TRUE) /
    besselI(kappa, nu = 0, expon.scaled = TRUE)
  return(result)
}