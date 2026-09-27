doubly_censored_expectation <- function(theta, sigma) {
  pM1 <- theta * (pnorm((1.0 - theta) / sigma) - pnorm((0.0 - theta) / sigma))
  pM2 <- sigma * (dnorm((0.0 - theta) / sigma) - dnorm((1.0 - theta) / sigma))
  pR <- 1.0 * (1.0 - pnorm((1.0 - theta) / sigma))
  output <- pM1 + pM2 + pR
  return(output)
}