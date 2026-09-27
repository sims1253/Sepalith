circ.cor <- function(alpha, beta, test = FALSE) {
  n <- length(alpha)
  alpha.bar <- circ.mean(alpha)
  beta.bar <- circ.mean(beta)
  num <- sum(sin(alpha - alpha.bar) * sin(beta - beta.bar))
  den <- sqrt(sum(sin(alpha - alpha.bar)^2) * sum(sin(beta - beta.bar)^2))
  r <- num / den
  result <- data.frame(r)
  if (test) {
    l20 <- mean(sin(alpha - alpha.bar)^2)
    l02 <- mean(sin(beta - beta.bar)^2)
    l22 <- mean((sin(alpha - alpha.bar)^2) * (sin(beta - beta.bar)^2))
    test.stat <- sqrt((n * l20 * l02) / l22) * r
    p.value <- 2 * (1 - pnorm(abs(test.stat)))
    result <- data.frame(r, test.stat, p.value)
  }
  result
}