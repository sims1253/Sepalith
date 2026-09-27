compute.effect.size <- function(df) {
  groups <- split(df$expression, df$group)

  if (length(groups) < 2) {
    return(NA)
  }

  g1 <- groups[[1]]
  g2 <- groups[[2]]

  mean1 <- mean(g1)
  mean2 <- mean(g2)

  sd1 <- sd(g1)
  sd2 <- sd(g2)

  n1 <- length(g1)
  n2 <- length(g2)

  pooled.sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
  d <- (mean1 - mean2) / pooled.sd

  return(d)
}