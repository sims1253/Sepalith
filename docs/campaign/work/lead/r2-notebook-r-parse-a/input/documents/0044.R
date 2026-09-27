optim.arbidept.adjp <- function(alpha, p, k) {
  m <- length(p)
  invp <- adjp <- numeric(m)
  s <- c()
  s[1] <- 0 # r  number of rejection, s number of acceptance
  adjp[1] <- invp[1] <- k * p[1]
  for (i in 2:m) {
    s[i] <- sum(adjp[1:i - 1] > alpha) # count number of acceptances for firtst (i-1) hypotheses.
    if (i <= k) {
      invp[i] <- k * p[i]
    } else if (i > k) {
      invp[i] <- (m - i + 1) * k * p[i] / (m - k + 1)
    }
    adjp[i] <- invp[i] * (s[i] < k) + 1 * (s[i] >= k)
  }
  return(adjp - alpha)
}