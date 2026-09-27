clust_diss2 <- function(x, A, B) {
  vpA <- eigen(x[A, A])$values[1]
  vpB <- eigen(x[B, B])$values[1]
  AUB <- c(A, B)
  vpAUB <- eigen(x[AUB, AUB])$values[1]
  crit <- vpA + vpB - vpAUB
  if (crit < 1e-7) {
    crit <- 0
  }
  return(crit)
}