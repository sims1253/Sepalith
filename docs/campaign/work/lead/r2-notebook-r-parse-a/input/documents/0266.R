PartialCorrelations <- function(Q) {
  if (length(dim(Q)) <= 1) {
    stop("Q needs to be at least a 2-dimensional matrix")
  }
  k <- dim(Q)[1]
  NAMES <- colnames(Q)
  if (length(dim(Q)) == 2) {
    Q <- array(Q, c(k, k, 1), dimnames = list(NAMES, NAMES))
  }
  pcc <- Q
  for (l in 1:dim(Q)[3]) {
    precision <- MASS::ginv(Q[,, l])
    theta <- diag(1 / sqrt(diag(precision)))
    pcc[,, l] <- -theta %*% precision %*% theta
  }
  return(pcc)
}