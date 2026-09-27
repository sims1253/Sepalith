solveOLS <- function(S, B) {
  D <- t(S) %*% S
  d <- t(S) %*% B
  A <- cbind(diag(dim(S)[2]))
  bzero <- c(rep(0, dim(S)[2]))
  solution <- solve.QP(D, d, A, bzero)$solution
  names(solution) <- colnames(S)
  print(round(solution / sum(solution), 5))
  return(solution / sum(solution))
}