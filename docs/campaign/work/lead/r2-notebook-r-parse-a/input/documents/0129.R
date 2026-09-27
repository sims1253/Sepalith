admm_spca_deflation <- function(Sig, vec) {
  p <- length(vec)
  term1 <- (Sig %*% outer(vec, vec) %*% Sig)
  term2 <- sum((as.vector(Sig %*% matrix(vec, nrow = p))) * vec)

  output <- Sig - term1 / term2
  return(output)
}