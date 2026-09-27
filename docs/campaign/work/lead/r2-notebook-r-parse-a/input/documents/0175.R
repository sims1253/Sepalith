loglike.mcfa <- function(Y, g, q, pivec, A, xi, omega, D, ...) {
  p <- ncol(Y)
  if (is.null(p)) {
    p <- 1
  }
  n <- nrow(Y)

  Fji <- array(NA, c(n, g))
  InvD <- diag(1 / diag(D))
  for (i in seq_len(g)) {
    InvS <- try(
      InvD -
        InvD %*%
          A %*%
          chol.inv(chol.inv(omega[,, i]) + t(A) %*% InvD %*% A) %*%
          t(A) %*%
          InvD
    )
    if (class(InvS)[1] == "try-error") {
      return(loglike <- paste('ill-cond. or sing. Sigma_', i, sep = ''))
    }
    logdetD <- log(det(as.matrix(omega[,, i]))) +
      sum(log(diag(D))) +
      log(det(chol.inv(omega[,, i]) + t(A) %*% InvD %*% A))

    MhalDist <- stats::mahalanobis(
      Y,
      t(A %*% xi[, i, drop = FALSE]),
      InvS,
      inverted = TRUE
    )
    Fji[, i] <- -0.5 * MhalDist - (p / 2) * log(2 * pi) - 0.5 * logdetD
  }
  Fji <- sweep(Fji, 2, log(pivec), '+')
  Fjmax <- apply(Fji, 1, max)
  Fji <- sweep(Fji, 1, Fjmax, '-')
  loglike <- sum(Fjmax, log(rowSums(exp(Fji))))
  return(loglike)
}