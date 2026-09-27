est.mcfa <- function(
  Y,
  g,
  q,
  itmax,
  tol,
  pivec,
  A,
  xi,
  omega,
  D,
  convMeas,
  ...
) {
  if (!is.matrix(Y)) {
    Y <- as.matrix(Y)
  }
  p <- ncol(Y)
  if (is.null(p)) {
    p <- 1
  }
  n <- nrow(Y)

  fit <- list(g = g, q = q, pivec = pivec, A = A, xi = xi, omega = omega, D = D)

  fit$logL <- loglike.mcfa(Y, g, q, pivec, A, xi, omega, D)
  if (class(fit$logL)[1] == 'character') {
    FIT <- paste(
      'Failed in computing the 
            log-likelihood before the EM-steps,',
      fit$logL
    )
    class(FIT) <- "error"
    return(FIT)
  }
  for (niter in seq_len(itmax)) {
    FIT <- do.call('Mstep.mcfa', c(list(Y = Y), fit))
    if (class(FIT)[1] == 'error') {
      FIT <- paste(
        'Computational error in ',
        niter,
        'iteration of the M-step:',
        FIT
      )
      class(FIT) <- "error"
      return(FIT)
    }
    FIT$logL <- do.call('loglike.mcfa', c(list(Y = Y), FIT))
    if (class(FIT$logL)[1] == 'character') {
      FIT <- paste(
        'Failed to compute log-likelihood after ',
        niter,
        'th the M-step:',
        FIT$logL,
        sep = ''
      )
      class(FIT) <- "error"
      return(FIT)
    }
    if ((FIT$logL == -Inf) | is.na(FIT$logL)) {
      FIT <- paste(
        'Log likelihood computed after the',
        niter,
        'th iteration of the M-step is not finite',
        sep = ''
      )
      class(FIT) <- "error"
      return(FIT)
    }
    if ((convMeas == "diff") & (abs(FIT$logL - fit$logL) < tol)) {
      break
    }
    if ((convMeas == "ratio") & (abs((FIT$logL - fit$logL) / FIT$logL) < tol)) {
      break
    }
    fit <- FIT
  }
  class(FIT) <- "mcfa"
  return(FIT)
}