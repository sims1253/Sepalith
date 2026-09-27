.EM_initialise <- function(basis, Z, X, Ve, mstar) {
  l <- list() # list of initial values
  nres <- max(basis@df$res) # number of basis function resolutions

  ## Initialise the expectations and covariances from E-step to reasonable values
  l$mu_eta_init <- Matrix(0, nbasis(basis), 1)
  l$mu_xi_init <- Matrix(0, mstar, 1)
  l$S_eta_init <- Diagonal(x = rep(1, nbasis(basis)))
  l$Q_eta_init <- Diagonal(x = rep(1, nbasis(basis)))

  ## Start with reasonable parameter estimates (that will be updated in M-step)
  l$K_init <- Diagonal(n = nbasis(basis), x = 1 / (1 / var(Z[, 1])))
  l$K_inv_init <- solve(l$K_init)

  if (!is.finite(determinant(t(X) %*% X)$modulus)) {
    stop(
      "Matrix of covariates has columns that are linearly dependent. Please change formula or covariates."
    )
  }

  if (ncol(X) == 0) {
    stop("We need at least one covariate in the model")
  } else {
    l$alphahat_init <- solve(t(X) %*% X) %*% t(X) %*% Z
  }
  l$sigma2fshat_init <- mean(diag(Ve)) / 4

  return(l)
}