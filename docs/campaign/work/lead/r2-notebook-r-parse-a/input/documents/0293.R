rwnorm2mix <- function(n, kappa1, kappa2, kappa3, mu1, mu2, pmix, ...) {
  allpar <- list(
    kappa1 = kappa1,
    kappa2 = kappa2,
    kappa3 = kappa3,
    mu1 = mu1,
    mu2 = mu2,
    pmix = pmix
  )

  allpar_len <- listLen(allpar)
  if (min(allpar_len) != max(allpar_len)) {
    stop(
      "component size mismatch: number of components of the input parameter vectors differ"
    )
  }

  if (any(allpar$pmix < 0)) {
    stop("\'pmix\' must be non-negative")
  }
  sum_pmix <- sum(allpar$pmix)
  if (signif(sum_pmix, 5) != 1) {
    if (sum_pmix <= 0) {
      stop("\'pmix\' must have at least one positive element")
    }
    allpar$pmix <- allpar$pmix / sum_pmix
    warning("\'pmix\' is rescaled to add up to 1")
  }

  if (any(c(allpar$kappa1, allpar$kappa2) <= 0)) {
    stop("kappa1 and kappa2 must be positive in wnorm2")
  }
  if (any(allpar$kappa1 * allpar$kappa2 - allpar$kappa3^2 <= 1e-10)) {
    stop("abs(kappa3) must be less than sqrt(kappa1*kappa2) in wnorm2")
  }
  if (any(allpar$mu1 < 0 | allpar$mu1 >= 2 * pi)) {
    allpar$mu1 <- prncp_reg(allpar$mu1)
  }
  if (any(allpar$mu2 < 0 | allpar$mu2 >= 2 * pi)) {
    allpar$mu2 <- prncp_reg(allpar$mu2)
  }

  out <- matrix(0, n, 2)
  ncomp <- allpar_len[1] # number of components
  comp_ind <- cID(tcrossprod(rep(1, n), allpar$pmix), ncomp, runif(n))
  # n samples from multinom(ncomp, pmix)
  for (j in seq_len(ncomp)) {
    obs_ind_j <- which(comp_ind == j)
    n_j <- length(obs_ind_j)
    if (n_j > 0) {
      out[obs_ind_j, ] <- rwnorm2(
        n_j,
        kappa1[j],
        kappa2[j],
        kappa3[j],
        mu1[j],
        mu2[j],
        ...
      )
    }
  }

  out
}