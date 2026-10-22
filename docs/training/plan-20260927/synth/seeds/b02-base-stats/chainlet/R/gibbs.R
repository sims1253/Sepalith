#' Allocate storage for a Gibbs chain
#'
#' @param nIter Number of iterations.
#' @param paramNames Character vector of parameter names.
#' @return A numeric matrix of `NA` with named columns.
#' @keywords internal
initChainMatrix <- function(nIter, paramNames) {
  return(matrix(NA_real_, nrow = nIter, ncol = length(paramNames),
    dimnames = list(NULL, paramNames)))
}

#' Draw a normal mean from its full conditional
#'
#' @param y Numeric vector of observations.
#' @param sigma2 Current value of the observation variance.
#' @param priorMean Prior mean of the location parameter.
#' @param priorVar Prior variance of the location parameter.
#' @return A single draw.
#' @family full conditionals
#' @keywords internal
drawMuNormal <- function(y, sigma2, priorMean, priorVar) {
  n <- length(y)
  postVar <- 1 / (1 / priorVar + n / sigma2)
  postMean <- postVar * (priorMean / priorVar + sum(y) / sigma2)
  return(stats::rnorm(1, postMean, sqrt(postVar)))
}

#' Draw a normal variance from its full conditional
#'
#' @param y Numeric vector of observations.
#' @param mu Current value of the location parameter.
#' @param priorShape Shape of the inverse-gamma prior on the variance.
#' @param priorRate Rate of the inverse-gamma prior on the variance.
#' @return A single draw.
#' @family full conditionals
#' @keywords internal
drawSigma2Normal <- function(y, mu, priorShape, priorRate) {
  postShape <- priorShape + length(y) / 2
  postRate <- priorRate + sum((y - mu)^2) / 2
  return(1 / stats::rgamma(1, shape = postShape, rate = postRate))
}

#' Draw regression coefficients from their full conditional
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @param sigma2 Current value of the observation variance.
#' @param priorPrec Prior precision applied to every coefficient.
#' @return A vector of coefficient draws.
#' @family full conditionals
#' @keywords internal
drawBetaRegression <- function(X, y, sigma2, priorPrec) {
  p <- ncol(X)
  precision <- crossprod(X) / sigma2 + diag(priorPrec, p)
  R <- chol(precision)
  mean <- backsolve(R, forwardsolve(t(R), crossprod(X, y) / sigma2))
  z <- stats::rnorm(p)
  return(as.vector(mean + backsolve(R, z)))
}

#' Draw a regression variance from its full conditional
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @param beta Current coefficient vector.
#' @param priorShape Shape of the inverse-gamma prior on the variance.
#' @param priorRate Rate of the inverse-gamma prior on the variance.
#' @return A single draw.
#' @family full conditionals
#' @keywords internal
drawSigma2Regression <- function(X, y, beta, priorShape, priorRate) {
  resid <- y - X %*% beta
  postShape <- priorShape + length(y) / 2
  postRate <- priorRate + sum(resid^2) / 2
  return(1 / stats::rgamma(1, shape = postShape, rate = postRate))
}

#' Gibbs sampler for a normal model
#'
#' Samples the mean and variance of a normal sample under a semi-conjugate
#' normal / inverse-gamma prior.
#'
#' @param y Numeric vector of observations.
#' @param nIter Number of iterations.
#' @param priorMean Prior mean of the location parameter.
#' @param priorVar Prior variance of the location parameter.
#' @param priorShape Shape of the inverse-gamma prior on the variance.
#' @param priorRate Rate of the inverse-gamma prior on the variance.
#' @return A matrix of draws with columns `mu` and `sigma2`.
#' @family Gibbs samplers
#' @export
gibbsNormal <- function(y, nIter = 5000L, priorMean = 0, priorVar = 100,
                        priorShape = 0.01, priorRate = 0.01) {
  y <- y[!is.na(y)]
  if (length(y) < 2L) {
    stop("`y` needs at least two non-missing values.", call. = FALSE)
  }
  draws <- initChainMatrix(nIter, c("mu", "sigma2"))
  mu <- mean(y)
  sigma2 <- stats::var(y)
  for (i in seq_len(nIter)) {
    mu <- drawMuNormal(y, sigma2, priorMean, priorVar)
    sigma2 <- drawSigma2Normal(y, mu, priorShape, priorRate)
    draws[i, ] <- c(mu, sigma2)
  }
  return(draws)
}

#' Gibbs sampler for Bayesian linear regression
#'
#' Alternates between the coefficient and variance full conditionals under
#' an independent normal prior on the coefficients.
#'
#' @param X Numeric design matrix.
#' @param y Numeric response vector.
#' @param nIter Number of iterations.
#' @param priorPrec Prior precision applied to every coefficient.
#' @param priorShape Shape of the inverse-gamma prior on the variance.
#' @param priorRate Rate of the inverse-gamma prior on the variance.
#' @return A matrix of draws with one column per coefficient plus `sigma2`.
#' @family Gibbs samplers
#' @export
gibbsRegression <- function(X, y, nIter = 5000L, priorPrec = 1e-4,
                            priorShape = 0.01, priorRate = 0.01) {
  if (nrow(X) != length(y)) {
    stop("`X` and `y` have incompatible dimensions.", call. = FALSE)
  }
  betaNames <- colnames(X)
  if (is.null(betaNames)) {
    betaNames <- paste0("beta", seq_len(ncol(X)))
  }
  draws <- initChainMatrix(nIter, c(betaNames, "sigma2"))
  beta <- qr.solve(X, y)
  sigma2 <- sum((y - X %*% beta)^2) / (nrow(X) - ncol(X))
  for (i in seq_len(nIter)) {
    beta <- drawBetaRegression(X, y, sigma2, priorPrec)
    sigma2 <- drawSigma2Regression(X, y, beta, priorShape, priorRate)
    draws[i, ] <- c(beta, sigma2)
  }
  return(draws)
}

#' Gibbs sampler for a hierarchical normal model
#'
#' Samples group means, the shared observation variance, and the population
#' mean and variance of a one-way random-effects model.
#'
#' @param y Numeric vector of observations.
#' @param group Factor or vector of group labels, one per observation.
#' @param nIter Number of iterations.
#' @param priorShape Shape of the inverse-gamma prior on the variance.
#' @param priorRate Rate of the inverse-gamma prior on the variance.
#' @return A matrix of draws with one column per group mean plus `mu`,
#'   `tau2` and `sigma2`.
#' @family Gibbs samplers
#' @export
gibbsHierarchical <- function(y, group, nIter = 5000L, priorShape = 0.01,
                              priorRate = 0.01) {
  group <- as.factor(group)
  if (length(group) != length(y)) {
    stop("`y` and `group` must have the same length.", call. = FALSE)
  }
  J <- nlevels(group)
  nj <- tabulate(group, nbins = J)
  sumY <- as.vector(tapply(y, group, sum))
  draws <- initChainMatrix(nIter, c(paste0("theta[", levels(group), "]"),
    "mu", "tau2", "sigma2"))
  theta <- sumY / nj
  mu <- mean(theta)
  tau2 <- max(stats::var(theta), 1e-3)
  sigma2 <- stats::var(y - theta[group])
  for (i in seq_len(nIter)) {
    # group means given everything else
    postVar <- 1 / (nj / sigma2 + 1 / tau2)
    postMean <- postVar * (sumY / sigma2 + mu / tau2)
    theta <- stats::rnorm(J, postMean, sqrt(postVar))
    # population mean with a flat prior
    mu <- stats::rnorm(1, mean(theta), sqrt(tau2 / J))
    tau2 <- drawSigma2Normal(theta, mu, priorShape, priorRate)
    sigma2 <- drawSigma2Normal(y - theta[group], 0, priorShape, priorRate)
    draws[i, ] <- c(theta, mu, tau2, sigma2)
  }
  return(draws)
}
