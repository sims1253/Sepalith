#' Log density of a normal prior
#'
#' @param theta Numeric parameter vector.
#' @param mean Prior mean.
#' @param sd Prior standard deviation.
#' @return The summed log density.
#' @export
logPriorNormal <- function(theta, mean = 0, sd = 10) {
  return(sum(stats::dnorm(theta, mean = mean, sd = sd, log = TRUE)))
}

#' Log density of a gamma prior
#'
#' @inheritParams logPriorNormal
#' @param shape Prior shape.
#' @param rate Prior rate.
#' @return The summed log density, or `-Inf` outside the support.
#' @export
logPriorGamma <- function(theta, shape = 1, rate = 1) {
  if (any(theta <= 0)) {
    return(-Inf)
  }
  return(sum(stats::dgamma(theta, shape = shape, rate = rate, log = TRUE)))
}

#' Log density of a beta prior
#'
#' @inheritParams logPriorNormal
#' @param shape1,shape2 Prior shape parameters.
#' @return The summed log density, or `-Inf` outside the support.
#' @export
logPriorBeta <- function(theta, shape1 = 1, shape2 = 1) {
  if (any(theta <= 0 | theta >= 1)) {
    return(-Inf)
  }
  return(sum(stats::dbeta(theta, shape1 = shape1, shape2 = shape2, log = TRUE)))
}

#' Combine a log-likelihood and log-prior
#'
#' @param logLik Function of `theta` returning the log-likelihood.
#' @param logPrior Function of `theta` returning the log-prior.
#' @return A function of `theta` returning the unnormalised log posterior.
#' @export
makeLogPosterior <- function(logLik, logPrior) {
  logLik <- match.fun(logLik)
  logPrior <- match.fun(logPrior)

  # Check the prior first: there is no point evaluating a possibly
  # expensive likelihood outside the support.
  return(function(theta) {
    lp <- logPrior(theta)
    if (!is.finite(lp)) {
      return(-Inf)
    }
    return(lp + logLik(theta))
  })
}

#' Random-walk Metropolis sampler
#'
#' Draws from a target density using independent Gaussian increments on
#' each coordinate.
#'
#' @param logTarget Function returning the unnormalised log density.
#' @param init Numeric vector of starting values.
#' @param nIter Number of iterations.
#' @param proposalSd Proposal standard deviation, scalar or per coordinate.
#' @return A matrix of draws with attribute `accepted`.
#' @export
rwMetropolis <- function(logTarget, init, nIter = 5000L, proposalSd = 0.5) {
  p <- length(init)
  proposalSd <- rep_len(proposalSd, p)
  draws <- matrix(NA_real_, nrow = nIter, ncol = p, dimnames = list(NULL, names(init)))
  current <- init
  currentLp <- logTarget(current)
  if (!is.finite(currentLp)) {
    stop("`init` has zero density under the target.", call. = FALSE)
  }

  accepted <- logical(nIter)
  for (i in seq_len(nIter)) {
    proposal <- current + stats::rnorm(p, sd = proposalSd)
    proposalLp <- logTarget(proposal)
    if (log(stats::runif(1)) < proposalLp - currentLp) {
      current <- proposal
      currentLp <- proposalLp
      accepted[i] <- TRUE
    }
    draws[i, ] <- current
  }

  attr(draws, "accepted") <- accepted
  return(draws)
}

#' Independence Metropolis sampler
#'
#' Draws from a target density using a fixed Gaussian proposal that does not
#' depend on the current state.
#'
#' @inheritParams rwMetropolis
#' @param proposalMean Mean of the Gaussian proposal.
#' @return A matrix of draws with attribute `accepted`.
#' @export
indepMetropolis <- function(logTarget, init, nIter = 5000L, proposalMean = init,
                            proposalSd = 1) {
  p <- length(init)
  proposalSd <- rep_len(proposalSd, p)
  logQ <- function(x) sum(stats::dnorm(x, proposalMean, proposalSd, log = TRUE))

  # Accept on the ratio of importance weights target / proposal.
  draws <- matrix(NA_real_, nrow = nIter, ncol = p, dimnames = list(NULL, names(init)))
  current <- init
  currentLw <- logTarget(current) - logQ(current)

  accepted <- logical(nIter)
  for (i in seq_len(nIter)) {
    proposal <- stats::rnorm(p, proposalMean, proposalSd)
    proposalLw <- logTarget(proposal) - logQ(proposal)
    if (log(stats::runif(1)) < proposalLw - currentLw) {
      current <- proposal
      currentLw <- proposalLw
      accepted[i] <- TRUE
    }
    draws[i, ] <- current
  }

  attr(draws, "accepted") <- accepted
  return(draws)
}

#' Acceptance rate of a Metropolis chain
#'
#' @param chain A matrix of draws returned by one of the samplers.
#' @return The proportion of accepted proposals.
#' @export
acceptanceRate <- function(chain) {
  accepted <- attr(chain, "accepted")
  if (is.null(accepted)) {
    stop("`chain` carries no acceptance record.", call. = FALSE)
  }
  return(mean(accepted))
}

#' Tune a proposal scale towards a target acceptance rate
#'
#' Runs short pilot chains and rescales the proposal standard deviation on
#' the log scale until the acceptance rate is close to `target`.
#'
#' @param logTarget Function returning the unnormalised log density.
#' @param init Numeric vector of starting values.
#' @param proposalSd Initial proposal standard deviation.
#' @param target Desired acceptance rate.
#' @param pilotIter Iterations per pilot run.
#' @param maxRounds Maximum number of pilot runs.
#' @return The tuned proposal standard deviation.
#' @export
tuneProposalSd <- function(logTarget, init, proposalSd = 1, target = 0.234,
                           pilotIter = 500L, maxRounds = 20L) {
  for (round in seq_len(maxRounds)) {
    pilot <- rwMetropolis(logTarget, init, nIter = pilotIter, proposalSd = proposalSd)
    rate <- acceptanceRate(pilot)
    if (abs(rate - target) < 0.05) {
      break
    }
    # Too many acceptances means steps are too small, and vice versa.
    proposalSd <- proposalSd * exp(rate - target)
    init <- pilot[pilotIter, ]
  }
  return(proposalSd)
}

#' Adaptive random-walk Metropolis sampler
#'
#' Learns a multivariate Gaussian proposal from the chain history after an
#' initial non-adaptive phase.
#'
#' @details The proposal covariance is the running covariance of the chain
#' scaled by `2.38^2 / p`, plus a small ridge so it stays positive definite.
#' The running estimates include the starting value as a pseudo-draw and
#' start from a diagonal covariance of 0.1.
#'
#' @inheritParams rwMetropolis
#' @param adaptStart Iteration after which the proposal starts adapting.
#' @param epsilon Small ridge added to the proposal covariance.
#' @return A matrix of draws with attribute `accepted`.
#' @export
adaptiveMetropolis <- function(logTarget, init, nIter = 5000L, adaptStart = 500L,
                               epsilon = 1e-6) {
  p <- length(init)
  scale <- 2.38^2 / p
  draws <- matrix(NA_real_, nrow = nIter, ncol = p, dimnames = list(NULL, names(init)))
  current <- init
  currentLp <- logTarget(current)
  if (!is.finite(currentLp)) {
    stop("`init` has zero density under the target.", call. = FALSE)
  }

  accepted <- logical(nIter)
  runMean <- init
  runCov <- diag(0.1, p)

  for (i in seq_len(nIter)) {
    # Fixed small steps until there is enough history to adapt to.
    propCov <- if (i > adaptStart) scale * runCov + diag(epsilon, p) else diag(0.01, p)
    proposal <- current + drop(stats::rnorm(p) %*% chol(propCov))
    proposalLp <- logTarget(proposal)
    if (log(stats::runif(1)) < proposalLp - currentLp) {
      current <- proposal
      currentLp <- proposalLp
      accepted[i] <- TRUE
    }
    draws[i, ] <- current

    # Welford-style update of the running mean and covariance
    delta <- current - runMean
    runMean <- runMean + delta / (i + 1)
    runCov <- runCov + (tcrossprod(delta) * i / (i + 1) - runCov) / (i + 1)
  }
  attr(draws, "accepted") <- accepted
  return(draws)
}
