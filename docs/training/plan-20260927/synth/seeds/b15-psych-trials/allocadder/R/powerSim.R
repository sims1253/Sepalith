#' Simulate one two-arm trial with a continuous endpoint
#'
#' @param n1 Number of participants in arm 1.
#' @param n2 Number of participants in arm 2.
#' @param delta True mean difference, arm 2 minus arm 1.
#' @param sd Common standard deviation of the endpoint.
#' @return A data frame with columns `arm` and `y`.
#' @family trial simulators
#' @keywords internal
simContinuousTrial <- function(n1, n2, delta, sd) {
  data.frame(
    arm = factor(rep(c("control", "active"), c(n1, n2)), levels = c("control", "active")),
    y = c(stats::rnorm(n1, 0, sd), stats::rnorm(n2, delta, sd))
  )
}

#' Simulate one two-arm trial with a binary endpoint
#'
#' @param n1 Number of participants in arm 1.
#' @param n2 Number of participants in arm 2.
#' @param p1 True response proportion in arm 1.
#' @param p2 True response proportion in arm 2.
#' @return A data frame with columns `arm` and `y`.
#' @family trial simulators
#' @keywords internal
simBinaryTrial <- function(n1, n2, p1, p2) {
  data.frame(
    arm = factor(rep(c("control", "active"), c(n1, n2)), levels = c("control", "active")),
    y = c(stats::rbinom(n1, 1, p1), stats::rbinom(n2, 1, p2))
  )
}

#' P-value for a simulated continuous trial
#'
#' @param trial A simulated trial with columns `arm` and `y`.
#' @return The two-sided Welch t-test p-value.
#' @family trial analysers
#' @keywords internal
analyseContinuousTrial <- function(trial) {
  fit <- stats::t.test(y ~ arm, data = trial)
  fit$p.value
}

#' P-value for a simulated binary trial
#'
#' @param trial A simulated trial with columns `arm` and `y`.
#' @return The two-sided chi-squared test p-value, or 1 if no events occurred.
#' @family trial analysers
#' @keywords internal
analyseBinaryTrial <- function(trial) {
  tab <- table(trial$arm, factor(trial$y, levels = 0:1))
  if (sum(tab[, "1"]) == 0) {
    return(1)
  }
  fit <- suppressWarnings(stats::prop.test(tab[, "1"], rowSums(tab)))
  fit$p.value
}

#' Simulation-based power for a two-arm trial
#'
#' Repeatedly simulates and analyses a trial and reports the proportion of
#' significant results with a Monte Carlo standard error.
#'
#' @param n1 Number of participants in arm 1.
#' @param ratio Allocation ratio of arm 2 to arm 1.
#' @param endpoint Endpoint type, `"continuous"` or `"binary"`.
#' @param ... Effect parameters: `delta` and `sd` for continuous endpoints,
#'   `p1` and `p2` for binary endpoints.
#' @param alpha Type I error rate.
#' @param nSim Number of simulated trials.
#' @param seed Random seed for the simulation.
#' @return A list with `power`, `mcse` and `nSim`.
#' @export
simulatePower <- function(n1, ratio = 1, endpoint = c("continuous", "binary"), ...,
                          alpha = 0.05, nSim = 1000, seed = NULL) {
  endpoint <- match.arg(endpoint)
  args <- list(...)
  n2 <- ceiling(ratio * n1)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  pvals <- numeric(nSim)
  for (i in seq_len(nSim)) {
    pvals[i] <- switch(endpoint,
      continuous = analyseContinuousTrial(
        simContinuousTrial(n1, n2, delta = args$delta, sd = args$sd)
      ),
      binary = analyseBinaryTrial(
        simBinaryTrial(n1, n2, p1 = args$p1, p2 = args$p2)
      )
    )
  }
  power <- mean(pvals < alpha)
  list(power = power, mcse = sqrt(power * (1 - power) / nSim), nSim = nSim)
}

#' Simulated power curve over sample sizes
#'
#' Evaluates [simulatePower()] at each value of `n1Values`, reusing the same
#' seed so that neighbouring points share random numbers.
#'
#' @param n1Values Integer vector of arm-1 sample sizes.
#' @param ... Passed to [simulatePower()].
#' @param seed Random seed for the simulation.
#' @return A data frame with columns `n1`, `power` and `mcse`.
#' @export
powerCurve <- function(n1Values, ..., seed = 1) {
  power <- numeric(length(n1Values))
  mcse <- numeric(length(n1Values))
  for (i in seq_along(n1Values)) {
    res <- simulatePower(n1Values[i], ..., seed = seed)
    power[i] <- res$power
    mcse[i] <- res$mcse
  }
  data.frame(n1 = n1Values, power = power, mcse = mcse)
}

#' Smallest sample size reaching a target power
#'
#' Bisection search over the arm-1 sample size using simulated power. Because
#' the power estimate is noisy, the result should be confirmed with a larger
#' `nSim` before going into a protocol.
#'
#' @param target Desired power.
#' @param lower,upper Bracketing values of the arm-1 sample size.
#' @param ... Passed to [simulatePower()].
#' @param seed Random seed for the simulation.
#' @return A list with the chosen `n1`, its estimated `power` and the number
#'   of `iterations` used.
#' @export
findSampleSize <- function(target = 0.8, lower = 10, upper = 1000, ..., seed = 1) {
  powerAt <- function(n) simulatePower(n, ..., seed = seed)$power
  if (powerAt(upper) < target) {
    stop("Target power not reached at `upper` = ", upper, ".", call. = FALSE)
  }
  if (powerAt(lower) >= target) {
    return(list(n1 = lower, power = powerAt(lower), iterations = 0L))
  }
  iterations <- 0L
  while (upper - lower > 1) {
    iterations <- iterations + 1L
    mid <- floor((lower + upper) / 2)
    if (powerAt(mid) >= target) {
      upper <- mid
    } else {
      lower <- mid
    }
  }
  list(n1 = upper, power = powerAt(upper), iterations = iterations)
}
