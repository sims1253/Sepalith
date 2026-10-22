#' Bias of a set of estimates
#'
#' @param estimates Numeric vector of estimates across replications.
#' @param truth True value of the estimand.
#' @return A named vector with the bias and its Monte Carlo standard error.
#' @family performance measures
#' @export
simBias <- function(estimates, truth) {
  estimates <- estimates[is.finite(estimates)]
  nSim <- length(estimates)
  value <- mean(estimates) - truth
  mcse <- stats::sd(estimates) / sqrt(nSim)
  return(c(value = value, mcse = mcse))
}

#' Empirical standard error of a set of estimates
#'
#' @param estimates Numeric vector of estimates across replications.
#' @param truth True value of the estimand. Unused, kept for a uniform
#'   interface with the other performance measures.
#' @return A named vector with the empirical SE and its Monte Carlo
#'   standard error.
#' @family performance measures
#' @export
simEmpSe <- function(estimates, truth) {
  estimates <- estimates[is.finite(estimates)]
  nSim <- length(estimates)
  value <- stats::sd(estimates)
  mcse <- value / sqrt(2 * (nSim - 1))
  return(c(value = value, mcse = mcse))
}

#' Root mean squared error of a set of estimates
#'
#' @param estimates Numeric vector of estimates across replications.
#' @param truth True value of the estimand.
#' @return A named vector with the RMSE and its Monte Carlo standard error.
#' @family performance measures
#' @export
simRmse <- function(estimates, truth) {
  estimates <- estimates[is.finite(estimates)]
  nSim <- length(estimates)
  sqErr <- (estimates - truth)^2
  value <- sqrt(mean(sqErr))
  mcse <- stats::sd(sqErr) / sqrt(nSim) / (2 * value)
  return(c(value = value, mcse = mcse))
}

#' Coverage of a set of confidence intervals
#'
#' @param lower,upper Numeric vectors of interval limits across replications.
#' @param truth True value of the estimand.
#' @return A named vector with the coverage and its Monte Carlo standard
#'   error.
#' @family performance measures
#' @export
simCoverage <- function(lower, upper, truth) {
  ok <- is.finite(lower) & is.finite(upper)
  nSim <- sum(ok)
  covered <- lower[ok] <= truth & upper[ok] >= truth
  value <- mean(covered)
  mcse <- sqrt(value * (1 - value) / nSim)
  return(c(value = value, mcse = mcse))
}

#' Run a simulation for one scenario
#'
#' Repeatedly generates a data set and applies an estimator to it, catching
#' estimator failures so that a single bad replicate does not stop the run.
#'
#' @param generate Function of the scenario parameters returning a data set.
#' @param estimate Function of a data set returning a named numeric vector.
#' @param nRep Number of replications.
#' @param params Named list of scenario parameters passed to `generate`.
#' @param seed Optional integer seed for reproducibility.
#' @return A matrix with one row per replication.
#' @family simulation drivers
#' @export
runSimulation <- function(generate, estimate, nRep = 1000L, params = list(), seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  first <- estimate(do.call(generate, params))
  out <- matrix(NA_real_, nrow = nRep, ncol = length(first),
    dimnames = list(NULL, names(first)))
  out[1, ] <- first
  for (r in seq_len(nRep)[-1]) {
    dat <- do.call(generate, params)
    res <- tryCatch(estimate(dat), error = function(e) NULL)
    if (!is.null(res) && length(res) == ncol(out)) {
      out[r, ] <- res
    }
  }
  return(out)
}

#' Build a full-factorial scenario grid
#'
#' @param ... Named vectors of scenario parameter values.
#' @return A data frame with one row per scenario and a `scenario` id column.
#' @family simulation drivers
#' @export
simulationGrid <- function(...) {
  grid <- expand.grid(..., KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  if (nrow(grid) == 0L) {
    stop("At least one non-empty parameter vector is required.", call. = FALSE)
  }
  return(cbind(scenario = seq_len(nrow(grid)), grid))
}

#' Run a simulation over every scenario of a grid
#'
#' @param grid A scenario grid as returned by [simulationGrid()].
#' @param generate Function of the scenario parameters returning a data set.
#' @param estimate Function of a data set returning a named numeric vector.
#' @param nRep Number of replications.
#' @param seed Optional integer seed for reproducibility.
#' @return A list of result matrices, one per scenario.
#' @family simulation drivers
#' @export
runSimulationGrid <- function(grid, generate, estimate, nRep = 1000L, seed = NULL) {
  paramCols <- setdiff(names(grid), "scenario")
  seeds <- if (is.null(seed)) rep(list(NULL), nrow(grid)) else as.list(seed + grid$scenario)
  results <- Map(function(i, s) {
    params <- as.list(grid[i, paramCols, drop = FALSE])
    runSimulation(generate, estimate, nRep = nRep, params = params, seed = s)
  }, seq_len(nrow(grid)), seeds)
  names(results) <- paste0("scenario", grid$scenario)
  return(results)
}

#' Tabulate performance measures across scenarios
#'
#' @param results A list of result matrices as returned by
#'   [runSimulationGrid()].
#' @param column Name of the estimate column to evaluate.
#' @param truth Numeric vector of true values, one per scenario.
#' @return A data frame with bias, empirical SE and RMSE per scenario.
#' @export
summarizeSimulation <- function(results, column, truth) {
  truth <- rep_len(truth, length(results))
  rows <- Map(function(res, tr) {
    est <- res[, column]
    c(bias = simBias(est, tr)[["value"]],
      empSe = simEmpSe(est, tr)[["value"]],
      rmse = simRmse(est, tr)[["value"]],
      failed = sum(!is.finite(est)))
  }, results, truth)
  out <- as.data.frame(do.call(rbind, rows))
  out$scenario <- names(results)
  return(out[c("scenario", "bias", "empSe", "rmse", "failed")])
}
