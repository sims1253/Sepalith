#' Initial compartment state
#'
#' @param N Total population size.
#' @param exposed Initial number of exposed individuals.
#' @param infectious Initial number of infectious individuals.
#' @param recovered Initial number of recovered individuals.
#' @return A named numeric vector with elements `S`, `E`, `I` and `R` that
#'   sums to `N`.
#' @export
initState = function(N, exposed = 0, infectious = 1, recovered = 0) {
  if (!is.numeric(N) || length(N) != 1L || N <= 0) {
    stop("`N` must be a single positive number.", call. = FALSE)
  }
  # Check `exposed`
  if (!is.numeric(exposed) || length(exposed) != 1L || exposed < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "exposed"), call. = FALSE)
  }
  # Check `infectious`
  if (!is.numeric(infectious) || length(infectious) != 1L || infectious < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "infectious"), call. = FALSE)
  }
  # Check `recovered`
  if (!is.numeric(recovered) || length(recovered) != 1L || recovered < 0) {
    stop(sprintf("`%s` must be a single non-negative number.", "recovered"), call. = FALSE)
  }

  susceptible = N - exposed - infectious - recovered
  if (susceptible < 0) {
    stop("Initial compartments exceed the population size `N`.", call. = FALSE)
  }
  c(S = susceptible, E = exposed, I = infectious, R = recovered)
}

#' Simulate a trajectory with a step function
#'
#' Applies `stepFun` repeatedly and stores the state at every whole day in a
#' preallocated matrix.
#'
#' @param state A named numeric vector with elements `S`, `E`, `I` and `R`.
#' @param stepFun A function `(state, params, dt)` returning the next state.
#' @param params A list of rates per day, passed to `stepFun`.
#' @param days Number of days to simulate.
#' @param dt Step length in days; must divide one day evenly.
#' @return A data frame with columns `time`, `S`, `E`, `I` and `R` and
#'   `days + 1` rows.
#' @export
simulateTrajectory = function(state, stepFun, params, days, dt = 1) {
  if (!is.function(stepFun)) {
    stop("`stepFun` must be a function.", call. = FALSE)
  }
  if (!is.numeric(days) || length(days) != 1L || days < 1) {
    stop(sprintf("`%s` must be a single positive number.", "days"), call. = FALSE)
  }
  if (!is.numeric(dt) || length(dt) != 1L || dt <= 0) {
    stop(sprintf("`%s` must be a single positive number.", "dt"), call. = FALSE)
  }
  stepsPerDay = round(1 / dt)
  if (abs(stepsPerDay * dt - 1) > 1e-8) {
    stop("`dt` must divide one day into a whole number of steps.", call. = FALSE)
  }
  compartments = c("S", "E", "I", "R")
  out = matrix(NA_real_, nrow = days + 1L, ncol = length(compartments),
    dimnames = list(NULL, compartments))
  out[1L, ] = state[compartments]

  # Fill one row per day
  for (day in seq_len(days)) {
    for (k in seq_len(stepsPerDay)) {
      state = stepFun(state, params, dt)
    }
    out[day + 1L, ] = state[compartments]
  }
  data.frame(time = 0:days, out)
}

#' Simulate the SEIRS equations with an ODE solver
#'
#' @param state A named numeric vector with elements `S`, `E`, `I` and `R`.
#' @param params A list of rates per day, see [checkSeirParams()].
#' @param times Output times in days.
#' @param method Integration method passed to [deSolve::ode()].
#' @return A data frame with columns `time`, `S`, `E`, `I` and `R`.
#' @export
simulateOde = function(state, params, times, method = "lsoda") {
  rhs = function(t, y, p) {
    N = sum(y)
    infection = p$beta * y[["S"]] * y[["I"]] / N
    list(c(
      S = -infection + p$xi * y[["R"]],
      E = infection - p$sigma * y[["E"]],
      I = p$sigma * y[["E"]] - p$gamma * y[["I"]],
      R = p$gamma * y[["I"]] - p$xi * y[["R"]]
    ))
  }
  solution = deSolve::ode(y = state, times = times, func = rhs, parms = params,
    method = method)
  as.data.frame(solution)
}

#' Summarise the key features of a trajectory
#'
#' @param traj A data frame with columns `time`, `S`, `E`, `I` and `R`.
#' @return A list with elements `minS`, `peakE`, `peakI` and `finalR`; each
#'   is a list with the `value` and the `time` at which it occurs.
#' @export
trajectorySummary = function(traj) {
  absent = setdiff(c("time", "S", "E", "I", "R"), names(traj))
  if (length(absent) > 0L) {
    stop(sprintf("Trajectory is missing column `%s`.", absent[[1L]]), call. = FALSE)
  }

  # Lowest number of susceptibles
  at = which.min(traj$S)
  minS = list(value = traj$S[[at]], time = traj$time[[at]])

  # Peak of the exposed compartment
  at = which.max(traj$E)
  peakE = list(value = traj$E[[at]], time = traj$time[[at]])

  # Peak of the infectious compartment
  at = which.max(traj$I)
  peakI = list(value = traj$I[[at]], time = traj$time[[at]])

  # Recovered at the end of the run
  at = nrow(traj)
  finalR = list(value = traj$R[[at]], time = traj$time[[at]])

  list(minS = minS, peakE = peakE, peakI = peakI, finalR = finalR)
}

#' Final epidemic size of the SIR model
#'
#' Solves `z = 1 - exp(-R0 * z)` by fixed-point iteration.
#'
#' @param R0 Basic reproduction number.
#' @param tol Convergence tolerance.
#' @param maxIter Maximum number of iterations.
#' @return The fraction of the population infected over the epidemic.
#' @export
finalSize = function(R0, tol = 1e-10, maxIter = 1000L) {
  if (R0 <= 1) {
    return(0)
  }
  z = 0.5
  for (iter in seq_len(maxIter)) {
    zNew = 1 - exp(-R0 * z)
    if (abs(zNew - z) < tol) {
      return(zNew)
    }
    z = zNew
  }
  warning("`finalSize()` did not converge.", call. = FALSE)
  z
}

#' Validate a trajectory data frame
#'
#' @param traj A data frame with columns `time`, `S`, `E`, `I` and `R`.
#' @param tol Tolerance for the conservation of the population size.
#' @return `traj`, invisibly.
#' @keywords internal
checkTrajectory = function(traj, tol = 1e-6) {
  columns = c("time", "S", "E", "I", "R")
  absent = setdiff(columns, names(traj))
  if (length(absent) > 0L) {
    stop(sprintf("Trajectory is missing column `%s`.", absent[[1L]]), call. = FALSE)
  }
  if (is.unsorted(traj$time, strictly = TRUE)) {
    stop("Column `time` must be strictly increasing.", call. = FALSE)
  }
  # Check `S`
  if (anyNA(traj$S) || any(traj$S < 0)) {
    stop(sprintf("Column `%s` must be non-negative and complete.", "S"), call. = FALSE)
  }
  # Check `E`
  if (anyNA(traj$E) || any(traj$E < 0)) {
    stop(sprintf("Column `%s` must be non-negative and complete.", "E"), call. = FALSE)
  }
  # Check `I`
  if (anyNA(traj$I) || any(traj$I < 0)) {
    stop(sprintf("Column `%s` must be non-negative and complete.", "I"), call. = FALSE)
  }
  # Check `R`
  if (anyNA(traj$R) || any(traj$R < 0)) {
    stop(sprintf("Column `%s` must be non-negative and complete.", "R"), call. = FALSE)
  }

  total = traj$S + traj$E + traj$I + traj$R
  if (diff(range(total)) > tol * total[[1L]]) {
    stop("The population size is not conserved.", call. = FALSE)
  }
  invisible(traj)
}
