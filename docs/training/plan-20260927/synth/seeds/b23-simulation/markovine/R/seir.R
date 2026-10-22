#' SEIR model derivatives
#'
#' Right-hand side of the closed-population SEIR model in the form expected
#' by [deSolve::ode()].
#'
#' @param time Current time, unused because the model is autonomous.
#' @param state Named numeric vector of compartment sizes.
#' @param parms Named list or vector with `beta`, `sigma` and `gamma`.
#' @return A list whose first element is the vector of derivatives.
#' @family seir models
#' @keywords internal
seir.derivs <- function(time, state, parms) {
  with(as.list(c(state, parms)), {
    N <- S + E + I + R
    infection <- beta * S * I / N
    dS <- -infection
    dE <- infection - sigma * E
    dI <- sigma * E - gamma * I
    dR <- gamma * I
    return(list(c(dS, dE, dI, dR), incidence = sigma * E))
  })
}

#' SEIRS model derivatives
#'
#' Like [seir.derivs()] but recovered individuals lose immunity at rate
#' `omega` and return to the susceptible compartment.
#'
#' @param time Current time, unused because the model is autonomous.
#' @param state Named numeric vector of compartment sizes.
#' @param parms Named list or vector with `beta`, `sigma`, `gamma` and
#'   `omega`.
#' @return A list whose first element is the vector of derivatives.
#' @family seir models
#' @keywords internal
seirs.derivs <- function(time, state, parms) {
  with(as.list(c(state, parms)), {
    N <- S + E + I + R
    infection <- beta * S * I / N
    dS <- -infection + omega * R
    dE <- infection - sigma * E
    dI <- sigma * E - gamma * I
    dR <- gamma * I - omega * R
    return(list(c(dS, dE, dI, dR), incidence = sigma * E))
  })
}

#' Initial state for an SEIR model
#'
#' @param N Total population size.
#' @param I0 Initial number of infectious individuals.
#' @param E0 Initial number of exposed individuals.
#' @param R0.immune Initial number of immune individuals.
#' @return A named numeric vector with elements `S`, `E`, `I` and `R`.
#' @family seir models
#' @export
seir.initial <- function(N, I0 = 1, E0 = 0, R0.immune = 0) {
  S0 <- N - I0 - E0 - R0.immune
  if (S0 < 0) {
    stop("Initial compartments exceed the population size.", call. = FALSE)
  }
  return(c(S = S0, E = E0, I = I0, R = R0.immune))
}

#' Solve an SEIR model
#'
#' Integrates the SEIR or SEIRS equations with [deSolve::ode()] and returns
#' a tidy data frame of compartment trajectories.
#'
#' @param init Named numeric vector of initial compartment sizes, see
#'   [seir.initial()].
#' @param parms Named list of rate parameters.
#' @param times Numeric vector of output times.
#' @param model Model structure: `"seir"` or `"seirs"`.
#' @param method Integration method passed to [deSolve::ode()].
#' @return A data frame with columns `time`, `S`, `E`, `I`, `R` and
#'   `incidence`.
#' @family seir models
#' @export
seir.solve <- function(init, parms, times = seq(0, 200, by = 1),
                       model = c("seir", "seirs"), method = "lsoda") {
  model <- match.arg(model)
  needed <- switch(model,
    seir = c("beta", "sigma", "gamma"),
    seirs = c("beta", "sigma", "gamma", "omega")
  )
  missing.parms <- setdiff(needed, names(parms))
  if (length(missing.parms) > 0) {
    stop("Missing parameters: ", paste(missing.parms, collapse = ", "), call. = FALSE)
  }
  func <- switch(model,
    seir = seir.derivs,
    seirs = seirs.derivs
  )
  out <- deSolve::ode(
    y = init[c("S", "E", "I", "R")],
    times = times,
    func = func,
    parms = parms[needed],
    method = method
  )
  out <- as.data.frame(out)
  out[c("S", "E", "I", "R")] <- lapply(out[c("S", "E", "I", "R")], pmax, 0)
  return(out)
}

#' Basic reproduction number of an SEIR model
#'
#' @param parms Named list of rate parameters.
#' @return The basic reproduction number `beta / gamma`.
#' @family seir models
#' @export
seir.r0 <- function(parms) {
  return(parms$beta / parms$gamma)
}

#' Final epidemic size
#'
#' Solves the final size equation `z = 1 - exp(-R0 * z)` for the fraction
#' of an initially fully susceptible population that is eventually
#' infected.
#'
#' @param r0 Basic reproduction number.
#' @return The final attack rate between zero and one.
#' @family seir models
#' @export
seir.final.size <- function(r0) {
  if (r0 <= 1) {
    return(0)
  }
  f <- function(z) z - 1 + exp(-r0 * z)
  return(stats::uniroot(f, lower = 1e-9, upper = 1, tol = 1e-12)$root)
}

#' Summarise an SEIR trajectory
#'
#' Extracts peak timing, peak prevalence and the cumulative attack rate from
#' a solved trajectory.
#'
#' @param traj A data frame as returned by [seir.solve()].
#' @return A one-row data frame of epidemic summary measures.
#' @family seir models
#' @export
seir.summary <- function(traj) {
  N <- traj$S[1] + traj$E[1] + traj$I[1] + traj$R[1]
  peak <- which.max(traj$I)
  last <- nrow(traj)
  return(data.frame(
    peak.time = traj$time[peak],
    peak.prevalence = traj$I[peak] / N,
    peak.incidence = max(traj$incidence),
    attack.rate = (traj$S[1] - traj$S[last]) / N,
    end.prevalence = traj$I[last] / N
  ))
}

#' Fit the transmission rate to incidence data
#'
#' Finds the `beta` that minimises the Poisson deviance between observed
#' daily case counts and the model incidence, holding the other rates
#' fixed.
#'
#' @param cases Integer vector of observed daily case counts.
#' @param init Named numeric vector of initial compartment sizes, see
#'   [seir.initial()].
#' @param parms Named list of rate parameters.
#' @param interval Search interval for `beta`.
#' @return A list with elements `beta`, `deviance` and `fitted`.
#' @family seir models
#' @export
seir.fit.beta <- function(cases, init, parms, interval = c(0.01, 3)) {
  times <- seq(0, length(cases), by = 1)
  deviance.at <- function(beta) {
    p <- utils::modifyList(parms, list(beta = beta))
    traj <- tryCatch(
      seir.solve(init, p, times = times),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    if (is.null(traj)) {
      return(Inf)
    }
    mu <- pmax(diff(-traj$S), 1e-9)
    return(2 * sum(ifelse(cases > 0, cases * log(cases / mu), 0) - (cases - mu)))
  }
  opt <- stats::optimize(deviance.at, interval = interval)
  best <- utils::modifyList(parms, list(beta = opt$minimum))
  traj <- seir.solve(init, best, times = times)
  return(list(
    beta = opt$minimum,
    deviance = opt$objective,
    fitted = diff(-traj$S)
  ))
}
