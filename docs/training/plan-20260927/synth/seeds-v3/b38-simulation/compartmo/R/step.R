#' Look up a rate with a default
#'
#' @param params A list of rates per day.
#' @param name Name of the rate.
#' @return The rate, or `0` when `params` has no element `name`.
#' @noRd
rateOf = function(params, name) {
  value = params[[name]]
  if (is.null(value)) 0 else value
}

#' Derivatives of the SEIRS equations
#'
#' The four compartments are written in parallel; every flow that leaves one
#' compartment enters the next one.
#'
#' @param state A named numeric vector with elements `S`, `E`, `I` and `R`.
#' @param params A list of rates per day, see [checkSeirParams()].
#' @return A named numeric vector with elements `S`, `E`, `I` and `R`.
#' @export
seirDerivs = function(state, params) {
  missingStates = setdiff(c("S", "E", "I", "R"), names(state))
  if (length(missingStates) > 0L) {
    stop(sprintf("`state` is missing compartment `%s`.", missingStates[[1L]]), call. = FALSE)
  }
  N = sum(state)
  S = state[["S"]]
  E = state[["E"]]
  I = state[["I"]]
  R = state[["R"]]
  mu = rateOf(params, "mu")
  infection = rateOf(params, "beta") * S * I / N

  # S: births and waning in, infection and deaths out
  dS = mu * N - infection + rateOf(params, "xi") * R - mu * S
  # E: infection in, progression and deaths out
  dE = infection - rateOf(params, "sigma") * E - mu * E
  # I: progression in, recovery and deaths out
  dI = rateOf(params, "sigma") * E - rateOf(params, "gamma") * I - mu * I
  # R: recovery in, waning and deaths out
  dR = rateOf(params, "gamma") * I - rateOf(params, "xi") * R - mu * R
  c(S = dS, E = dE, I = dI, R = dR)
}

#' Deterministic update steps
#'
#' Both schemes integrate the equations of [seirDerivs()], including births
#' and deaths at rate `mu`.
#'
#' @inheritParams seirDerivs
#' @param dt Step length in days.
#' @return The state after one step, a named numeric vector with elements
#'   `S`, `E`, `I` and `R`.
#' @rdname deterministic-steps
#' @export
eulerStep = function(state, params, dt) {
  N = sum(state)
  mu = rateOf(params, "mu")
  flows = c(
    infection = rateOf(params, "beta") * state[["S"]] * state[["I"]] / N,
    progression = rateOf(params, "sigma") * state[["E"]],
    recovery = rateOf(params, "gamma") * state[["I"]],
    waning = rateOf(params, "xi") * state[["R"]]
  ) * dt
  births = mu * N * dt
  deaths = mu * state * dt

  # Update S, E, I and R in parallel from the same flows
  out = state
  out[["S"]] = state[["S"]] + births - flows[["infection"]] + flows[["waning"]] - deaths[["S"]]
  out[["E"]] = state[["E"]] + flows[["infection"]] - flows[["progression"]] - deaths[["E"]]
  out[["I"]] = state[["I"]] + flows[["progression"]] - flows[["recovery"]] - deaths[["I"]]
  out[["R"]] = state[["R"]] + flows[["recovery"]] - flows[["waning"]] - deaths[["R"]]
  pmax(out, 0)
}

#' @rdname deterministic-steps
#' @export
rk4Step = function(state, params, dt) {
  k1 = seirDerivs(state, params)
  k2 = seirDerivs(state + dt / 2 * k1, params)
  k3 = seirDerivs(state + dt / 2 * k2, params)
  k4 = seirDerivs(state + dt * k3, params)
  out = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  pmax(out, 0)
}

#' Stochastic update steps
#'
#' `binomialStep()` draws each flow as a binomial count with the exit
#' probability `1 - exp(-rate * dt)`; `tauLeapStep()` draws Poisson counts
#' and caps them at the size of the source compartment.
#'
#' @inheritParams seirDerivs
#' @param dt Step length in days.
#' @return The state after one step, a named integer vector with elements
#'   `S`, `E`, `I` and `R`.
#' @rdname stochastic-steps
#' @export
binomialStep = function(state, params, dt) {
  N = sum(state)
  state = round(state)
  probs = 1 - exp(-dt * c(
    infection = rateOf(params, "beta") * state[["I"]] / N,
    progression = rateOf(params, "sigma"),
    recovery = rateOf(params, "gamma"),
    waning = rateOf(params, "xi")
  ))

  # Draw every flow from the compartment it leaves
  infection = stats::rbinom(1L, state[["S"]], probs[["infection"]])
  progression = stats::rbinom(1L, state[["E"]], probs[["progression"]])
  recovery = stats::rbinom(1L, state[["I"]], probs[["recovery"]])
  waning = stats::rbinom(1L, state[["R"]], probs[["waning"]])

  out = state
  out[["S"]] = state[["S"]] - infection + waning
  out[["E"]] = state[["E"]] + infection - progression
  out[["I"]] = state[["I"]] + progression - recovery
  out[["R"]] = state[["R"]] + recovery - waning
  storage.mode(out) = "integer"
  out
}

#' @rdname stochastic-steps
#' @export
tauLeapStep = function(state, params, dt) {
  N = sum(state)
  state = round(state)
  means = dt * c(
    infection = rateOf(params, "beta") * state[["S"]] * state[["I"]] / N,
    progression = rateOf(params, "sigma") * state[["E"]],
    recovery = rateOf(params, "gamma") * state[["I"]],
    waning = rateOf(params, "xi") * state[["R"]]
  )

  # Draw every flow and cap it at the compartment it leaves
  infection = min(stats::rpois(1L, means[["infection"]]), state[["S"]])
  progression = min(stats::rpois(1L, means[["progression"]]), state[["E"]])
  recovery = min(stats::rpois(1L, means[["recovery"]]), state[["I"]])
  waning = min(stats::rpois(1L, means[["waning"]]), state[["R"]])

  out = state
  out[["S"]] = state[["S"]] - infection + waning
  out[["E"]] = state[["E"]] + infection - progression
  out[["I"]] = state[["I"]] + progression - recovery
  out[["R"]] = state[["R"]] + recovery - waning
  storage.mode(out) = "integer"
  out
}

#' Advance a compartment state by one step
#'
#' @inheritParams seirDerivs
#' @param dt Step length in days.
#' @param method Update scheme: `"euler"`, `"rk4"`, `"binomial"` or
#'   `"tauleap"`.
#' @return The state after one step.
#' @export
stepModel = function(state, params, dt = 1, method = c("euler", "rk4", "binomial", "tauleap")) {
  method = match.arg(method)
  missingStates = setdiff(c("S", "E", "I", "R"), names(state))
  if (length(missingStates) > 0L) {
    stop(sprintf("`state` is missing compartment `%s`.", missingStates[[1L]]), call. = FALSE)
  }
  if (!is.numeric(dt) || length(dt) != 1L || dt <= 0) {
    stop("`dt` must be a single positive number.", call. = FALSE)
  }
  state = state[c("S", "E", "I", "R")]

  switch(method,
    euler = {
      eulerStep(state, params, dt)
    },
    rk4 = {
      rk4Step(state, params, dt)
    },
    binomial = {
      binomialStep(state, params, dt)
    },
    tauleap = {
      tauLeapStep(state, params, dt)
    }
  )
}
