#' Wilson score interval for a proportion
#'
#' @param x Number of events.
#' @param n Number of trials.
#' @param conf.level Confidence level of the interval.
#' @return A two-column matrix with `lower` and `upper` bounds.
#' @noRd
wilson.interval <- function(x, n, conf.level = 0.95) {
  z <- stats::qnorm(1 - (1 - conf.level) / 2)
  p <- x / n

  # Wilson behaves far better than the Wald interval near 0 and 1
  denom <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / denom
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom

  cbind(lower = pmax(0, centre - half), upper = pmin(1, centre + half))
}

#' Attack rate with confidence interval
#'
#' Proportion of an exposed population that became a case during an outbreak.
#'
#' @param cases Number of cases.
#' @param population Size of the population at risk.
#' @param conf.level Confidence level of the interval.
#' @export
#' @return A data frame with columns `estimate`, `lower` and `upper`.
attack.rate <- function(cases, population, conf.level = 0.95) {
  if (any(cases > population)) {
    stop("`cases` cannot exceed `population`.", call. = FALSE)
  }

  ci <- wilson.interval(cases, population, conf.level)
  data.frame(estimate = cases / population, lower = ci[, "lower"], upper = ci[, "upper"])
}

#' Case fatality ratio with confidence interval
#'
#' Proportion of cases with a known outcome that died.
#'
#' @param deaths Number of deaths among cases.
#' @param cases Number of cases with a known outcome.
#' @param conf.level Confidence level of the interval.
#' @export
#' @return A data frame with columns `estimate`, `lower` and `upper`.
case.fatality.ratio <- function(deaths, cases, conf.level = 0.95) {
  if (any(deaths > cases)) {
    stop("`deaths` cannot exceed `cases`.", call. = FALSE)
  }

  ci <- wilson.interval(deaths, cases, conf.level)
  data.frame(estimate = deaths / cases, lower = ci[, "lower"], upper = ci[, "upper"])
}

#' Incidence rate with confidence interval
#'
#' Cases per unit of person-time, with an exact Poisson interval.
#'
#' @param cases Number of cases.
#' @param person.time Total person-time at risk.
#' @param per Multiplier for reporting, e.g. `1e5` for rates per 100,000.
#' @param conf.level Confidence level of the interval.
#' @export
#' @return A data frame with columns `estimate`, `lower` and `upper`.
incidence.rate <- function(cases, person.time, per = 1e5, conf.level = 0.95) {
  if (any(person.time <= 0)) {
    stop("`person.time` must be positive.", call. = FALSE)
  }

  # Exact (Garwood) Poisson limits via the chi-squared distribution
  alpha <- 1 - conf.level
  lower <- ifelse(cases == 0, 0, stats::qchisq(alpha / 2, 2 * cases) / 2)
  upper <- stats::qchisq(1 - alpha / 2, 2 * (cases + 1)) / 2

  data.frame(
    estimate = cases / person.time * per,
    lower = lower / person.time * per,
    upper = upper / person.time * per
  )
}

#' Advance an SIR model by one time step
#'
#' Discrete-time update of susceptible, infectious and recovered counts.
#'
#' @param state A named numeric vector with elements `S`, `I` and `R`.
#' @param beta Transmission rate per time step.
#' @param gamma Recovery rate per time step.
#' @return A named numeric vector with the updated `S`, `I` and `R`.
#' @keywords internal
sir.step <- function(state, beta, gamma) {
  N <- sum(state)

  # Cap both flows so no compartment can go negative with large rates
  new.inf <- min(beta * state[["S"]] * state[["I"]] / N, state[["S"]])
  new.rec <- min(gamma * state[["I"]], state[["I"]] + new.inf)

  c(
    S = state[["S"]] - new.inf,
    I = state[["I"]] + new.inf - new.rec,
    R = state[["R"]] + new.rec
  )
}

#' Simulate an SIR epidemic
#'
#' Iterates [sir.step()] from an initial state and records the trajectory.
#'
#' @param N Total population size.
#' @param I0 Initial number of infectious individuals.
#' @param beta Transmission rate per time step.
#' @param gamma Recovery rate per time step.
#' @param n.steps Number of time steps to simulate.
#' @return A data frame with columns `time`, `S`, `I`, `R` and `incidence`.
#' @export
sir.simulate <- function(N, I0 = 1, beta, gamma, n.steps = 100) {
  if (I0 < 1 || I0 > N) {
    stop("`I0` must be between 1 and `N`.", call. = FALSE)
  }

  out <- matrix(NA_real_, nrow = n.steps + 1, ncol = 3, dimnames = list(NULL, c("S", "I", "R")))
  out[1, ] <- c(N - I0, I0, 0)

  for (t in seq_len(n.steps)) {
    out[t + 1, ] <- sir.step(out[t, ], beta, gamma)
  }

  res <- data.frame(time = 0:n.steps, out)
  # New infections per step are the drop in susceptibles
  res$incidence <- c(NA, -diff(res$S))
  res
}

#' Basic reproduction number from an epidemic growth rate
#'
#' Converts an exponential growth rate into R0 under a chosen generation
#' interval distribution.
#'
#' @param r Exponential growth rate per day.
#' @param gi.mean Mean of the generation interval in days.
#' @param gi.sd Standard deviation of the generation interval in days; only
#'   used for the gamma distribution.
#' @param dist Generation interval distribution.
#' @return A numeric vector of R0 values the same length as `r`.
#' @export
r0.from.growth <- function(r, gi.mean, gi.sd = NULL, dist = c("exponential", "fixed", "gamma")) {
  dist <- match.arg(dist)
  if (dist == "gamma" && is.null(gi.sd)) {
    stop("`gi.sd` is required for a gamma generation interval.", call. = FALSE)
  }

  # Wallinga-Lipsitch: R0 = 1 / M(-r) for the generation interval density
  switch(dist,
    exponential = 1 + r * gi.mean,
    fixed = exp(r * gi.mean),
    gamma = {
      shape <- (gi.mean / gi.sd)^2
      rate <- gi.mean / gi.sd^2
      (1 + r / rate)^shape
    }
  )
}
