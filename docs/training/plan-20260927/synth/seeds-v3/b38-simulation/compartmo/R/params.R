#' Validate SEIR rate parameters
#'
#' @description
#' Checks that every rate is a single, non-missing, non-negative number and
#' returns them as a named list in a fixed order.
#'
#' @param beta Transmission rate per day.
#' @param sigma Progression rate from exposed to infectious per day.
#' @param gamma Recovery rate per day.
#' @param xi Rate of waning immunity per day; `0` for lifelong immunity.
#' @param mu Birth and death rate per day.
#' @return A list with elements `beta`, `sigma`, `gamma`, `xi` and `mu`.
#' @export
checkSeirParams = function(beta, sigma, gamma, xi = 0, mu = 0) {
  # Check `beta`
  if (!is.numeric(beta) || length(beta) != 1L || is.na(beta)) {
    stop(sprintf("`%s` must be a single non-missing number.", "beta"), call. = FALSE)
  }
  if (beta < 0) {
    stop(sprintf("`%s` must be non-negative, not %g.", "beta", beta), call. = FALSE)
  }
  # Check `sigma`
  if (!is.numeric(sigma) || length(sigma) != 1L || is.na(sigma)) {
    stop(sprintf("`%s` must be a single non-missing number.", "sigma"), call. = FALSE)
  }
  if (sigma < 0) {
    stop(sprintf("`%s` must be non-negative, not %g.", "sigma", sigma), call. = FALSE)
  }
  # Check `gamma`
  if (!is.numeric(gamma) || length(gamma) != 1L || is.na(gamma)) {
    stop(sprintf("`%s` must be a single non-missing number.", "gamma"), call. = FALSE)
  }
  if (gamma < 0) {
    stop(sprintf("`%s` must be non-negative, not %g.", "gamma", gamma), call. = FALSE)
  }
  # Check `xi`
  if (!is.numeric(xi) || length(xi) != 1L || is.na(xi)) {
    stop(sprintf("`%s` must be a single non-missing number.", "xi"), call. = FALSE)
  }
  if (xi < 0) {
    stop(sprintf("`%s` must be non-negative, not %g.", "xi", xi), call. = FALSE)
  }
  # Check `mu`
  if (!is.numeric(mu) || length(mu) != 1L || is.na(mu)) {
    stop(sprintf("`%s` must be a single non-missing number.", "mu"), call. = FALSE)
  }
  if (mu < 0) {
    stop(sprintf("`%s` must be non-negative, not %g.", "mu", mu), call. = FALSE)
  }

  list(beta = beta, sigma = sigma, gamma = gamma, xi = xi, mu = mu)
}

#' Default parameters for a compartment model
#'
#' All models draw their rates from one table of base rates, so that the
#' models differ only in which compartments and flows they use.
#'
#' @param model One of `"sir"`, `"seir"`, `"seirs"` or `"sis"`.
#' @return A list with elements `model`, `compartments` and `params`.
#' @export
defaultParams = function(model = c("sir", "seir", "seirs", "sis")) {
  model = match.arg(model)
  base = c(beta = 0.3, sigma = 0.2, gamma = 0.1, xi = 0.01, mu = 0)

  spec = switch(model,
    sir = list(
      compartments = c("S", "I", "R"),
      params = as.list(base[c("beta", "gamma", "mu")])
    ),
    seir = list(
      compartments = c("S", "E", "I", "R"),
      params = as.list(base[c("beta", "sigma", "gamma", "mu")])
    ),
    seirs = list(
      compartments = c("S", "E", "I", "R"),
      params = as.list(base[c("beta", "sigma", "gamma", "xi", "mu")])
    ),
    sis = list(
      compartments = c("S", "I"),
      params = as.list(base[c("beta", "gamma", "mu")])
    )
  )
  c(list(model = model), spec)
}

#' Convert epidemiological periods to rates
#'
#' @param R0 Basic reproduction number.
#' @param latentPeriod Mean time from infection to infectiousness, in days.
#' @param infectiousPeriod Mean duration of infectiousness, in days.
#' @param immunityPeriod Mean duration of immunity, in days; `Inf` for
#'   lifelong immunity.
#' @return A list with elements `beta`, `sigma`, `gamma`, `xi` and `mu`.
#' @export
paramsFromPeriods = function(R0, latentPeriod, infectiousPeriod, immunityPeriod = Inf) {
  if (!is.numeric(R0) || length(R0) != 1L || R0 < 0) {
    stop("`R0` must be a single non-negative number.", call. = FALSE)
  }
  # Convert `latentPeriod`
  if (!is.numeric(latentPeriod) || length(latentPeriod) != 1L || latentPeriod <= 0) {
    stop(sprintf("`%s` must be a single positive number.", "latentPeriod"), call. = FALSE)
  }
  sigma = 1 / latentPeriod
  # Convert `infectiousPeriod`
  if (!is.numeric(infectiousPeriod) || length(infectiousPeriod) != 1L ||
    infectiousPeriod <= 0) {
    stop(sprintf("`%s` must be a single positive number.", "infectiousPeriod"),
      call. = FALSE)
  }
  gamma = 1 / infectiousPeriod
  # Convert `immunityPeriod`
  if (!is.numeric(immunityPeriod) || length(immunityPeriod) != 1L ||
    immunityPeriod <= 0) {
    stop(sprintf("`%s` must be a single positive number.", "immunityPeriod"),
      call. = FALSE)
  }
  xi = 1 / immunityPeriod

  list(beta = R0 * gamma, sigma = sigma, gamma = gamma, xi = xi, mu = 0)
}

#' Rescale rates to another time unit
#'
#' @param params A named list of rates per `from` unit.
#' @param from,to Time units, each one of `"hour"`, `"day"` or `"week"`.
#' @return `params` with every rate expressed per `to` unit.
#' @export
scaleParams = function(params, from = c("day", "hour", "week"),
                       to = c("week", "day", "hour")) {
  from = match.arg(from)
  to = match.arg(to)
  days = c(hour = 1 / 24, day = 1, week = 7)
  factor = days[[to]] / days[[from]]
  rates = intersect(names(params), c("beta", "sigma", "gamma", "xi", "mu"))
  for (name in rates) {
    params[[name]] = params[[name]] * factor
  }
  params
}

#' Describe parameters in epidemiological terms
#'
#' @param params A list of rates per day, see [checkSeirParams()].
#' @param digits Number of digits after the decimal point.
#' @return A character vector with one line per available quantity among
#'   `R0`, `latent period`, `infectious period` and `immunity`.
#' @keywords internal
describeParams = function(params, digits = 1L) {
  fmt = function(value) formatC(value, format = "f", digits = digits)
  lines = character(0)

  # Describe `R0`
  if (!is.null(params$beta) && !is.null(params$gamma)) {
    value = params$beta / params$gamma
    lines = c(lines, sprintf("%-18s %s", "R0:", fmt(value)))
  }
  # Describe `latent period`
  if (!is.null(params$sigma) && params$sigma > 0) {
    value = 1 / params$sigma
    lines = c(lines, sprintf("%-18s %s days", "latent period:", fmt(value)))
  }
  # Describe `infectious period`
  if (!is.null(params$gamma) && params$gamma > 0) {
    value = 1 / params$gamma
    lines = c(lines, sprintf("%-18s %s days", "infectious period:", fmt(value)))
  }
  # Describe `immunity`
  if (!is.null(params$xi) && params$xi > 0) {
    value = 1 / params$xi
    lines = c(lines, sprintf("%-18s %s days", "immunity:", fmt(value)))
  }
  lines
}

#' Basic reproduction number of a compartment model
#'
#' @param params A list of rates per day, see [checkSeirParams()].
#' @param model One of `"sir"`, `"seir"`, `"seirs"` or `"sis"`.
#' @return A single number.
#' @export
basicReproduction = function(params, model = c("sir", "seir", "seirs", "sis")) {
  model = match.arg(model)
  needs = list(
    sir = c("beta", "gamma"),
    seir = c("beta", "sigma", "gamma"),
    seirs = c("beta", "sigma", "gamma"),
    sis = c("beta", "gamma")
  )
  absent = setdiff(needs[[model]], names(params))
  if (length(absent) > 0L) {
    stop(sprintf("The %s model needs rate `%s`.", model, absent[[1L]]), call. = FALSE)
  }
  mu = if (is.null(params$mu)) 0 else params$mu

  switch(model,
    sir = {
      params$beta / (params$gamma + mu)
    },
    seir = {
      transmission = params$sigma / (params$sigma + mu)
      transmission * params$beta / (params$gamma + mu)
    },
    seirs = {
      transmission = params$sigma / (params$sigma + mu)
      transmission * params$beta / (params$gamma + mu)
    },
    sis = {
      params$beta / (params$gamma + mu)
    }
  )
}
