#' Transmission kernels
#'
#' Map the distance between an infectious and a susceptible agent to a
#' per-contact infection probability. All kernels are cut off at the contact
#' radius: the constant kernel is flat inside it, the exponential kernel
#' decays with distance and the Gaussian kernel is a bump of width
#' `radius / 2`.
#'
#' @param dist Numeric vector of distances between infectious and
#'   susceptible agents.
#' @param beta Per-contact transmission probability at distance zero.
#' @param radius Contact radius in arena units.
#' @return A numeric vector of infection probabilities.
#' @keywords internal
kernel_constant <- function(dist, beta, radius) {
  ifelse(dist <= radius, beta, 0)
}

#' @rdname kernel_constant
kernel_exponential <- function(dist, beta, radius) {
  ifelse(dist <= radius, beta * exp(-dist / radius), 0)
}

#' @rdname kernel_constant
kernel_gaussian <- function(dist, beta, radius) {
  ifelse(dist <= radius, beta * exp(-(dist^2) / (2 * (radius / 2)^2)), 0)
}

#' Expose susceptible agents to infectious neighbours
#'
#' For every susceptible agent, combines the infection probabilities from
#' all infectious agents in range and draws whether the agent becomes
#' exposed this step.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param beta Per-contact transmission probability at distance zero.
#' @param radius Contact radius in arena units.
#' @param kernel Shape of the distance kernel: `"constant"`,
#'   `"exponential"` or `"gaussian"`.
#' @return The population with newly exposed agents moved to state `"E"`.
#' @export
expose_contacts <- function(pop, beta = 0.2, radius = 2,
                            kernel = c("constant", "exponential", "gaussian")) {
  kernel <- match.arg(kernel)
  kernel_fun <- switch(kernel,
    constant = kernel_constant,
    exponential = kernel_exponential,
    gaussian = kernel_gaussian
  )

  sus <- which(pop$state == "S")
  inf <- which(pop$state == "I")
  if (length(sus) == 0 || length(inf) == 0) {
    return(pop)
  }

  escape <- numeric(length(sus))
  for (k in seq_along(sus)) {
    dx <- pop$x[inf] - pop$x[sus[k]]
    dy <- pop$y[inf] - pop$y[sus[k]]
    p <- kernel_fun(sqrt(dx^2 + dy^2), beta, radius)
    # probability of escaping every infectious contact
    escape[k] <- prod(1 - p)
  }

  infected <- stats::runif(length(sus)) > escape
  if (any(infected)) {
    newly <- sus[infected]
    pop$state[newly] <- "E"
    pop$days_in_state[newly] <- 0L
  }
  pop
}

#' Progress exposed agents to infectious
#'
#' Each exposed agent becomes infectious with a fixed daily probability,
#' giving a geometric latent period.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param rate Daily probability of leaving the current state.
#' @return The population with progressed agents moved to state `"I"`.
#' @export
progress_exposed <- function(pop, rate = 1 / 3) {
  idx <- which(pop$state == "E")
  if (length(idx) == 0) {
    return(pop)
  }

  moves <- idx[stats::runif(length(idx)) < rate]
  pop$state[moves] <- "I"
  pop$days_in_state[moves] <- 0L
  pop
}

#' Progress infectious agents to recovered
#'
#' Each infectious agent recovers with a fixed daily probability, giving a
#' geometric infectious period.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param rate Daily probability of leaving the current state.
#' @return The population with progressed agents moved to state `"R"`.
#' @export
progress_infectious <- function(pop, rate = 1 / 5) {
  idx <- which(pop$state == "I")
  if (length(idx) == 0) {
    return(pop)
  }

  moves <- idx[stats::runif(length(idx)) < rate]
  pop$state[moves] <- "R"
  pop$days_in_state[moves] <- 0L
  pop
}

#' Advance the population by one day
#'
#' Runs movement, transmission and disease progression in that order and
#' increments the time every agent has spent in its state.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param params A named list of model parameters, see [run_herd()].
#' @return The population after one simulated day.
#' @export
step_population <- function(pop, params) {
  # Age first so agents that change state below restart at zero.
  pop$days_in_state <- pop$days_in_state + 1L

  pop <- move_random(pop, step_sd = params$step_sd, boundary = params$boundary)
  pop <- expose_contacts(pop,
    beta = params$beta,
    radius = params$radius,
    kernel = params$kernel
  )

  # Disease progression comes last, after contacts for the day are drawn.
  pop <- progress_exposed(pop, rate = params$sigma)
  pop <- progress_infectious(pop, rate = params$gamma)
  pop
}

#' Run an agent-based SEIR simulation
#'
#' Steps a population forward for a number of days, recording compartment
#' counts after every step. Stops early once no exposed or infectious
#' agents remain.
#'
#' @param pop A data frame of agents as returned by [new_population()].
#' @param days Maximum number of days to simulate.
#' @param params A named list of model parameters. Missing entries are
#'   filled from the defaults `beta = 0.2`, `radius = 2`, `kernel =
#'   "constant"`, `sigma = 1/3`, `gamma = 1/5`, `step_sd = 1`, `boundary =
#'   "wrap"`.
#' @param keep_snapshots Whether to store the full population at each step.
#' @return A list with elements `counts` (a data frame of daily counts),
#'   `final` (the last population) and `snapshots` (a list or `NULL`).
#' @export
run_herd <- function(pop, days = 100, params = list(), keep_snapshots = FALSE) {
  validate_population(pop)
  defaults <- list(
    beta = 0.2, radius = 2, kernel = "constant", sigma = 1 / 3,
    gamma = 1 / 5, step_sd = 1, boundary = "wrap"
  )
  params <- utils::modifyList(defaults, params)

  # Row 1 holds the initial state, row d + 1 the state after day d.
  counts <- matrix(NA_integer_, nrow = days + 1, ncol = 4,
    dimnames = list(NULL, c("S", "E", "I", "R")))
  counts[1, ] <- population_counts(pop)
  snapshots <- if (keep_snapshots) vector("list", days + 1) else NULL
  if (keep_snapshots) {
    snapshots[[1]] <- pop
  }

  last_day <- days
  for (day in seq_len(days)) {
    pop <- step_population(pop, params)
    counts[day + 1, ] <- population_counts(pop)
    if (keep_snapshots) {
      snapshots[[day + 1]] <- pop
    }
    if (counts[day + 1, "E"] + counts[day + 1, "I"] == 0) {
      last_day <- day
      break
    }
  }

  # Trim the unused rows left over from an early stop.
  rows <- seq_len(last_day + 1)
  counts_df <- data.frame(day = rows - 1L, counts[rows, , drop = FALSE])
  if (keep_snapshots) {
    snapshots <- snapshots[rows]
  }
  list(counts = counts_df, final = pop, snapshots = snapshots)
}
